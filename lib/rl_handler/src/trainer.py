from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from src.models.frontier_policy import FrontierPolicyNetwork
from src.reward_metrics import (
    js_divergence,
    js_normalized,
    kl_divergence,
    listwise_loss,
    masked_mae,
    masked_rmse,
    masked_softmax,
    pairwise_ranking_loss,
    score_std_within_frontier,
)


class OnnxFrontierPolicyWrapper(nn.Module):
    def __init__(self, core: FrontierPolicyNetwork):
        super().__init__()
        self.core = core

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        membership: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.core(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            membership=membership,
            candidate_batch=None,
            mask=mask,
        )


class OnnxFrontierPolicySeparatedWrapper(nn.Module):
    def __init__(self, core: FrontierPolicyNetwork):
        super().__init__()
        self.core = core

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        membership: torch.Tensor,
        goal_node_features: torch.Tensor,
        goal_edge_index: torch.Tensor,
        goal_edge_attr: torch.Tensor,
        goal_batch: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.core(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            membership=membership,
            candidate_batch=None,
            mask=mask,
            goal_node_features=goal_node_features,
            goal_edge_index=goal_edge_index,
            goal_edge_attr=goal_edge_attr,
            goal_batch=goal_batch,
        )


class RLFrontierTrainer:
    def __init__(
        self,
        model: FrontierPolicyNetwork,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str | torch.device | None = None,
        reward_formulation: str = "negative_distance",
        m_failed_state: float = 100000.0,
        kind_of_data: str = "merged",
        track_rank_correlation: bool = True,
        reward_loss_weight: float = 1.0,
        ranking_loss_weight: float = 0.0,
        ranking_loss_type: str = "none",
        reward_temperature: float = 0.5,
    ):
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.reward_formulation = reward_formulation
        self.m_failed_state = float(m_failed_state)
        self.kind_of_data = kind_of_data
        self.track_rank_correlation = bool(track_rank_correlation)
        self.reward_loss_weight = float(reward_loss_weight)
        self.ranking_loss_weight = float(ranking_loss_weight)
        self.ranking_loss_type = str(ranking_loss_type)
        self.reward_temperature = float(reward_temperature)
        if self.ranking_loss_type not in {"none", "pairwise", "listwise"}:
            raise ValueError(
                f"Unsupported ranking_loss_type={self.ranking_loss_type}. "
                "Expected one of: none, pairwise, listwise."
            )
        if self.reward_temperature <= 0.0:
            raise ValueError("reward_temperature must be > 0.")

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    @staticmethod
    def _split_logits(logits: torch.Tensor, frontier_ptr: torch.Tensor) -> List[torch.Tensor]:
        chunks = []
        for i in range(frontier_ptr.numel() - 1):
            s = int(frontier_ptr[i].item())
            e = int(frontier_ptr[i + 1].item())
            chunks.append(logits[s:e])
        return chunks

    @staticmethod
    def _average_ranks(values: torch.Tensor) -> torch.Tensor:
        sorted_vals, sorted_idx = torch.sort(values)
        n = int(values.numel())
        ranks = torch.zeros(n, dtype=torch.float32, device=values.device)
        i = 0
        while i < n:
            j = i + 1
            while j < n and bool(
                torch.isclose(sorted_vals[j], sorted_vals[i], rtol=0.0, atol=1e-12).item()
            ):
                j += 1
            avg_rank = 0.5 * (float(i) + float(j - 1))
            ranks[sorted_idx[i:j]] = avg_rank
            i = j
        return ranks

    @classmethod
    def _spearman_rank_corr(cls, logits: torch.Tensor, rewards: torch.Tensor) -> float:
        if logits.numel() < 2:
            return 0.0
        rank_logits = cls._average_ranks(logits.detach())
        rank_rewards = cls._average_ranks(rewards.detach())
        x = rank_logits - rank_logits.mean()
        y = rank_rewards - rank_rewards.mean()
        denom = torch.sqrt((x * x).sum()) * torch.sqrt((y * y).sum())
        if float(denom.item()) <= 1e-12:
            return 0.0
        return float(((x * y).sum() / denom).item())

    def _build_model_kwargs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        kwargs: Dict[str, torch.Tensor] = {
            "node_features": batch["node_features"],
            "edge_index": batch["edge_index"],
            "edge_attr": batch["edge_attr"],
            "membership": batch["membership"],
            "candidate_batch": batch["candidate_batch"],
        }
        if "pool_node_index" in batch and "pool_membership" in batch:
            kwargs["pool_node_index"] = batch["pool_node_index"]
            kwargs["pool_membership"] = batch["pool_membership"]
        if "goal_node_features" in batch:
            kwargs["goal_node_features"] = batch["goal_node_features"]
            kwargs["goal_edge_index"] = batch["goal_edge_index"]
            kwargs["goal_edge_attr"] = batch["goal_edge_attr"]
            kwargs["goal_batch"] = batch["goal_batch"]
        return kwargs

    @staticmethod
    def _build_padded_frontier_scores(
        flat_scores: torch.Tensor,
        frontier_ptr: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        padded = flat_scores.new_zeros(mask.shape, dtype=flat_scores.dtype)
        for i in range(frontier_ptr.numel() - 1):
            s = int(frontier_ptr[i].item())
            e = int(frontier_ptr[i + 1].item())
            n = int(e - s)
            if n > 0:
                padded[i, :n] = flat_scores[s:e]
        return padded

    @staticmethod
    def _infer_frontier_mask(frontier_ptr: torch.Tensor) -> torch.Tensor:
        batch_size = int(frontier_ptr.numel() - 1)
        sizes = [
            int(frontier_ptr[i + 1].item() - frontier_ptr[i].item())
            for i in range(batch_size)
        ]
        max_n = max(sizes) if sizes else 0
        mask = torch.zeros((batch_size, max_n), dtype=torch.bool, device=frontier_ptr.device)
        for i, n in enumerate(sizes):
            if n > 0:
                mask[i, :n] = True
        return mask

    def _get_frontier_scores_rewards_mask(
        self,
        batch: Dict[str, torch.Tensor],
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frontier_ptr = batch["frontier_ptr"]
        mask = batch.get("frontier_mask")
        if mask is None:
            mask = self._infer_frontier_mask(frontier_ptr)
            rewards = self._build_padded_frontier_scores(batch["rewards"], frontier_ptr, mask)
        else:
            mask = mask.bool()
            rewards = batch.get("padded_rewards")
            if rewards is None:
                rewards = self._build_padded_frontier_scores(batch["rewards"], frontier_ptr, mask)
        pred_scores = self._build_padded_frontier_scores(logits, frontier_ptr, mask)
        return pred_scores, rewards, mask

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = self.model(**self._build_model_kwargs(batch))
        frontier_ptr = batch["frontier_ptr"]
        rewards = batch["rewards"]

        terms = []
        for i in range(frontier_ptr.numel() - 1):
            s = int(frontier_ptr[i].item())
            e = int(frontier_ptr[i + 1].item())
            probs = torch.softmax(logits[s:e], dim=0)
            frontier_rewards = rewards[s:e]
            terms.append((probs * frontier_rewards).sum())
        expected_reward = torch.stack(terms).mean()
        reward_loss = -expected_reward

        pred_scores, true_rewards, mask = self._get_frontier_scores_rewards_mask(batch, logits)
        if self.ranking_loss_type == "pairwise":
            rank_loss = pairwise_ranking_loss(pred_scores, true_rewards, mask)
        elif self.ranking_loss_type == "listwise":
            rank_loss = listwise_loss(
                pred_scores, true_rewards, mask, self.reward_temperature
            )
        else:
            rank_loss = reward_loss.new_tensor(0.0)

        total_loss = (
            self.reward_loss_weight * reward_loss
            + self.ranking_loss_weight * rank_loss
        )
        return {
            "total_loss": total_loss,
            "reward_loss": reward_loss,
            "ranking_loss": rank_loss,
            "expected_reward": expected_reward,
            "logits": logits,
        }

    def train(
        self,
        train_loader,
        eval_loader,
        n_epochs: int,
        checkpoint_dir: str,
        model_name: str = "frontier_policy",
        eval_every: int = 1,
    ) -> Dict[str, List[float]]:
        checkpoint = Path(checkpoint_dir)
        checkpoint.mkdir(parents=True, exist_ok=True)

        best_eval_reward = -float("inf")
        best_eval_epoch = -1
        history: Dict[str, List[float]] = {
            "train_reward": [],
            "train_total_loss": [],
            "train_reward_loss": [],
            "train_ranking_loss": [],
            "eval_epochs": [],
            "eval_reward": [],
            "eval_reward_std": [],
            "eval_regret": [],
            "eval_regret_std": [],
            "eval_oracle_accuracy": [],
            "eval_failure_choice_rate": [],
            "eval_failure_avoidance_accuracy": [],
            "eval_chosen_distance": [],
            "eval_chosen_distance_std": [],
            "eval_oracle_distance": [],
            "eval_oracle_distance_std": [],
            "eval_reward_top1_accuracy": [],
            "eval_reward_rank_correlation": [],
            "eval_reward_kl": [],
            "eval_reward_js": [],
            "eval_reward_js_normalized": [],
            "eval_reward_mae": [],
            "eval_reward_rmse": [],
            "eval_score_std_within_frontier": [],
        }
        eval_snapshots: List[Dict[str, object]] = []

        epoch_pbar = tqdm(range(n_epochs), desc="training", leave=True)
        for epoch in epoch_pbar:
            self.model.train()
            total_train_reward = 0.0
            total_train_total_loss = 0.0
            total_train_reward_loss = 0.0
            total_train_ranking_loss = 0.0
            batches = 0

            for raw_batch in train_loader:
                batch = self._move_to_device(raw_batch)
                self.optimizer.zero_grad()
                loss_terms = self.compute_loss(batch)
                loss = loss_terms["total_loss"]
                loss.backward()
                self.optimizer.step()

                total_train_reward += float(loss_terms["expected_reward"].detach().item())
                total_train_total_loss += float(loss_terms["total_loss"].detach().item())
                total_train_reward_loss += float(loss_terms["reward_loss"].detach().item())
                total_train_ranking_loss += float(loss_terms["ranking_loss"].detach().item())
                batches += 1

            mean_train_reward = total_train_reward / max(1, batches)
            mean_train_total_loss = total_train_total_loss / max(1, batches)
            mean_train_reward_loss = total_train_reward_loss / max(1, batches)
            mean_train_ranking_loss = total_train_ranking_loss / max(1, batches)
            history["train_reward"].append(mean_train_reward)
            history["train_total_loss"].append(mean_train_total_loss)
            history["train_reward_loss"].append(mean_train_reward_loss)
            history["train_ranking_loss"].append(mean_train_ranking_loss)

            if (epoch + 1) % eval_every == 0:
                metrics = self.evaluate(eval_loader)
                eval_epoch = epoch + 1
                history["eval_epochs"].append(eval_epoch)
                history["eval_reward"].append(metrics["mean_reward"])
                history["eval_reward_std"].append(metrics["std_reward"])
                history["eval_regret"].append(metrics["regret_mean"])
                history["eval_regret_std"].append(metrics["regret_std"])
                history["eval_oracle_accuracy"].append(metrics["oracle_accuracy"])
                history["eval_failure_choice_rate"].append(metrics["failure_choice_rate"])
                history["eval_failure_avoidance_accuracy"].append(
                    metrics["failure_avoidance_accuracy"]
                )
                history["eval_chosen_distance"].append(metrics["mean_chosen_distance"])
                history["eval_chosen_distance_std"].append(
                    metrics["std_chosen_distance"]
                )
                history["eval_oracle_distance"].append(metrics["mean_oracle_distance"])
                history["eval_oracle_distance_std"].append(
                    metrics["std_oracle_distance"]
                )
                history["eval_reward_top1_accuracy"].append(metrics["reward_top1_accuracy"])
                history["eval_reward_rank_correlation"].append(
                    metrics["reward_rank_correlation"]
                )
                history["eval_reward_kl"].append(metrics["eval_reward_kl"])
                history["eval_reward_js"].append(metrics["eval_reward_js"])
                history["eval_reward_js_normalized"].append(
                    metrics["eval_reward_js_normalized"]
                )
                history["eval_reward_mae"].append(metrics["eval_reward_mae"])
                history["eval_reward_rmse"].append(metrics["eval_reward_rmse"])
                history["eval_score_std_within_frontier"].append(
                    metrics["eval_score_std_within_frontier"]
                )

                eval_snapshots.append(
                    {
                        "epoch": eval_epoch,
                        "mean_reward": metrics["mean_reward"],
                        "std_reward": metrics["std_reward"],
                        "mean_chosen_distance": metrics["mean_chosen_distance"],
                        "std_chosen_distance": metrics["std_chosen_distance"],
                        "mean_oracle_distance": metrics["mean_oracle_distance"],
                        "std_oracle_distance": metrics["std_oracle_distance"],
                        "regret_mean": metrics["regret_mean"],
                        "regret_std": metrics["regret_std"],
                        "oracle_accuracy": metrics["oracle_accuracy"],
                        "n_failure_choices": metrics["n_failure_choices"],
                        "failure_choice_rate": metrics["failure_choice_rate"],
                        "n_failure_frontiers": metrics["n_failure_frontiers"],
                        "failure_frontier_rate": metrics["failure_frontier_rate"],
                        "n_failure_oracle_choices": metrics["n_failure_oracle_choices"],
                        "failure_avoidance_accuracy": metrics["failure_avoidance_accuracy"],
                        "n_failure_avoidance_applicable": metrics[
                            "n_failure_avoidance_applicable"
                        ],
                        "n_failure_avoidance_success": metrics[
                            "n_failure_avoidance_success"
                        ],
                        "n_only_failure_frontiers": metrics["n_only_failure_frontiers"],
                        "chosen_rewards": metrics["chosen_rewards"],
                        "chosen_distances": metrics["chosen_distances"],
                        "oracle_distances": metrics["oracle_distances"],
                        "regrets": metrics["regrets"],
                        "oracle_matches": metrics["oracle_matches"],
                        "chosen_is_failure": metrics["chosen_is_failure"],
                        "oracle_is_failure": metrics["oracle_is_failure"],
                        "frontier_has_failure": metrics["frontier_has_failure"],
                        "frontier_has_safe": metrics["frontier_has_safe"],
                        "failure_avoided_when_possible": metrics[
                            "failure_avoided_when_possible"
                        ],
                        "frontier_sizes": metrics["frontier_sizes"],
                        "oracle_ranks": metrics["oracle_ranks"],
                        "oracle_probabilities": metrics["oracle_probabilities"],
                        "reward_top1_matches": metrics["reward_top1_matches"],
                        "reward_rank_correlations": metrics["reward_rank_correlations"],
                        "reward_top1_accuracy": metrics["reward_top1_accuracy"],
                        "reward_rank_correlation": metrics["reward_rank_correlation"],
                        "reward_kl": metrics["eval_reward_kl"],
                        "reward_js": metrics["eval_reward_js"],
                        "reward_js_normalized": metrics["eval_reward_js_normalized"],
                        "reward_mae": metrics["eval_reward_mae"],
                        "reward_rmse": metrics["eval_reward_rmse"],
                        "score_std_within_frontier": metrics[
                            "eval_score_std_within_frontier"
                        ],
                        "failure_metrics_applicable": metrics[
                            "failure_metrics_applicable"
                        ],
                    }
                )

                if metrics["mean_reward"] > best_eval_reward:
                    best_eval_reward = metrics["mean_reward"]
                    best_eval_epoch = eval_epoch
                    self.save_model(
                        checkpoint / f"{model_name}.pt",
                        metrics={"best_eval_epoch": best_eval_epoch, **metrics},
                    )
                epoch_pbar.set_postfix(
                    train_reward=f"{mean_train_reward:.4f}",
                    train_loss=f"{mean_train_total_loss:.4f}",
                    eval_reward=f"{metrics['mean_reward']:.4f}",
                    regret=f"{metrics['regret_mean']:.4f}",
                    oracle_accuracy=f"{metrics['oracle_accuracy']:.4f}",
                    reward_top1=f"{metrics['reward_top1_accuracy']:.4f}",
                )
            else:
                epoch_pbar.set_postfix(
                    train_reward=f"{mean_train_reward:.4f}",
                    train_loss=f"{mean_train_total_loss:.4f}",
                )

        self._save_history_and_plots(
            history=history,
            eval_snapshots=eval_snapshots,
            output_dir=checkpoint,
            train_loader=train_loader,
            eval_loader=eval_loader,
            best_eval_epoch=best_eval_epoch,
        )
        return history

    def evaluate(self, loader, verbose: bool = False, max_print_errors: int = 10) -> Dict[str, object]:
        self.model.eval()

        chosen_rewards: List[float] = []
        chosen_distances: List[float] = []
        oracle_distances: List[float] = []
        regrets: List[float] = []
        oracle_matches: List[int] = []
        oracle_ranks: List[int] = []
        oracle_probabilities: List[float] = []
        reward_top1_matches: List[int] = []
        reward_rank_correlations: List[float] = []

        chosen_indices: List[int] = []
        oracle_indices_local: List[int] = []
        chosen_actions: List[int] = []
        oracle_actions: List[int] = []
        frontier_sizes: List[int] = []
        chosen_is_failure: List[int] = []
        oracle_is_failure: List[int] = []
        frontier_has_failure: List[int] = []
        frontier_has_safe: List[int] = []
        failure_avoided_when_possible: List[int] = []
        reward_kl_values: List[float] = []
        reward_js_values: List[float] = []
        reward_js_norm_values: List[float] = []
        reward_abs_errors: List[float] = []
        reward_sq_errors: List[float] = []
        score_std_values: List[float] = []

        error_details: List[Dict[str, object]] = []
        all_details: List[Dict[str, object]] = []

        with torch.no_grad():
            for raw_batch in loader:
                batch = self._move_to_device(raw_batch)

                logits = self.model(**self._build_model_kwargs(batch))

                frontier_ptr = batch["frontier_ptr"]
                rewards = batch["rewards"]
                distances = batch["distances"]
                action_map = batch["action_map"]
                oracle_index_global = batch["oracle_index"]
                pred_scores_2d, true_rewards_2d, frontier_mask_2d = (
                    self._get_frontier_scores_rewards_mask(batch, logits)
                )
                p_true = masked_softmax(
                    true_rewards_2d, frontier_mask_2d, self.reward_temperature
                )
                p_pred = masked_softmax(
                    pred_scores_2d, frontier_mask_2d, self.reward_temperature
                )

                # Single-candidate frontiers map to one-point distributions [1.0], so KL/JS are 0.
                batch_reward_kl = kl_divergence(p_true, p_pred, frontier_mask_2d)
                batch_reward_js = js_divergence(p_true, p_pred, frontier_mask_2d)
                batch_reward_js_norm = js_normalized(p_true, p_pred, frontier_mask_2d)
                reward_kl_values.extend([float(x) for x in batch_reward_kl.detach().cpu().tolist()])
                reward_js_values.extend([float(x) for x in batch_reward_js.detach().cpu().tolist()])
                reward_js_norm_values.extend(
                    [float(x) for x in batch_reward_js_norm.detach().cpu().tolist()]
                )
                _ = masked_mae(pred_scores_2d, true_rewards_2d, frontier_mask_2d)
                _ = masked_rmse(pred_scores_2d, true_rewards_2d, frontier_mask_2d)
                valid_abs = (pred_scores_2d - true_rewards_2d).abs()[frontier_mask_2d]
                valid_sq = ((pred_scores_2d - true_rewards_2d) ** 2)[frontier_mask_2d]
                reward_abs_errors.extend([float(x) for x in valid_abs.detach().cpu().tolist()])
                reward_sq_errors.extend([float(x) for x in valid_sq.detach().cpu().tolist()])
                _ = score_std_within_frontier(pred_scores_2d, frontier_mask_2d)

                for i in range(frontier_ptr.numel() - 1):
                    s = int(frontier_ptr[i].item())
                    e = int(frontier_ptr[i + 1].item())

                    frontier_logits = logits[s:e]
                    frontier_rewards = rewards[s:e]
                    frontier_distances = distances[s:e]
                    frontier_actions = action_map[s:e]

                    chosen_local = int(torch.argmax(frontier_logits).item())
                    oracle_global = int(oracle_index_global[i].item())
                    oracle_local = oracle_global - s

                    chosen_global = s + chosen_local

                    chosen_reward = float(frontier_rewards[chosen_local].item())
                    chosen_distance = float(frontier_distances[chosen_local].item())
                    oracle_distance = float(frontier_distances[oracle_local].item())

                    chosen_action = int(frontier_actions[chosen_local].item())
                    oracle_action = int(frontier_actions[oracle_local].item())

                    frontier_size = e - s
                    regret = chosen_distance - oracle_distance
                    oracle_match = int(chosen_local == oracle_local)
                    action_match = int(chosen_action == oracle_action)

                    probs = torch.softmax(frontier_logits, dim=0)
                    chosen_prob = float(probs[chosen_local].item())
                    oracle_prob = float(probs[oracle_local].item())
                    ranking = torch.argsort(frontier_logits, descending=True)
                    oracle_rank = int((ranking == oracle_local).nonzero(as_tuple=False)[0].item()) + 1
                    reward_best_local = int(torch.argmax(frontier_rewards).item())
                    reward_top1_match = int(chosen_local == reward_best_local)
                    if self.track_rank_correlation:
                        reward_rank_corr = self._spearman_rank_corr(frontier_logits, frontier_rewards)
                    else:
                        reward_rank_corr = 0.0

                    is_chosen_failure = int(
                        torch.isclose(
                            frontier_distances[chosen_local],
                            torch.tensor(self.m_failed_state, device=frontier_distances.device),
                            rtol=0.0,
                            atol=1e-9,
                        ).item()
                    )
                    is_oracle_failure = int(
                        torch.isclose(
                            frontier_distances[oracle_local],
                            torch.tensor(self.m_failed_state, device=frontier_distances.device),
                            rtol=0.0,
                            atol=1e-9,
                        ).item()
                    )
                    candidate_is_failure = torch.isclose(
                        frontier_distances,
                        torch.tensor(self.m_failed_state, device=frontier_distances.device),
                        rtol=0.0,
                        atol=1e-9,
                    )
                    has_failure = int(candidate_is_failure.any().item())
                    has_safe = int((~candidate_is_failure).any().item())
                    avoided_failure_if_possible = int((has_safe == 0) or (is_chosen_failure == 0))

                    detail = {
                        "frontier_id_in_batch": i,
                        "frontier_size": frontier_size,
                        "chosen_index_local": chosen_local,
                        "oracle_index_local": oracle_local,
                        "chosen_index_global": chosen_global,
                        "oracle_index_global": oracle_global,
                        "chosen_action": chosen_action,
                        "oracle_action": oracle_action,
                        "chosen_reward": chosen_reward,
                        "chosen_distance": chosen_distance,
                        "oracle_distance": oracle_distance,
                        "regret": regret,
                        "oracle_match": bool(oracle_match),
                        "action_match": bool(action_match),
                        "chosen_prob": chosen_prob,
                        "oracle_prob": oracle_prob,
                        "oracle_rank": oracle_rank,
                        "reward_best_index_local": reward_best_local,
                        "reward_top1_match": bool(reward_top1_match),
                        "reward_rank_correlation": float(reward_rank_corr),
                        "frontier_distances": [float(x) for x in frontier_distances.detach().cpu().tolist()],
                        "frontier_actions": [int(x) for x in frontier_actions.detach().cpu().tolist()],
                        "frontier_logits": [float(x) for x in frontier_logits.detach().cpu().tolist()],
                        "frontier_probs": [float(x) for x in probs.detach().cpu().tolist()],
                        # Failure frontier: contains at least one failure candidate.
                        "chosen_is_failure": bool(is_chosen_failure),
                        "oracle_is_failure": bool(is_oracle_failure),
                        "frontier_has_failure": bool(has_failure),
                        "frontier_has_safe_candidate": bool(has_safe),
                        "failure_avoided_when_possible": bool(
                            (has_safe == 0) or (is_chosen_failure == 0)
                        ),
                    }
                    all_details.append(detail)

                    chosen_rewards.append(chosen_reward)
                    chosen_distances.append(chosen_distance)
                    oracle_distances.append(oracle_distance)
                    regrets.append(regret)
                    oracle_matches.append(oracle_match)

                    chosen_indices.append(chosen_local)
                    oracle_indices_local.append(oracle_local)
                    chosen_actions.append(chosen_action)
                    oracle_actions.append(oracle_action)
                    frontier_sizes.append(frontier_size)
                    if frontier_size > 1:
                        score_std_values.append(
                            float(frontier_logits.std(unbiased=False).item())
                        )
                    chosen_is_failure.append(is_chosen_failure)
                    oracle_is_failure.append(is_oracle_failure)
                    frontier_has_failure.append(has_failure)
                    frontier_has_safe.append(has_safe)
                    failure_avoided_when_possible.append(avoided_failure_if_possible)
                    oracle_ranks.append(oracle_rank)
                    oracle_probabilities.append(oracle_prob)
                    reward_top1_matches.append(reward_top1_match)
                    reward_rank_correlations.append(float(reward_rank_corr))

                    if not oracle_match:
                        error_details.append(detail)

        n_frontiers = len(chosen_rewards)
        if n_frontiers == 0:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_chosen_distance": 0.0,
                "std_chosen_distance": 0.0,
                "mean_oracle_distance": 0.0,
                "std_oracle_distance": 0.0,
                "regret": 0.0,
                "regret_mean": 0.0,
                "regret_std": 0.0,
                "oracle_accuracy": 0.0,
                "oracle_accuracy (exact match with best candidate)": 0.0,
                "n_frontiers": 0,
                "n_errors": 0,
                "error_rate": 0.0,
                "n_action_errors": 0,
                "action_error_rate": 0.0,
                "n_failure_choices": 0,
                "failure_choice_rate": 0.0,
                "n_failure_frontiers": 0,
                "failure_frontier_rate": 0.0,
                "n_failure_oracle_choices": 0,
                "n_failure_avoidance_applicable": 0,
                "n_failure_avoidance_success": 0,
                "failure_avoidance_accuracy": 0.0,
                "n_only_failure_frontiers": 0,
                "chosen_rewards": [],
                "chosen_distances": [],
                "oracle_distances": [],
                "regrets": [],
                "oracle_matches": [],
                "chosen_is_failure": [],
                "oracle_is_failure": [],
                "frontier_has_failure": [],
                "frontier_has_safe": [],
                "failure_avoided_when_possible": [],
                "frontier_sizes": [],
                "oracle_ranks": [],
                "oracle_probabilities": [],
                "reward_top1_matches": [],
                "reward_rank_correlations": [],
                "reward_top1_accuracy": 0.0,
                "reward_rank_correlation": 0.0,
                "eval_reward_kl": 0.0,
                "eval_reward_js": 0.0,
                "eval_reward_js_normalized": 0.0,
                "eval_reward_mae": 0.0,
                "eval_reward_rmse": 0.0,
                "eval_score_std_within_frontier": 0.0,
                "failure_metrics_applicable": False,
                "details": [],
                "errors": [],
            }

        reward_t = torch.tensor(chosen_rewards, dtype=torch.float32)
        chosen_dist_t = torch.tensor(chosen_distances, dtype=torch.float32)
        oracle_dist_t = torch.tensor(oracle_distances, dtype=torch.float32)
        regret_t = torch.tensor(regrets, dtype=torch.float32)
        oracle_match_t = torch.tensor(oracle_matches, dtype=torch.float32)
        chosen_action_t = torch.tensor(chosen_actions, dtype=torch.long)
        oracle_action_t = torch.tensor(oracle_actions, dtype=torch.long)

        n_errors = int((oracle_match_t == 0).sum().item())
        n_action_errors = int((chosen_action_t != oracle_action_t).sum().item())
        chosen_failure_t = torch.tensor(chosen_is_failure, dtype=torch.float32)
        oracle_failure_t = torch.tensor(oracle_is_failure, dtype=torch.float32)
        frontier_has_failure_t = torch.tensor(frontier_has_failure, dtype=torch.float32)
        frontier_has_safe_t = torch.tensor(frontier_has_safe, dtype=torch.float32)
        reward_top1_t = torch.tensor(reward_top1_matches, dtype=torch.float32)
        n_failure_choices = int(chosen_failure_t.sum().item())
        n_failure_frontiers = int(frontier_has_failure_t.sum().item())
        n_failure_oracle_choices = int(oracle_failure_t.sum().item())
        n_failure_avoidance_applicable = int(frontier_has_safe_t.sum().item())
        n_failure_avoidance_success = int(
            (
                (frontier_has_safe_t == 1.0)
                & (chosen_failure_t == 0.0)
            ).sum().item()
        )
        n_only_failure_frontiers = int((frontier_has_safe_t == 0.0).sum().item())
        failure_avoidance_accuracy = (
            float(n_failure_avoidance_success / n_failure_avoidance_applicable)
            if n_failure_avoidance_applicable > 0
            else 0.0
        )
        failure_metrics_applicable = n_failure_frontiers > 0
        eval_reward_mae = (
            float(sum(reward_abs_errors) / len(reward_abs_errors))
            if reward_abs_errors
            else 0.0
        )
        eval_reward_rmse = (
            float((sum(reward_sq_errors) / len(reward_sq_errors)) ** 0.5)
            if reward_sq_errors
            else 0.0
        )
        eval_score_std_within_frontier = (
            float(sum(score_std_values) / len(score_std_values))
            if score_std_values
            else 0.0
        )

        metrics: Dict[str, object] = {
            "mean_reward": float(reward_t.mean().item()),
            "std_reward": float(reward_t.std(unbiased=False).item()),
            "mean_chosen_distance": float(chosen_dist_t.mean().item()),
            "std_chosen_distance": float(chosen_dist_t.std(unbiased=False).item()),
            "mean_oracle_distance": float(oracle_dist_t.mean().item()),
            "std_oracle_distance": float(oracle_dist_t.std(unbiased=False).item()),
            # Oracle accuracy is exact-match: chosen candidate index equals oracle-best index.
            "regret": float(regret_t.mean().item()),
            # Mean regret is the average chosen-vs-oracle distance gap in this pipeline.
            "regret_mean": float(regret_t.mean().item()),
            "regret_std": float(regret_t.std(unbiased=False).item()),
            "oracle_accuracy": float(oracle_match_t.mean().item()),
            "oracle_accuracy (exact match with best candidate)": float(
                oracle_match_t.mean().item()
            ),
            "n_frontiers": n_frontiers,
            "n_errors": n_errors,
            "error_rate": float(n_errors / n_frontiers),
            "n_action_errors": n_action_errors,
            "action_error_rate": float(n_action_errors / n_frontiers),
            "n_failure_choices": n_failure_choices,
            "failure_choice_rate": float(n_failure_choices / n_frontiers),
            "n_failure_frontiers": n_failure_frontiers,
            "failure_frontier_rate": float(n_failure_frontiers / n_frontiers),
            "n_failure_oracle_choices": n_failure_oracle_choices,
            "n_failure_avoidance_applicable": n_failure_avoidance_applicable,
            "n_failure_avoidance_success": n_failure_avoidance_success,
            "failure_avoidance_accuracy": failure_avoidance_accuracy,
            "failure_metrics_applicable": failure_metrics_applicable,
            "n_only_failure_frontiers": n_only_failure_frontiers,
            "chosen_rewards": chosen_rewards,
            "chosen_distances": chosen_distances,
            "oracle_distances": oracle_distances,
            "regrets": regrets,
            "oracle_matches": oracle_matches,
            "chosen_indices": chosen_indices,
            "oracle_indices": oracle_indices_local,
            "chosen_actions": chosen_actions,
            "oracle_actions": oracle_actions,
            "chosen_is_failure": chosen_is_failure,
            "oracle_is_failure": oracle_is_failure,
            "frontier_has_failure": frontier_has_failure,
            "frontier_has_safe": frontier_has_safe,
            "failure_avoided_when_possible": failure_avoided_when_possible,
            "frontier_sizes": frontier_sizes,
            "oracle_ranks": oracle_ranks,
            "oracle_probabilities": oracle_probabilities,
            "reward_top1_matches": reward_top1_matches,
            "reward_rank_correlations": reward_rank_correlations,
            "reward_top1_accuracy": float(reward_top1_t.mean().item()),
            "reward_rank_correlation": self._safe_mean(reward_rank_correlations),
            "eval_reward_kl": self._safe_mean(reward_kl_values),
            "eval_reward_js": self._safe_mean(reward_js_values),
            "eval_reward_js_normalized": self._safe_mean(reward_js_norm_values),
            "eval_reward_mae": eval_reward_mae,
            "eval_reward_rmse": eval_reward_rmse,
            "eval_score_std_within_frontier": eval_score_std_within_frontier,
            "details": all_details,
            "errors": error_details,
        }

        if verbose:
            print("\nEvaluation summary")
            print(f"frontiers           : {metrics['n_frontiers']} (number of decision points)")
            print(f"errors              : {metrics['n_errors']}/{metrics['n_frontiers']} ({100.0 * metrics['error_rate']:.2f}%) (agent ≠ oracle index)")
            print(f"action errors       : {metrics['n_action_errors']}/{metrics['n_frontiers']} ({100.0 * metrics['action_error_rate']:.2f}%) (planner action mismatch)")
            print(f"oracle accuracy     : {metrics['oracle_accuracy']:.4f} (exact match with best candidate)")
            print(f"reward top1 acc     : {metrics['reward_top1_accuracy']:.4f} (argmax(logits) vs argmax(reward))")
            print(f"reward rank corr    : {metrics['reward_rank_correlation']:.4f} (Spearman across candidates)")
            print(f"reward KL           : {metrics['eval_reward_kl']:.4f} (p_true || p_pred)")
            print(f"reward JS           : {metrics['eval_reward_js']:.4f}")
            print(f"reward JS norm      : {metrics['eval_reward_js_normalized']:.4f} (0..1)")
            print(f"reward MAE          : {metrics['eval_reward_mae']:.4f}")
            print(f"reward RMSE         : {metrics['eval_reward_rmse']:.4f}")
            print(f"score std/frontier  : {metrics['eval_score_std_within_frontier']:.4f}")
            print(f"failure choices     : {metrics['n_failure_choices']}/{metrics['n_frontiers']} ({100.0 * metrics['failure_choice_rate']:.2f}%)")
            print(f"failure frontiers   : {metrics['n_failure_frontiers']}/{metrics['n_frontiers']} ({100.0 * metrics['failure_frontier_rate']:.2f}%)")
            if not metrics["failure_metrics_applicable"]:
                print("failure metrics     : not informative for this split (no failure frontiers)")
            print(f"failure avoidance   : {metrics['failure_avoidance_accuracy']:.4f} on {metrics['n_failure_avoidance_applicable']} applicable frontiers")
            print(f"only-failure fronts : {metrics['n_only_failure_frontiers']} (not counted against failure avoidance)")
            print(f"mean reward         : {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f} (expected policy reward)")
            print(f"chosen distance     : {metrics['mean_chosen_distance']:.4f} ± {metrics['std_chosen_distance']:.4f} (distance of selected candidate)")
            print(f"oracle distance     : {metrics['mean_oracle_distance']:.4f} ± {metrics['std_oracle_distance']:.4f} (best achievable distance)")
            print(f"regret              : {metrics['regret']:.4f} ± {metrics['regret_std']:.4f} (distance gap to optimal)")
            print("note                : oracle accuracy (exact match) and regret (magnitude gap) can disagree")
            if error_details:
                print("\nFirst errors:")
                for k, err in enumerate(error_details[:max_print_errors]):
                    print(
                        f"{k}) size={err['frontier_size']} | "
                        f"agent_idx={err['chosen_index_local']} action={err['chosen_action']} dist={err['chosen_distance']:.4f} prob={err['chosen_prob']:.4f} | "
                        f"oracle_idx={err['oracle_index_local']} action={err['oracle_action']} dist={err['oracle_distance']:.4f} prob={err['oracle_prob']:.4f} | "
                        f"regret={err['regret']:.4f}"
                    )

        return metrics

    def save_model(self, out_path: str | Path, metrics: Optional[Dict[str, float]] = None):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "config": {
                "node_input_dim": self.model.encoder.input_proj.in_features,
                "hidden_dim": self.model.encoder.input_proj.out_features,
                "gnn_layers": len(self.model.encoder.layers),
                "conv_type": self.model.encoder.conv_type,
                "pooling_type": self.model.pooling_type,
                "use_global_context": self.model.use_global_context,
                "mlp_depth": (len(self.model.policy_head) - 1) // 2,
                "use_goal_separate_input": self.model.use_goal_separate_input,
            },
            "metrics": metrics or {},
        }
        torch.save(payload, out_path)

    @staticmethod
    def load_model(path: str | Path, device: Optional[str | torch.device] = None) -> FrontierPolicyNetwork:
        payload = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = payload["config"]
        model = FrontierPolicyNetwork(**cfg)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    def to_onnx(self, out_path: str | Path, node_input_dim: int) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_nodes, n_edges, n_candidates = 8, 12, 4
        if self.kind_of_data == "separated" and self.model.use_goal_separate_input:
            wrapper = OnnxFrontierPolicySeparatedWrapper(self.model).eval().cpu()
            n_goal_nodes, n_goal_edges = 6, 8
            dummy_inputs = (
                torch.zeros((n_nodes, node_input_dim), dtype=torch.float32),
                torch.zeros((2, n_edges), dtype=torch.int64),
                torch.zeros((n_edges, 1), dtype=torch.float32),
                torch.arange(n_nodes, dtype=torch.int64) % n_candidates,
                torch.zeros((n_goal_nodes, node_input_dim), dtype=torch.float32),
                torch.zeros((2, n_goal_edges), dtype=torch.int64),
                torch.zeros((n_goal_edges, 1), dtype=torch.float32),
                torch.zeros((n_goal_nodes,), dtype=torch.int64),
                torch.ones((n_candidates,), dtype=torch.bool),
            )
            input_names = [
                "node_features",
                "edge_index",
                "edge_attr",
                "membership",
                "goal_node_features",
                "goal_edge_index",
                "goal_edge_attr",
                "goal_batch",
                "mask",
            ]
            dynamic_axes = {
                "node_features": {0: "N"},
                "edge_index": {1: "E"},
                "edge_attr": {0: "E"},
                "membership": {0: "N"},
                "goal_node_features": {0: "GN"},
                "goal_edge_index": {1: "GE"},
                "goal_edge_attr": {0: "GE"},
                "goal_batch": {0: "GN"},
                "mask": {0: "F"},
                "logits": {0: "F"},
            }
        else:
            wrapper = OnnxFrontierPolicyWrapper(self.model).eval().cpu()
            dummy_inputs = (
                torch.zeros((n_nodes, node_input_dim), dtype=torch.float32),
                torch.zeros((2, n_edges), dtype=torch.int64),
                torch.zeros((n_edges, 1), dtype=torch.float32),
                torch.arange(n_nodes, dtype=torch.int64) % n_candidates,
                torch.ones((n_candidates,), dtype=torch.bool),
            )
            input_names = [
                "node_features",
                "edge_index",
                "edge_attr",
                "membership",
                "mask",
            ]
            dynamic_axes = {
                "node_features": {0: "N"},
                "edge_index": {1: "E"},
                "edge_attr": {0: "E"},
                "membership": {0: "N"},
                "mask": {0: "F"},
                "logits": {0: "F"},
            }

        torch.onnx.export(
            wrapper,
            dummy_inputs,
            out_path.as_posix(),
            opset_version=18,
            dynamo=False,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
        )

    @staticmethod
    def _safe_mean(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _get_loader_frontier_sizes(loader) -> List[int]:
        dataset = getattr(loader, "dataset", None)
        samples = getattr(dataset, "samples", None)
        if samples is None:
            return []
        sizes = []
        for s in samples:
            rewards = s.get("rewards", None)
            if rewards is not None:
                sizes.append(int(rewards.size(0)))
        return sizes

    @staticmethod
    def _plot_curve_with_band(
        out_path: Path,
        x: Sequence[int],
        y: Sequence[float],
        y_std: Optional[Sequence[float]],
        xlabel: str,
        ylabel: str,
        label: Optional[str] = None,
    ) -> None:
        plt.figure(figsize=(6, 4), dpi=300)
        if y:
            plt.plot(x, y, label=label)
            if y_std is not None and len(y_std) == len(y):
                lo = [m - s for m, s in zip(y, y_std)]
                hi = [m + s for m, s in zip(y, y_std)]
                plt.fill_between(x, lo, hi, alpha=0.2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if label is not None:
            plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    @staticmethod
    def _save_placeholder_plot(out_path: Path, message: str, title: str = "") -> None:
        plt.figure(figsize=(6, 4), dpi=300)
        if title:
            plt.title(title)
        plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    @staticmethod
    def _plot_frontier_size_histogram(
        out_path: Path,
        train_sizes: Sequence[int],
        eval_sizes: Sequence[int],
    ) -> None:
        plt.figure(figsize=(6, 4), dpi=300)
        if train_sizes:
            plt.hist(train_sizes, bins=min(20, max(1, len(set(train_sizes)))), alpha=0.6, label="train")
        if eval_sizes:
            plt.hist(eval_sizes, bins=min(20, max(1, len(set(eval_sizes)))), alpha=0.6, label="eval")
        plt.xlabel("Frontier Size")
        plt.ylabel("Count")
        if train_sizes or eval_sizes:
            plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    @staticmethod
    def _plot_regret_distribution(out_path: Path, regrets: Sequence[float]) -> None:
        plt.figure(figsize=(6, 4), dpi=300)
        if regrets:
            plt.hist(regrets, bins=min(30, max(1, int(len(regrets) ** 0.5))), alpha=0.8)
        else:
            plt.text(0.5, 0.5, "No regret data available", ha="center", va="center")
        plt.xlabel("Regret")
        plt.ylabel("Frontier Count")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    @staticmethod
    def _plot_performance_by_frontier_size(
        out_path: Path,
        frontier_sizes: Sequence[int],
        regrets: Sequence[float],
        oracle_matches: Sequence[float],
    ) -> None:
        by_size_regret: Dict[int, List[float]] = defaultdict(list)
        by_size_acc: Dict[int, List[float]] = defaultdict(list)
        for size, reg, match in zip(frontier_sizes, regrets, oracle_matches):
            by_size_regret[int(size)].append(float(reg))
            by_size_acc[int(size)].append(float(match))

        sizes = sorted(by_size_regret.keys())
        mean_regret = [sum(by_size_regret[s]) / len(by_size_regret[s]) for s in sizes]
        mean_acc = [sum(by_size_acc[s]) / len(by_size_acc[s]) for s in sizes]

        fig, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
        if sizes:
            ax1.plot(sizes, mean_regret, marker="o", label="Mean Regret")
        ax1.set_xlabel("Frontier Size")
        ax1.set_ylabel("Mean Regret")

        ax2 = ax1.twinx()
        if sizes:
            ax2.plot(sizes, mean_acc, marker="s", linestyle="--", label="Oracle Accuracy")
        ax2.set_ylabel("Oracle Accuracy")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax1.legend(h1 + h2, l1 + l2, loc="best")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    @staticmethod
    def _plot_failure_distribution_last(
        out_path: Path,
        chosen_is_failure: Sequence[int],
    ) -> None:
        n_failure = int(sum(int(x) for x in chosen_is_failure))
        n_non_failure = int(len(chosen_is_failure) - n_failure)
        plt.figure(figsize=(6, 4), dpi=300)
        if not chosen_is_failure:
            plt.text(0.5, 0.5, "No evaluation decisions available", ha="center", va="center")
        else:
            plt.bar(["Chosen Failure", "Chosen Non-Failure"], [n_failure, n_non_failure], alpha=0.85)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    @staticmethod
    def _plot_action_outcome_heatmap_last(
        out_path: Path,
        oracle_matches: Sequence[int],
        chosen_is_failure: Sequence[int],
    ) -> None:
        if not oracle_matches or not chosen_is_failure or len(oracle_matches) != len(chosen_is_failure):
            RLFrontierTrainer._save_placeholder_plot(
                out_path,
                "No valid data available for action-outcome heatmap.",
                title="Action Outcome Heatmap",
            )
            return

        # Rows: correct / incorrect action. Columns: failure / non-failure chosen outcome.
        matrix = torch.zeros((2, 2), dtype=torch.int32)
        for m, f in zip(oracle_matches, chosen_is_failure):
            row = 0 if int(m) == 1 else 1
            col = 0 if int(f) == 1 else 1
            matrix[row, col] += 1

        plt.figure(figsize=(6, 4), dpi=300)
        im = plt.imshow(matrix.numpy(), cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks([0, 1], ["Failure", "Non-Failure"])
        plt.yticks([0, 1], ["Correct", "Incorrect"])
        plt.xlabel("Chosen Outcome")
        plt.ylabel("Action Correctness")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, int(matrix[i, j].item()), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    @staticmethod
    def _plot_failure_by_frontier_size(
        out_path: Path,
        frontier_sizes: Sequence[int],
        chosen_is_failure: Sequence[int],
    ) -> None:
        if not frontier_sizes or not chosen_is_failure or len(frontier_sizes) != len(chosen_is_failure):
            RLFrontierTrainer._save_placeholder_plot(
                out_path,
                "No valid data available for failure rate by frontier size.",
                title="Failure by Frontier Size",
            )
            return
        by_size_total: Dict[int, int] = defaultdict(int)
        by_size_failure: Dict[int, int] = defaultdict(int)
        for size, is_failure in zip(frontier_sizes, chosen_is_failure):
            s = int(size)
            by_size_total[s] += 1
            by_size_failure[s] += int(is_failure)
        sizes = sorted(by_size_total.keys())
        rates = [
            float(by_size_failure[s] / by_size_total[s]) if by_size_total[s] > 0 else 0.0
            for s in sizes
        ]
        plt.figure(figsize=(6, 4), dpi=300)
        if sizes:
            plt.plot(sizes, rates, marker="o")
        else:
            plt.text(0.5, 0.5, "No frontier-size data available", ha="center", va="center")
        plt.xlabel("Frontier Size")
        plt.ylabel("Failure Choice Rate")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    @staticmethod
    def _plot_failure_gap_distribution(
        out_path: Path,
        regrets: Sequence[float],
        chosen_is_failure: Sequence[int],
    ) -> None:
        if not regrets or not chosen_is_failure or len(regrets) != len(chosen_is_failure):
            RLFrontierTrainer._save_placeholder_plot(
                out_path,
                "No valid data available for failure-regret distribution.",
                title="Failure Regret Distribution",
            )
            return
        failure_regrets = [float(r) for r, is_failure in zip(regrets, chosen_is_failure) if int(is_failure) == 1]
        plt.figure(figsize=(6, 4), dpi=300)
        if failure_regrets:
            plt.hist(
                failure_regrets,
                bins=min(30, max(1, int(len(failure_regrets) ** 0.5))),
                alpha=0.8,
            )
            plt.xlabel("Regret (Failure Choices Only)")
            plt.ylabel("Frontier Count")
        else:
            plt.text(0.5, 0.5, "No failure choices in final evaluation", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def _save_metrics_summary(
        self,
        output_dir: Path,
        history: Dict[str, List[float]],
        eval_snapshots: List[Dict[str, object]],
        train_loader,
        eval_loader,
        best_eval_epoch: int,
    ) -> None:
        train_sizes = self._get_loader_frontier_sizes(train_loader)
        eval_sizes = self._get_loader_frontier_sizes(eval_loader)
        final = eval_snapshots[-1] if eval_snapshots else {}
        summary = {
            "best_eval_reward": max(history["eval_reward"]) if history["eval_reward"] else 0.0,
            "best_eval_epoch": best_eval_epoch if best_eval_epoch >= 0 else None,
            "final_eval_reward": final.get("mean_reward", 0.0),
            "final_eval_reward_std": final.get("std_reward", 0.0),
            "final_regret": final.get("regret_mean", 0.0),
            "final_regret_std": final.get("regret_std", 0.0),
            "final_oracle_accuracy": final.get("oracle_accuracy", 0.0),
            "final_failure_choice_rate": final.get("failure_choice_rate", 0.0),
            "final_failure_avoidance_accuracy": final.get("failure_avoidance_accuracy", 0.0),
            "final_failure_frontier_rate": final.get("failure_frontier_rate", 0.0),
            "final_n_failure_choices": final.get("n_failure_choices", 0),
            "final_n_failure_frontiers": final.get("n_failure_frontiers", 0),
            "final_n_failure_avoidance_applicable": final.get("n_failure_avoidance_applicable", 0),
            "final_n_only_failure_frontiers": final.get("n_only_failure_frontiers", 0),
            "failure_metrics_applicable": bool(final.get("failure_metrics_applicable", False)),
            "final_chosen_distance": final.get("mean_chosen_distance", 0.0),
            "final_chosen_distance_std": final.get("std_chosen_distance", 0.0),
            "final_oracle_distance": final.get("mean_oracle_distance", 0.0),
            "final_oracle_distance_std": final.get("std_oracle_distance", 0.0),
            "final_reward_top1_accuracy": final.get("reward_top1_accuracy", 0.0),
            "final_reward_rank_correlation": final.get("reward_rank_correlation", 0.0),
            "final_reward_kl": final.get("reward_kl", 0.0),
            "final_reward_js": final.get("reward_js", 0.0),
            "final_reward_js_normalized": final.get("reward_js_normalized", 0.0),
            "final_reward_mae": final.get("reward_mae", 0.0),
            "final_reward_rmse": final.get("reward_rmse", 0.0),
            "final_score_std_within_frontier": final.get("score_std_within_frontier", 0.0),
            "num_train_frontiers": len(train_sizes),
            "num_eval_frontiers": len(eval_sizes),
            "mean_train_frontier_size": self._safe_mean(train_sizes),
            "mean_eval_frontier_size": self._safe_mean(eval_sizes),
            "m_failed_state": self.m_failed_state,
        }
        with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    def _save_history_and_plots(
        self,
        history: Dict[str, List[float]],
        eval_snapshots: List[Dict[str, object]],
        output_dir: Path,
        train_loader,
        eval_loader,
        best_eval_epoch: int,
    ) -> None:
        with (output_dir / "history_losses.json").open("w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
        with (output_dir / "eval_snapshots.json").open("w", encoding="utf-8") as fh:
            json.dump(eval_snapshots, fh, indent=2)

        train_x = list(range(1, len(history["train_reward"]) + 1))
        eval_x = history["eval_epochs"] if history["eval_epochs"] else list(
            range(1, len(history["eval_reward"]) + 1)
        )

        self._plot_curve_with_band(
            output_dir / "train_reward.png",
            x=train_x,
            y=history["train_reward"],
            y_std=None,
            xlabel="Epoch",
            ylabel="Expected Train Reward",
            label=None,
        )
        self._plot_curve_with_band(
            output_dir / "eval_reward.png",
            x=eval_x,
            y=history["eval_reward"],
            y_std=history["eval_reward_std"],
            xlabel="Epoch",
            ylabel="Evaluation Reward",
            label="Eval Reward",
        )
        self._plot_curve_with_band(
            output_dir / "eval_regret.png",
            x=eval_x,
            y=history["eval_regret"],
            y_std=history["eval_regret_std"],
            xlabel="Epoch",
            ylabel="Regret",
            label="Eval Regret",
        )
        self._plot_curve_with_band(
            output_dir / "eval_oracle_accuracy.png",
            x=eval_x,
            y=history["eval_oracle_accuracy"],
            y_std=None,
            xlabel="Epoch",
            ylabel="Oracle Accuracy",
            label="Oracle Accuracy",
        )
        self._plot_curve_with_band(
            output_dir / "eval_failure_choice_rate.png",
            x=eval_x,
            y=history["eval_failure_choice_rate"],
            y_std=None,
            xlabel="Epoch",
            ylabel="Failure Choice Rate",
            label="Failure Choice Rate",
        )
        self._plot_curve_with_band(
            output_dir / "eval_reward_top1_accuracy.png",
            x=eval_x,
            y=history["eval_reward_top1_accuracy"],
            y_std=None,
            xlabel="Epoch",
            ylabel="Reward Top-1 Accuracy",
            label="Reward Top-1 Accuracy",
        )
        self._plot_curve_with_band(
            output_dir / "eval_reward_rank_correlation.png",
            x=eval_x,
            y=history["eval_reward_rank_correlation"],
            y_std=None,
            xlabel="Epoch",
            ylabel="Reward Rank Correlation",
            label="Reward Rank Correlation",
        )
        self._plot_curve_with_band(
            output_dir / "eval_reward_js_normalized.png",
            x=eval_x,
            y=history["eval_reward_js_normalized"],
            y_std=None,
            xlabel="Epoch",
            ylabel="Normalized JS",
            label="Eval Reward JS (Normalized)",
        )
        self._plot_curve_with_band(
            output_dir / "eval_reward_kl.png",
            x=eval_x,
            y=history["eval_reward_kl"],
            y_std=None,
            xlabel="Epoch",
            ylabel="KL Divergence",
            label="Eval Reward KL",
        )
        self._plot_curve_with_band(
            output_dir / "eval_score_std_within_frontier.png",
            x=eval_x,
            y=history["eval_score_std_within_frontier"],
            y_std=None,
            xlabel="Epoch",
            ylabel="Score Std Within Frontier",
            label="Eval Score Std Within Frontier",
        )

        plt.figure(figsize=(6, 4), dpi=300)
        if history["eval_oracle_accuracy"]:
            plt.plot(eval_x, history["eval_oracle_accuracy"], label="Oracle Accuracy")
        if history["eval_failure_avoidance_accuracy"]:
            plt.plot(
                eval_x,
                history["eval_failure_avoidance_accuracy"],
                label="Failure Avoidance Accuracy",
            )
        if history["eval_oracle_accuracy"] or history["eval_failure_avoidance_accuracy"]:
            plt.legend(loc="best")
        else:
            plt.text(0.5, 0.5, "No evaluation snapshots available", ha="center", va="center")
        plt.xlabel("Epoch")
        plt.ylabel("Rate")
        plt.tight_layout()
        plt.savefig(output_dir / "eval_failure_vs_correctness.png")
        plt.close()

        plt.figure(figsize=(6, 4), dpi=300)
        if history["eval_chosen_distance"]:
            plt.plot(eval_x, history["eval_chosen_distance"], label="Chosen Distance")
            if len(history["eval_chosen_distance_std"]) == len(history["eval_chosen_distance"]):
                lo = [m - s for m, s in zip(history["eval_chosen_distance"], history["eval_chosen_distance_std"])]
                hi = [m + s for m, s in zip(history["eval_chosen_distance"], history["eval_chosen_distance_std"])]
                plt.fill_between(eval_x, lo, hi, alpha=0.2)
        if history["eval_oracle_distance"]:
            plt.plot(eval_x, history["eval_oracle_distance"], label="Oracle Distance")
            if len(history["eval_oracle_distance_std"]) == len(history["eval_oracle_distance"]):
                lo = [m - s for m, s in zip(history["eval_oracle_distance"], history["eval_oracle_distance_std"])]
                hi = [m + s for m, s in zip(history["eval_oracle_distance"], history["eval_oracle_distance_std"])]
                plt.fill_between(eval_x, lo, hi, alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel("Distance")
        if history["eval_chosen_distance"] or history["eval_oracle_distance"]:
            plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / "eval_distances.png")
        plt.close()

        train_sizes = self._get_loader_frontier_sizes(train_loader)
        eval_sizes = self._get_loader_frontier_sizes(eval_loader)
        self._plot_frontier_size_histogram(
            output_dir / "frontier_size_hist_train_eval.png",
            train_sizes=train_sizes,
            eval_sizes=eval_sizes,
        )

        final_snapshot = eval_snapshots[-1] if eval_snapshots else {}
        self._plot_regret_distribution(
            output_dir / "eval_regret_distribution_last.png",
            regrets=final_snapshot.get("regrets", []),
        )
        self._plot_performance_by_frontier_size(
            output_dir / "eval_performance_by_frontier_size.png",
            frontier_sizes=final_snapshot.get("frontier_sizes", []),
            regrets=final_snapshot.get("regrets", []),
            oracle_matches=final_snapshot.get("oracle_matches", []),
        )
        self._plot_failure_distribution_last(
            output_dir / "eval_failure_distribution_last.png",
            chosen_is_failure=final_snapshot.get("chosen_is_failure", []),
        )
        self._plot_action_outcome_heatmap_last(
            output_dir / "eval_action_outcome_heatmap_last.png",
            oracle_matches=final_snapshot.get("oracle_matches", []),
            chosen_is_failure=final_snapshot.get("chosen_is_failure", []),
        )
        self._plot_failure_by_frontier_size(
            output_dir / "eval_failure_by_frontier_size.png",
            frontier_sizes=final_snapshot.get("frontier_sizes", []),
            chosen_is_failure=final_snapshot.get("chosen_is_failure", []),
        )
        self._plot_failure_gap_distribution(
            output_dir / "eval_failure_gap_distribution_last.png",
            regrets=final_snapshot.get("regrets", []),
            chosen_is_failure=final_snapshot.get("chosen_is_failure", []),
        )

        self._save_metrics_summary(
            output_dir=output_dir,
            history=history,
            eval_snapshots=eval_snapshots,
            train_loader=train_loader,
            eval_loader=eval_loader,
            best_eval_epoch=best_eval_epoch,
        )

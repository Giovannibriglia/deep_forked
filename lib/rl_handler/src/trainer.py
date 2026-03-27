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


class RLFrontierTrainer:
    def __init__(
        self,
        model: FrontierPolicyNetwork,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str | torch.device | None = None,
        reward_formulation: str = "negative_distance",
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

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.model(
            node_features=batch["node_features"],
            edge_index=batch["edge_index"],
            edge_attr=batch["edge_attr"],
            membership=batch["membership"],
            candidate_batch=batch["candidate_batch"],
        )
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
        return -expected_reward

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
            "eval_epochs": [],
            "eval_reward": [],
            "eval_reward_std": [],
            "eval_regret": [],
            "eval_regret_std": [],
            "eval_oracle_accuracy": [],
            "eval_chosen_distance": [],
            "eval_chosen_distance_std": [],
            "eval_oracle_distance": [],
            "eval_oracle_distance_std": [],
        }
        eval_snapshots: List[Dict[str, object]] = []

        epoch_pbar = tqdm(range(n_epochs), desc="training", leave=True)
        for epoch in epoch_pbar:
            self.model.train()
            total_train_reward = 0.0
            batches = 0

            for raw_batch in train_loader:
                batch = self._move_to_device(raw_batch)
                self.optimizer.zero_grad()
                loss = self.compute_loss(batch)
                loss.backward()
                self.optimizer.step()

                total_train_reward += -float(loss.detach().item())
                batches += 1

            mean_train_reward = total_train_reward / max(1, batches)
            history["train_reward"].append(mean_train_reward)

            if (epoch + 1) % eval_every == 0:
                metrics = self.evaluate(eval_loader)
                eval_epoch = epoch + 1
                history["eval_epochs"].append(eval_epoch)
                history["eval_reward"].append(metrics["mean_reward"])
                history["eval_reward_std"].append(metrics["std_reward"])
                history["eval_regret"].append(metrics["regret_mean"])
                history["eval_regret_std"].append(metrics["regret_std"])
                history["eval_oracle_accuracy"].append(metrics["oracle_accuracy"])
                history["eval_chosen_distance"].append(metrics["mean_chosen_distance"])
                history["eval_chosen_distance_std"].append(
                    metrics["std_chosen_distance"]
                )
                history["eval_oracle_distance"].append(metrics["mean_oracle_distance"])
                history["eval_oracle_distance_std"].append(
                    metrics["std_oracle_distance"]
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
                        "chosen_rewards": metrics["chosen_rewards"],
                        "chosen_distances": metrics["chosen_distances"],
                        "oracle_distances": metrics["oracle_distances"],
                        "regrets": metrics["regrets"],
                        "oracle_matches": metrics["oracle_matches"],
                        "frontier_sizes": metrics["frontier_sizes"],
                        "oracle_ranks": metrics["oracle_ranks"],
                        "oracle_probabilities": metrics["oracle_probabilities"],
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
                    eval_reward=f"{metrics['mean_reward']:.4f}",
                    regret=f"{metrics['regret_mean']:.4f}",
                )
            else:
                epoch_pbar.set_postfix(train_reward=f"{mean_train_reward:.4f}")

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

        chosen_indices: List[int] = []
        oracle_indices_local: List[int] = []
        chosen_actions: List[int] = []
        oracle_actions: List[int] = []
        frontier_sizes: List[int] = []

        error_details: List[Dict[str, object]] = []
        all_details: List[Dict[str, object]] = []

        with torch.no_grad():
            for raw_batch in loader:
                batch = self._move_to_device(raw_batch)

                logits = self.model(
                    node_features=batch["node_features"],
                    edge_index=batch["edge_index"],
                    edge_attr=batch["edge_attr"],
                    membership=batch["membership"],
                    candidate_batch=batch["candidate_batch"],
                )

                frontier_ptr = batch["frontier_ptr"]
                rewards = batch["rewards"]
                distances = batch["distances"]
                action_map = batch["action_map"]
                oracle_index_global = batch["oracle_index"]

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
                        "frontier_distances": [float(x) for x in frontier_distances.detach().cpu().tolist()],
                        "frontier_actions": [int(x) for x in frontier_actions.detach().cpu().tolist()],
                        "frontier_logits": [float(x) for x in frontier_logits.detach().cpu().tolist()],
                        "frontier_probs": [float(x) for x in probs.detach().cpu().tolist()],
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
                "regret_std": 0.0,
                "oracle_accuracy (exact match with best candidate)": 0.0,
                "n_frontiers": 0,
                "n_errors": 0,
                "error_rate": 0.0,
                "n_action_errors": 0,
                "action_error_rate": 0.0,
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

        metrics: Dict[str, object] = {
            "mean_reward": float(reward_t.mean().item()),
            "std_reward": float(reward_t.std(unbiased=False).item()),
            "mean_chosen_distance": float(chosen_dist_t.mean().item()),
            "std_chosen_distance": float(chosen_dist_t.std(unbiased=False).item()),
            "mean_oracle_distance": float(oracle_dist_t.mean().item()),
            "std_oracle_distance": float(oracle_dist_t.std(unbiased=False).item()),
            "regret": float(regret_t.mean().item()),
            "regret_std": float(regret_t.std(unbiased=False).item()),
            "oracle_accuracy": float(oracle_match_t.mean().item()),
            "n_frontiers": n_frontiers,
            "n_errors": n_errors,
            "error_rate": float(n_errors / n_frontiers),
            "n_action_errors": n_action_errors,
            "action_error_rate": float(n_action_errors / n_frontiers),
            "chosen_indices": chosen_indices,
            "oracle_indices": oracle_indices_local,
            "chosen_actions": chosen_actions,
            "oracle_actions": oracle_actions,
            "frontier_sizes": frontier_sizes,
            "details": all_details,
            "errors": error_details,
        }

        if verbose:
            print("\nEvaluation summary")
            print(f"frontiers           : {metrics['n_frontiers']} (number of decision points)")
            print(f"errors              : {metrics['n_errors']}/{metrics['n_frontiers']} ({100.0 * metrics['error_rate']:.2f}%) (agent ≠ oracle index)")
            print(f"action errors       : {metrics['n_action_errors']}/{metrics['n_frontiers']} ({100.0 * metrics['action_error_rate']:.2f}%) (planner action mismatch)")
            print(f"oracle accuracy     : {metrics['oracle_accuracy']:.4f} (exact match with best candidate)")
            print(f"mean reward         : {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f} (expected policy reward)")
            print(f"chosen distance     : {metrics['mean_chosen_distance']:.4f} ± {metrics['std_chosen_distance']:.4f} (distance of selected candidate)")
            print(f"oracle distance     : {metrics['mean_oracle_distance']:.4f} ± {metrics['std_oracle_distance']:.4f} (best achievable distance)")
            print(f"regret              : {metrics['regret']:.4f} ± {metrics['regret_std']:.4f} (distance gap to optimal)")
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
        wrapper = OnnxFrontierPolicyWrapper(self.model).eval().cpu()

        n_nodes, n_edges, n_candidates = 8, 12, 4
        dummy_inputs = (
            torch.zeros((n_nodes, node_input_dim), dtype=torch.float32),
            torch.zeros((2, n_edges), dtype=torch.int64),
            torch.zeros((n_edges, 1), dtype=torch.float32),
            torch.arange(n_nodes, dtype=torch.int64) % n_candidates,
            torch.ones((n_candidates,), dtype=torch.bool),
        )

        torch.onnx.export(
            wrapper,
            dummy_inputs,
            out_path.as_posix(),
            opset_version=18,
            dynamo=False,
            input_names=[
                "node_features",
                "edge_index",
                "edge_attr",
                "membership",
                "mask",
            ],
            output_names=["logits"],
            dynamic_axes={
                "node_features": {0: "N"},
                "edge_index": {1: "E"},
                "edge_attr": {0: "E"},
                "membership": {0: "N"},
                "mask": {0: "F"},
                "logits": {0: "F"},
            },
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
            "final_chosen_distance": final.get("mean_chosen_distance", 0.0),
            "final_chosen_distance_std": final.get("std_chosen_distance", 0.0),
            "final_oracle_distance": final.get("mean_oracle_distance", 0.0),
            "final_oracle_distance_std": final.get("std_oracle_distance", 0.0),
            "num_train_frontiers": len(train_sizes),
            "num_eval_frontiers": len(eval_sizes),
            "mean_train_frontier_size": self._safe_mean(train_sizes),
            "mean_eval_frontier_size": self._safe_mean(eval_sizes),
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

        self._save_metrics_summary(
            output_dir=output_dir,
            history=history,
            eval_snapshots=eval_snapshots,
            train_loader=train_loader,
            eval_loader=eval_loader,
            best_eval_epoch=best_eval_epoch,
        )

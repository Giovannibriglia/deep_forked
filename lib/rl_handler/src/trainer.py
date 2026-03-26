from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

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
        history: Dict[str, List[float]] = {
            "train_reward": [],
            "eval_reward": [],
            "eval_regret": [],
            "eval_oracle_accuracy": [],
            "eval_chosen_distance": [],
            "eval_oracle_distance": [],
        }

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
                history["eval_reward"].append(metrics["mean_reward"])
                history["eval_regret"].append(metrics["regret"])
                history["eval_oracle_accuracy"].append(metrics["oracle_accuracy"])
                history["eval_chosen_distance"].append(metrics["mean_chosen_distance"])
                history["eval_oracle_distance"].append(metrics["mean_oracle_distance"])

                if metrics["mean_reward"] > best_eval_reward:
                    best_eval_reward = metrics["mean_reward"]
                    self.save_model(checkpoint / f"{model_name}_{epoch}.pt", metrics=metrics)
                epoch_pbar.set_postfix(
                    train_reward=f"{mean_train_reward:.4f}",
                    eval_reward=f"{metrics['mean_reward']:.4f}",
                    regret=f"{metrics['regret']:.4f}",
                )
            else:
                epoch_pbar.set_postfix(train_reward=f"{mean_train_reward:.4f}")

        self._save_history_and_plots(history, checkpoint)
        return history

    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        chosen_rewards = []
        chosen_distances = []
        best_distances = []
        oracle_matches = []

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

                for i in range(frontier_ptr.numel() - 1):
                    s = int(frontier_ptr[i].item())
                    e = int(frontier_ptr[i + 1].item())
                    local_logits = logits[s:e]
                    local_rewards = rewards[s:e]
                    local_distances = distances[s:e]
                    chosen_local = int(torch.argmax(local_logits).item())
                    chosen_rewards.append(float(local_rewards[chosen_local].item()))
                    chosen_distances.append(float(local_distances[chosen_local].item()))

                    oracle_local = int(torch.argmin(local_distances).item())
                    best_distances.append(float(local_distances[oracle_local].item()))
                    oracle_matches.append(float(chosen_local == oracle_local))

        chosen_arr = torch.tensor(chosen_distances, dtype=torch.float32)
        best_arr = torch.tensor(best_distances, dtype=torch.float32)
        rewards_arr = torch.tensor(chosen_rewards, dtype=torch.float32)
        regret_arr = chosen_arr - best_arr

        return {
            "mean_reward": float(rewards_arr.mean().item()) if rewards_arr.numel() else 0.0,
            "std_reward": float(rewards_arr.std(unbiased=False).item()) if rewards_arr.numel() else 0.0,
            "mean_chosen_distance": float(chosen_arr.mean().item()) if chosen_arr.numel() else 0.0,
            "mean_oracle_distance": float(best_arr.mean().item()) if best_arr.numel() else 0.0,
            "oracle_accuracy": float(sum(oracle_matches) / len(oracle_matches)) if oracle_matches else 0.0,
            "regret": float(regret_arr.mean().item()) if regret_arr.numel() else 0.0,
        }

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
    def _save_history_and_plots(history: Dict[str, List[float]], output_dir: Path) -> None:
        with (output_dir / "history_losses.json").open("w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)

        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(history["train_reward"], label="train_reward")
        plt.xlabel("Epoch")
        plt.ylabel("Expected Reward")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / "train_reward.png")
        plt.close()

        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(history["eval_reward"], label="eval_reward")
        plt.xlabel("Eval step")
        plt.ylabel("Reward")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / "eval_reward.png")
        plt.close()

        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(history["eval_regret"], label="eval_regret")
        plt.xlabel("Eval step")
        plt.ylabel("Regret")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / "eval_regret.png")
        plt.close()

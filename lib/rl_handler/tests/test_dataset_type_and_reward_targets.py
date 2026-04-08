from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


RL_HANDLER_ROOT = Path(__file__).resolve().parents[1]
if str(RL_HANDLER_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_HANDLER_ROOT))

from src.data import (
    FrontierRow,
    build_frontier_samples,
    build_stress_eval_frontiers_for_dataset,
    group_clean_frontiers,
    normalize_distance_to_reward,
)
from src.graph_utils import TWO_48_MINUS_1, load_pyg_graph
from src.models.frontier_policy import FrontierPolicyNetwork
from src.trainer import RLFrontierTrainer


def _write_dot(
    path: Path,
    nodes: list[str],
    edges: list[tuple[str, str, int]],
) -> None:
    lines = ["digraph G {"]
    for node in nodes:
        lines.append(f'  "{node}" [shape="circle"];')
    for src, dst, label in edges:
        lines.append(f'  "{src}" -> "{dst}" [label="{label}"];')
    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")


class _ConstantLogitModel(nn.Module):
    def __init__(self, logits: list[float]):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, **kwargs) -> torch.Tensor:
        return self.logits


class TestDatasetTypeAndRewardTargets(unittest.TestCase):
    def test_distance_to_reward_mapping_new_scale(self) -> None:
        self.assertAlmostEqual(normalize_distance_to_reward(0.0, max_regular_distance=50.0), 0.0)
        self.assertAlmostEqual(
            normalize_distance_to_reward(25.0, max_regular_distance=50.0),
            -0.35,
            places=6,
        )
        self.assertAlmostEqual(
            normalize_distance_to_reward(50.0, max_regular_distance=50.0),
            -0.7,
            places=6,
        )
        self.assertAlmostEqual(
            normalize_distance_to_reward(50.1, max_regular_distance=50.0),
            -1.0,
            places=6,
        )

    def test_group_clean_frontiers_removes_singleton_and_all_failure(self) -> None:
        rows = [
            FrontierRow("s0.dot", 0, 10.0, "pred_keep", ""),
            FrontierRow("s1.dot", 0, 80.0, "pred_keep", ""),
            FrontierRow("s2.dot", 0, 1.0, "pred_singleton", ""),
            FrontierRow("s3.dot", 0, 90.0, "pred_all_failure", ""),
            FrontierRow("s4.dot", 0, 100.0, "pred_all_failure", ""),
        ]
        cleaned = group_clean_frontiers(
            rows=rows,
            kind_of_data="merged",
            max_regular_distance_for_reward=50.0,
            failure_reward_value=-1.0,
        )
        self.assertEqual(len(cleaned), 1)
        only = cleaned[0]
        self.assertEqual(only["predecessor_path"], "pred_keep")
        self.assertEqual(len(only["successor_paths"]), 2)
        self.assertEqual(len(only["rewards"]), 2)
        self.assertGreater(float(max(only["rewards"])), -1.0)

    def test_build_frontier_samples_builds_random_and_stress_evals(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "training_data" / "prob_0"
            root.mkdir(parents=True, exist_ok=True)

            succ_a = root / "succ_a.dot"
            succ_b = root / "succ_b.dot"
            succ_c = root / "succ_c.dot"
            _write_dot(succ_a, nodes=["1", "2"], edges=[("1", "2", 1)])
            _write_dot(succ_b, nodes=["2", "3"], edges=[("2", "3", 2)])
            _write_dot(succ_c, nodes=["3", "4"], edges=[("3", "4", 3)])

            csv_path = root / "frontiers.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "File Path,Depth,Distance From Goal,Goal,File Path Predecessor,Action",
                        f"{succ_a.as_posix()},0,0.0,,pred0,0",
                        f"{succ_b.as_posix()},0,5.0,,pred0,1",
                        f"{succ_a.as_posix()},0,3.0,,pred1,0",
                        f"{succ_c.as_posix()},0,8.0,,pred1,1",
                        f"{succ_c.as_posix()},0,200.0,,pred_fail,0",
                        f"{succ_b.as_posix()},0,300.0,,pred_fail,1",
                    ]
                ),
                encoding="utf-8",
            )

            train_samples, eval_samples, eval2_samples, eval3_samples, params = build_frontier_samples(
                folder_data=(Path(td) / "training_data").as_posix(),
                list_subset_train=[],
                kind_of_data="merged",
                dataset_type="MAPPED",
                seed=42,
                n_max_dataset_queries=8,
                max_size_frontier=5,
            )

            self.assertEqual(len(train_samples), 2)
            self.assertTrue(len(eval_samples) >= 1)
            self.assertTrue(len(eval2_samples) >= 1)
            self.assertTrue(len(eval3_samples) >= 1)
            self.assertTrue(len(eval_samples) <= 8)
            self.assertTrue(len(eval2_samples) <= 8)
            self.assertTrue(len(eval3_samples) <= 8)
            self.assertIn("split_summary", params)
            for sample in eval2_samples:
                rewards = sample["reward_target"]
                best = float(rewards.max().item())
                n_best = int((torch.abs(rewards - best) <= 1e-9).sum().item())
                self.assertEqual(n_best, 1)

    def test_build_frontier_samples_can_skip_eval_build(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "training_data" / "prob_0"
            root.mkdir(parents=True, exist_ok=True)

            succ_a = root / "succ_a.dot"
            succ_b = root / "succ_b.dot"
            _write_dot(succ_a, nodes=["1", "2"], edges=[("1", "2", 1)])
            _write_dot(succ_b, nodes=["2", "3"], edges=[("2", "3", 2)])

            csv_path = root / "frontiers.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "File Path,Depth,Distance From Goal,Goal,File Path Predecessor,Action",
                        f"{succ_a.as_posix()},0,0.0,,pred0,0",
                        f"{succ_b.as_posix()},0,5.0,,pred0,1",
                    ]
                ),
                encoding="utf-8",
            )

            train_samples, eval_samples, eval2_samples, eval3_samples, params = build_frontier_samples(
                folder_data=(Path(td) / "training_data").as_posix(),
                list_subset_train=[],
                kind_of_data="merged",
                dataset_type="MAPPED",
                seed=42,
                build_eval_data=False,
            )

            self.assertEqual(len(train_samples), 1)
            self.assertEqual(len(eval_samples), 0)
            self.assertEqual(len(eval2_samples), 0)
            self.assertEqual(len(eval3_samples), 0)
            self.assertFalse(bool(params.get("build_eval_data", True)))

    def test_stress_eval_builds_distinct_fifo_and_lifo_when_frontier_overflows(self) -> None:
        clean_frontiers = [
            {
                "predecessor_path": "pred0",
                "goal_path": "",
                "successor_paths": ["s1", "s2", "s3", "s4"],
                "distances": [1.0, 2.0, 3.0, 4.0],
                "depths": [1, 1, 1, 1],
                "rewards": [0.0, -0.1, -0.2, -0.3],
            },
            {
                "predecessor_path": "s1",
                "goal_path": "",
                "successor_paths": ["s5", "s6", "s7"],
                "distances": [5.0, 6.0, 7.0],
                "depths": [2, 2, 2],
                "rewards": [-0.1, -0.2, -0.3],
            },
        ]
        fifo, lifo = build_stress_eval_frontiers_for_dataset(
            clean_frontiers=clean_frontiers,
            kind_of_data="merged",
            n_max_dataset_queries=10,
            max_size_frontier=4,
            dataset_tag="synthetic",
        )
        self.assertGreaterEqual(len(fifo), 2)
        self.assertGreaterEqual(len(lifo), 2)
        self.assertEqual(len(fifo), len(lifo))
        self.assertEqual(fifo[0]["successor_paths"], lifo[0]["successor_paths"])
        self.assertNotEqual(fifo[1]["successor_paths"], lifo[1]["successor_paths"])

    def test_hashed_node_ids_are_normalized_in_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dot_path = Path(td) / "hashed.dot"
            raw_ids = [0, 2**48 - 1]
            _write_dot(
                dot_path,
                nodes=[str(x) for x in raw_ids],
                edges=[(str(raw_ids[0]), str(raw_ids[1]), 3)],
            )
            graph = load_pyg_graph(dot_path.as_posix(), dataset_type="HASHED")

            # Graph loader keeps raw IDs.
            self.assertTrue(
                torch.allclose(
                    graph.node_features.view(-1),
                    torch.tensor(raw_ids, dtype=torch.float32),
                )
            )

            model = FrontierPolicyNetwork(
                node_input_dim=1,
                hidden_dim=8,
                gnn_layers=1,
                conv_type="gcn",
                pooling_type="mean",
                dataset_type="HASHED",
            )
            normalized = model.encoder._prepare_node_features(graph.node_features)
            expected = torch.tensor(raw_ids, dtype=torch.float32).view(-1, 1) / TWO_48_MINUS_1
            self.assertTrue(torch.allclose(normalized, expected, atol=1e-6))

    def test_trainer_uses_reward_target_as_optimization_signal(self) -> None:
        model = _ConstantLogitModel([0.0, 0.0])
        trainer = RLFrontierTrainer(model=model, device="cpu")
        batch = {
            "node_features": torch.zeros((1, 1), dtype=torch.float32),
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "edge_attr": torch.zeros((0, 1), dtype=torch.int64),
            "membership": torch.zeros((1,), dtype=torch.long),
            "candidate_batch": torch.tensor([0, 0], dtype=torch.long),
            "frontier_ptr": torch.tensor([0, 2], dtype=torch.long),
            "reward_target": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "rewards": torch.tensor([-99.0, -99.0], dtype=torch.float32),
        }
        loss_terms = trainer.compute_loss(batch)
        self.assertAlmostEqual(float(loss_terms["expected_reward"].item()), 0.5, places=6)
        self.assertAlmostEqual(float(loss_terms["total_loss"].item()), -0.5, places=6)

    def test_evaluate_builds_regimes(self) -> None:
        model = _ConstantLogitModel([0.0, 1.0, 1.0, 0.0])
        trainer = RLFrontierTrainer(model=model, device="cpu")
        batch = {
            "node_features": torch.zeros((1, 1), dtype=torch.float32),
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "edge_attr": torch.zeros((0, 1), dtype=torch.int64),
            "membership": torch.zeros((1,), dtype=torch.long),
            "candidate_batch": torch.tensor([0, 0, 1, 1], dtype=torch.long),
            "frontier_ptr": torch.tensor([0, 2, 4], dtype=torch.long),
            "reward_target": torch.tensor([0.0, -1.0, -1.0, -1.0], dtype=torch.float32),
            "rewards": torch.tensor([0.0, -1.0, -1.0, -1.0], dtype=torch.float32),
        }
        metrics = trainer.evaluate([batch], verbose=False)
        self.assertEqual(int(metrics["n_frontiers"]), 2)
        self.assertIn("regimes", metrics)
        self.assertEqual(int(metrics["regimes"]["all"]["n_frontiers"]), 2)
        self.assertEqual(int(metrics["regimes"]["with_failure"]["n_frontiers"]), 2)
        self.assertEqual(int(metrics["regimes"]["no_failure"]["n_frontiers"]), 0)


if __name__ == "__main__":
    unittest.main()

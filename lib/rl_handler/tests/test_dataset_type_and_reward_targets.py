from __future__ import annotations

import sys
import tempfile
import torch
import unittest
import warnings
from pathlib import Path
from torch import nn

RL_HANDLER_ROOT = Path(__file__).resolve().parents[1]
if str(RL_HANDLER_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_HANDLER_ROOT))

from src.data import (
    FRONTIER_LABEL_COMMON,
    FRONTIER_LABEL_CONSERVATIVE,
    FRONTIER_LABEL_GREEDY,
    FRONTIER_LABEL_RANDOM,
    FrontierRow,
    build_frontier_samples,
    build_tree_strategy_frontiers_for_dataset,
    build_stress_eval_frontiers_for_dataset,
    group_clean_frontiers,
    normalize_distance_to_reward,
)
from src.graph_utils import load_pyg_graph
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

    def test_distance_to_reward_warns_when_distance_exceeds_max(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reward = normalize_distance_to_reward(50.1, max_regular_distance=50.0)
        self.assertAlmostEqual(reward, -1.0, places=6)
        self.assertTrue(
            any(
                issubclass(w.category, RuntimeWarning)
                and "greater than max_regular_distance" in str(w.message)
                for w in caught
            )
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

            self.assertGreaterEqual(len(train_samples), 1)
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

    def test_tree_strategy_frontiers_generation_and_random_ratio(self) -> None:
        rows = [
            FrontierRow("s_a", 1, 1.0, "root", ""),
            FrontierRow("s_b", 1, 70.0, "root", ""),
            FrontierRow("s_c", 2, 3.0, "s_a", ""),
            FrontierRow("s_d", 2, 90.0, "s_a", ""),
            FrontierRow("s_e", 2, 2.0, "s_b", ""),
            FrontierRow("s_f", 2, 110.0, "s_b", ""),
        ]
        frontiers, summary = build_tree_strategy_frontiers_for_dataset(
            rows=rows,
            kind_of_data="merged",
            dataset_id="synthetic/frontiers.csv",
            onnx_frontier_size=4,
            random_frontier_ratio=0.5,
            random_frontier_with_failure_ratio=0.5,
            max_regular_distance_for_reward=50.0,
            failure_reward_value=-1.0,
            seed=7,
        )

        self.assertGreater(len(frontiers), 0)
        valid_labels = {
            FRONTIER_LABEL_GREEDY,
            FRONTIER_LABEL_CONSERVATIVE,
            FRONTIER_LABEL_RANDOM,
            FRONTIER_LABEL_COMMON,
        }
        for frontier in frontiers:
            self.assertIn(str(frontier.get("frontier_label", "")), valid_labels)
            self.assertTrue(1 <= len(frontier["successor_paths"]) <= 4)
            rewards = [float(x) for x in frontier["rewards"]]
            self.assertTrue(any(r > -1.0 + 1e-9 for r in rewards))

        self.assertIn("generated_greedy_conservative_after_dedup", summary)
        self.assertIn("random_generation_stats", summary)
        random_stats = summary["random_generation_stats"]
        self.assertEqual(
            int(random_stats["target_random_frontiers"]),
            int(round(summary["generated_greedy_conservative_after_dedup"] * 0.5)),
        )

    def test_hashed_node_ids_are_normalized_in_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dot_path = Path(td) / "hashed.dot"
            raw_ids = [-(2**63), -1, 0, (2**63) - 1]
            _write_dot(
                dot_path,
                nodes=[str(x) for x in raw_ids],
                edges=[(str(raw_ids[0]), str(raw_ids[1]), 3)],
            )
            graph = load_pyg_graph(dot_path.as_posix(), dataset_type="HASHED")

            # Graph loader keeps scalar signed int64 node IDs.
            expected_chunks = torch.tensor(raw_ids, dtype=torch.int64).view(-1, 1)
            self.assertTrue(
                torch.equal(graph.node_features, expected_chunks)
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
            raw_ids_f64 = torch.tensor(raw_ids, dtype=torch.float64)
            expected = torch.where(
                raw_ids_f64 >= 0.0,
                raw_ids_f64 / float((1 << 63) - 1),
                raw_ids_f64 / float(1 << 63),
            ).to(torch.float32).view(-1, 1)
            self.assertTrue(torch.allclose(normalized, expected, atol=1e-6))
            label_ids = model.encoder._node_label_ids(
                graph.node_features,
                device=torch.device("cpu"),
            )
            expected_label_ids = torch.remainder(
                torch.tensor(raw_ids, dtype=torch.long),
                int(model.encoder.num_node_labels),
            ).to(torch.long)
            self.assertTrue(torch.equal(label_ids, expected_label_ids))

    def test_edge_label_ids_are_bucketed_with_abs_mod_k(self) -> None:
        model = FrontierPolicyNetwork(
            node_input_dim=1,
            hidden_dim=8,
            gnn_layers=1,
            conv_type="gcn",
            pooling_type="mean",
            dataset_type="MAPPED",
            num_edge_labels=8,
        )
        i64_min = torch.iinfo(torch.long).min
        raw_edge_ids = torch.tensor(
            [[-9], [0], [1], [8], [14], [-14], [i64_min]],
            dtype=torch.int64,
        )
        bucketed = model.encoder._edge_label_ids(
            raw_edge_ids,
            device=torch.device("cpu"),
        )
        expected = torch.tensor([1, 0, 1, 0, 6, 6, 0], dtype=torch.long)
        self.assertTrue(torch.equal(bucketed, expected))
        self.assertTrue(
            bool(((bucketed >= 0) & (bucketed < model.encoder.num_edge_labels)).all().item())
        )

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

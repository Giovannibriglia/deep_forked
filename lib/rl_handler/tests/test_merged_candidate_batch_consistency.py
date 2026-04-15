from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data


RL_HANDLER_ROOT = Path(__file__).resolve().parents[1]
if str(RL_HANDLER_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_HANDLER_ROOT))

from src.data import frontier_collate_fn
from src.graph_utils import combine_graphs
from src.models.frontier_policy import FrontierPolicyNetwork


class TestMergedCandidateBatchConsistency(unittest.TestCase):
    @staticmethod
    def _build_identical_graph() -> Data:
        return Data(
            node_features=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
            node_names=torch.tensor([11.0, 22.0], dtype=torch.float32),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr=torch.tensor([[1.0]], dtype=torch.float32),
        )

    def _build_sample(self) -> dict:
        g0 = self._build_identical_graph()
        g1 = self._build_identical_graph()
        combined = combine_graphs(
            frontier_graphs=[g0, g1],
            goal_graph=None,
            kind_of_data="merged",
            action_ids=[0, 1],
        )
        sample = {
            "node_features": combined.node_features,
            "edge_index": combined.edge_index,
            "edge_attr": combined.edge_attr,
            "membership": combined.membership,
            "action_map": combined.action_map,
            "reward_target": torch.tensor([0.0, -0.5], dtype=torch.float32),
            "rewards": torch.tensor([0.0, -1.0], dtype=torch.float32),
            "distance_raw": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "distances": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "is_failure": torch.tensor([False, False], dtype=torch.bool),
            "oracle_index": torch.tensor(0, dtype=torch.long),
            "oracle_reward": torch.tensor(0.0, dtype=torch.float32),
            "goal_path": "",
            "predecessor_path": "synthetic.dot",
            "frontier_has_failure": False,
        }
        return sample

    @staticmethod
    def _run_forward(batch: dict) -> torch.Tensor:
        model = FrontierPolicyNetwork(
            node_input_dim=int(batch["node_features"].size(1)),
            hidden_dim=8,
            gnn_layers=1,
            conv_type="gcn",
            pooling_type="mean",
            use_global_context=True,
            mlp_depth=1,
        )
        return model(
            node_features=batch["node_features"],
            edge_index=batch["edge_index"],
            edge_attr=batch["edge_attr"],
            membership=batch["membership"],
            candidate_batch=batch["candidate_batch"],
        )

    def test_merged_composition_keeps_candidates_disconnected(self) -> None:
        sample = self._build_sample()
        self.assertIsNone(sample.get("pool_node_index"))
        self.assertIsNone(sample.get("pool_membership"))
        self.assertTrue(
            torch.equal(
                sample["membership"],
                torch.tensor([0, 0, 1, 1], dtype=torch.long),
            )
        )
        self.assertTrue(
            torch.equal(
                sample["edge_index"],
                torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
            )
        )

    def test_merged_batch_forward_does_not_require_pool_tensors(self) -> None:
        sample = self._build_sample()
        batch = frontier_collate_fn([sample], pad_frontiers=True)
        self.assertNotIn("pool_node_index", batch)
        self.assertNotIn("pool_membership", batch)
        logits = self._run_forward(batch)
        self.assertEqual(int(logits.numel()), int(batch["candidate_batch"].numel()))


if __name__ == "__main__":
    unittest.main()

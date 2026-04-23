from __future__ import annotations

import sys
import unittest
from pathlib import Path

RL_HANDLER_ROOT = Path(__file__).resolve().parents[1]
if str(RL_HANDLER_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_HANDLER_ROOT))

from src.data import (
    FRONTIER_LABEL_CONSERVATIVE,
    FRONTIER_LABEL_GREEDY,
    _prune_frontiers_by_jaccard_similarity,
)


def _frontier(
    dataset_id: str,
    successor_paths: list[str],
    rewards: list[float],
    frontier_label: str = FRONTIER_LABEL_GREEDY,
) -> dict:
    return {
        "dataset_id": str(dataset_id),
        "frontier_label": str(frontier_label),
        "successor_paths": list(successor_paths),
        "rewards": [float(x) for x in rewards],
    }


class TestJaccardFrontierPruningScope(unittest.TestCase):
    def test_prunes_same_dataset_and_same_size_even_across_labels_and_failure_mix(self) -> None:
        frontiers = [
            _frontier(
                dataset_id="d0",
                successor_paths=["a", "b"],
                rewards=[0.0, -1.0],
                frontier_label=FRONTIER_LABEL_GREEDY,
            ),
            _frontier(
                dataset_id="d0",
                successor_paths=["b", "a"],
                rewards=[0.0, 0.0],
                frontier_label=FRONTIER_LABEL_CONSERVATIVE,
            ),
        ]

        kept, stats = _prune_frontiers_by_jaccard_similarity(
            frontiers=frontiers,
            jaccard_similarity_threshold=0.95,
            failure_reward_value=-1.0,
        )

        self.assertEqual(len(kept), 1)
        self.assertEqual(int(stats["n_dropped"]), 1)

    def test_does_not_prune_across_datasets(self) -> None:
        frontiers = [
            _frontier(dataset_id="d0", successor_paths=["a", "b"], rewards=[0.0, 0.0]),
            _frontier(dataset_id="d1", successor_paths=["a", "b"], rewards=[0.0, 0.0]),
        ]

        kept, stats = _prune_frontiers_by_jaccard_similarity(
            frontiers=frontiers,
            jaccard_similarity_threshold=0.95,
            failure_reward_value=-1.0,
        )

        self.assertEqual(len(kept), 2)
        self.assertEqual(int(stats["n_dropped"]), 0)

    def test_does_not_prune_across_frontier_sizes(self) -> None:
        frontiers = [
            _frontier(dataset_id="d0", successor_paths=["a", "b"], rewards=[0.0, 0.0]),
            _frontier(dataset_id="d0", successor_paths=["a", "b", "c"], rewards=[0.0, 0.0, 0.0]),
        ]

        kept, stats = _prune_frontiers_by_jaccard_similarity(
            frontiers=frontiers,
            jaccard_similarity_threshold=0.50,
            failure_reward_value=-1.0,
        )

        self.assertEqual(len(kept), 2)
        self.assertEqual(int(stats["n_dropped"]), 0)


if __name__ == "__main__":
    unittest.main()

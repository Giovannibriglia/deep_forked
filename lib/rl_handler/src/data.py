from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.graph_utils import combine_graphs, load_pyg_graph


CSV_COLUMNS = [
    "File Path",
    "Depth",
    "Distance From Goal",
    "Goal",
    "File Path Predecessor",
    "Action",
]


@dataclass
class FrontierRow:
    successor_path: str
    depth: int
    distance: float
    goal_path: str
    predecessor_path: str
    action: str


def _safe_int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_action_id(action_value: str, fallback: int) -> int:
    try:
        return int(action_value)
    except (TypeError, ValueError):
        return fallback


def read_frontier_csv(csv_path: Path) -> List[FrontierRow]:
    rows: List[FrontierRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in CSV_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path}: missing CSV columns {missing}")
        for record in reader:
            rows.append(
                FrontierRow(
                    successor_path=record["File Path"],
                    depth=_safe_int(record["Depth"]),
                    distance=_safe_float(record["Distance From Goal"]),
                    goal_path=record["Goal"],
                    predecessor_path=record["File Path Predecessor"],
                    action=record["Action"],
                )
            )
    return rows


def group_frontiers(rows: Sequence[FrontierRow]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[FrontierRow]] = {}
    for row in rows:
        grouped.setdefault(row.predecessor_path, []).append(row)

    samples: List[Dict[str, Any]] = []
    for predecessor, group_rows in grouped.items():
        if not group_rows:
            continue
        goal_path = group_rows[0].goal_path
        samples.append(
            {
                "predecessor_path": predecessor,
                "goal_path": goal_path,
                "successor_paths": [r.successor_path for r in group_rows],
                "distances": [float(r.distance) for r in group_rows],
                "depths": [int(r.depth) for r in group_rows],
                "actions": [r.action for r in group_rows],
            }
        )
    return samples


def split_frontiers(
    samples: Sequence[Dict[str, Any]],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(samples) <= 1:
        return list(samples), []
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_eval = max(1, int(round(len(samples) * test_size)))
    eval_idx = set(idx[:n_eval])
    train_samples = [samples[i] for i in range(len(samples)) if i not in eval_idx]
    eval_samples = [samples[i] for i in range(len(samples)) if i in eval_idx]
    return train_samples, eval_samples


def build_frontier_samples(
    folder_data: str,
    list_subset_train: Sequence[str],
    kind_of_data: str,
    dataset_type: str,
    test_size: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    root = Path(folder_data)
    bitmask = dataset_type == "BITMASK"

    all_frontiers: List[Dict[str, Any]] = []
    for prob_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if list_subset_train and os.path.basename(prob_dir) not in list_subset_train:
            continue
        csv_files = sorted(p for p in prob_dir.iterdir() if p.suffix == ".csv")
        if not csv_files:
            continue
        for csv_path in csv_files:
            rows = read_frontier_csv(csv_path)
            all_frontiers.extend(group_frontiers(rows))

    train_meta, eval_meta = split_frontiers(all_frontiers, test_size=test_size, seed=seed)
    train_samples = [
        _materialize_frontier(frontier, kind_of_data=kind_of_data, bitmask=bitmask)
        for frontier in tqdm(train_meta, desc="Building train frontiers")
    ]
    eval_samples = [
        _materialize_frontier(frontier, kind_of_data=kind_of_data, bitmask=bitmask)
        for frontier in tqdm(eval_meta, desc="Building eval frontiers")
    ]
    params = {
        "folder_data": str(root),
        "kind_of_data": kind_of_data,
        "dataset_type": dataset_type,
        "num_frontiers_total": len(all_frontiers),
        "num_frontiers_train": len(train_samples),
        "num_frontiers_eval": len(eval_samples),
        "seed": seed,
        "test_size": test_size,
    }
    return train_samples, eval_samples, params


def _materialize_frontier(
    frontier: Dict[str, Any],
    kind_of_data: str,
    bitmask: bool,
) -> Dict[str, Any]:
    successor_graphs = [load_pyg_graph(p, bitmask=bitmask) for p in frontier["successor_paths"]]
    goal_graph = load_pyg_graph(frontier["goal_path"], bitmask=bitmask)

    action_ids = [_to_action_id(a, idx) for idx, a in enumerate(frontier["actions"])]
    combined = combine_graphs(
        frontier_graphs=successor_graphs,
        goal_graph=goal_graph,
        kind_of_data=kind_of_data,
        action_ids=action_ids,
    )

    distances = torch.tensor(frontier["distances"], dtype=torch.float32)
    rewards = -distances
    best_idx = int(torch.argmin(distances).item())
    return {
        "node_features": combined.node_features,
        "edge_index": combined.edge_index,
        "edge_attr": combined.edge_attr,
        "membership": combined.membership,
        "action_map": combined.action_map,
        "rewards": rewards,
        "distances": distances,
        "oracle_index": torch.tensor(best_idx, dtype=torch.long),
        "oracle_reward": rewards[best_idx].clone(),
        "goal_path": frontier["goal_path"],
        "predecessor_path": frontier["predecessor_path"],
    }


class FrontierDataset(Dataset):
    def __init__(self, samples: Sequence[Dict[str, Any]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def frontier_collate_fn(batch: Sequence[Dict[str, Any]], pad_frontiers: bool = True) -> Dict[str, Any]:
    node_parts: List[torch.Tensor] = []
    edge_index_parts: List[torch.Tensor] = []
    edge_attr_parts: List[torch.Tensor] = []
    membership_parts: List[torch.Tensor] = []
    action_parts: List[torch.Tensor] = []
    reward_parts: List[torch.Tensor] = []
    distance_parts: List[torch.Tensor] = []
    oracle_parts: List[torch.Tensor] = []
    oracle_reward_parts: List[torch.Tensor] = []
    candidate_batch_parts: List[torch.Tensor] = []
    frontier_ptr = [0]

    node_offset = 0
    membership_offset = 0
    for batch_idx, item in enumerate(batch):
        n_nodes = int(item["node_features"].size(0))
        n_candidates = int(item["rewards"].size(0))

        node_parts.append(item["node_features"])
        membership = item["membership"].clone()
        candidate_mask = membership >= 0
        membership[candidate_mask] += membership_offset
        membership_parts.append(membership)

        if item["edge_index"].numel() > 0:
            edge_index_parts.append(item["edge_index"] + node_offset)
            edge_attr_parts.append(item["edge_attr"].to(torch.float32))

        action_parts.append(item["action_map"])
        reward_parts.append(item["rewards"])
        distance_parts.append(item["distances"])
        oracle_parts.append(item["oracle_index"].view(1) + membership_offset)
        oracle_reward_parts.append(item["oracle_reward"].view(1))
        candidate_batch_parts.append(torch.full((n_candidates,), batch_idx, dtype=torch.long))

        frontier_ptr.append(frontier_ptr[-1] + n_candidates)
        node_offset += n_nodes
        membership_offset += n_candidates

    out = {
        "node_features": torch.cat(node_parts, dim=0),
        "edge_index": (
            torch.cat(edge_index_parts, dim=1)
            if edge_index_parts
            else torch.zeros((2, 0), dtype=torch.long)
        ),
        "edge_attr": (
            torch.cat(edge_attr_parts, dim=0)
            if edge_attr_parts
            else torch.zeros((0, 1), dtype=torch.float32)
        ),
        "membership": torch.cat(membership_parts, dim=0),
        "action_map": torch.cat(action_parts, dim=0),
        "rewards": torch.cat(reward_parts, dim=0),
        "distances": torch.cat(distance_parts, dim=0),
        "oracle_index": torch.cat(oracle_parts, dim=0),
        "oracle_reward": torch.cat(oracle_reward_parts, dim=0),
        "candidate_batch": torch.cat(candidate_batch_parts, dim=0),
        "frontier_ptr": torch.tensor(frontier_ptr, dtype=torch.long),
    }

    if pad_frontiers:
        max_n = max(x.size(0) for x in reward_parts)
        bsz = len(batch)
        mask = torch.zeros((bsz, max_n), dtype=torch.bool)
        padded_rewards = torch.zeros((bsz, max_n), dtype=torch.float32)
        padded_distances = torch.zeros((bsz, max_n), dtype=torch.float32)
        padded_actions = torch.full((bsz, max_n), -1, dtype=torch.long)
        for i, (r, d, a) in enumerate(zip(reward_parts, distance_parts, action_parts)):
            n = int(r.size(0))
            mask[i, :n] = True
            padded_rewards[i, :n] = r
            padded_distances[i, :n] = d
            padded_actions[i, :n] = a
        out["frontier_mask"] = mask
        out["padded_rewards"] = padded_rewards
        out["padded_distances"] = padded_distances
        out["padded_actions"] = padded_actions

    return out


def get_dataloaders(
    train_samples: Sequence[Dict[str, Any]],
    eval_samples: Sequence[Dict[str, Any]],
    batch_size: int = 8,
    seed: int = 42,
    num_workers: int = 0,
    pad_frontiers: bool = True,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _loader(samples: Sequence[Dict[str, Any]], shuffle: bool) -> DataLoader:
        return DataLoader(
            FrontierDataset(samples),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: frontier_collate_fn(x, pad_frontiers=pad_frontiers),
            generator=generator,
        )

    return _loader(train_samples, True), _loader(eval_samples, False)


def save_samples(
    out_path: str | Path,
    train_samples: Sequence[Dict[str, Any]],
    eval_samples: Sequence[Dict[str, Any]],
    params: Optional[Dict[str, Any]] = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_samples": list(train_samples),
            "eval_samples": list(eval_samples),
            "params": params or {},
        },
        out_path,
    )
    return out_path


def load_saved_samples(path: str | Path) -> Dict[str, Any]:
    payload = torch.load(path, weights_only=False)
    return payload


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

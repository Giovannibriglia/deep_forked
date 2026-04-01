from __future__ import annotations

import csv
import math
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


def _is_failure_candidate(distance: float, m_failed_state: float) -> bool:
    return math.isclose(float(distance), float(m_failed_state), rel_tol=0.0, abs_tol=1e-9)


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


def _sample_failure_frontier_indices(
    samples: Sequence[Dict[str, Any]],
    test_size: float,
    seed: int,
    m_failed_state: float,
    max_percentage_of_failure_states: float,
    max_percentage_of_failure_states_test: float,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    rng = random.Random(seed)
    max_percentage_of_failure_states = min(
        1.0, max(0.0, float(max_percentage_of_failure_states))
    )
    max_percentage_of_failure_states_test = min(
        1.0, max(0.0, float(max_percentage_of_failure_states_test))
    )

    failure_indices: List[int] = []
    non_failure_indices: List[int] = []
    for idx, sample in enumerate(samples):
        distances = [float(d) for d in sample.get("distances", [])]
        has_failure = any(_is_failure_candidate(d, m_failed_state) for d in distances)
        if has_failure:
            failure_indices.append(idx)
        else:
            non_failure_indices.append(idx)

    # Keep non-failure splitting unchanged ("as usual"), with deterministic seed.
    if len(non_failure_indices) <= 1:
        train_non_failure_indices = list(non_failure_indices)
        eval_non_failure_indices = []
    else:
        non_failure_shuffled = list(non_failure_indices)
        rng.shuffle(non_failure_shuffled)
        n_eval_non_failure = max(1, int(round(len(non_failure_shuffled) * test_size)))
        eval_non_failure_set = set(non_failure_shuffled[:n_eval_non_failure])
        train_non_failure_indices = [
            i for i in non_failure_indices if i not in eval_non_failure_set
        ]
        eval_non_failure_indices = [i for i in non_failure_indices if i in eval_non_failure_set]

    rng.shuffle(failure_indices)
    total_failure = len(failure_indices)
    n_failure_train_target = min(
        total_failure,
        max(0, int(total_failure * max_percentage_of_failure_states)),
    )
    n_failure_eval_target = min(
        total_failure,
        max(0, int(total_failure * max_percentage_of_failure_states_test)),
    )

    train_failure_indices = failure_indices[:n_failure_train_target]
    remaining_failure_indices = failure_indices[n_failure_train_target:]

    eval_failure_indices = remaining_failure_indices[:n_failure_eval_target]
    needed = n_failure_eval_target - len(eval_failure_indices)
    reused_train_failure_indices: List[int] = []
    if needed > 0:
        reused_train_failure_indices = train_failure_indices[:needed]
        eval_failure_indices.extend(reused_train_failure_indices)

    train_indices = train_non_failure_indices + train_failure_indices
    eval_indices = eval_non_failure_indices + eval_failure_indices

    # Keep dataset usable in degenerate settings (e.g., all-failure data with very low
    # requested train percentage that rounds to zero).
    if samples and not train_indices:
        fallback_idx = failure_indices[0] if failure_indices else 0
        train_indices.append(fallback_idx)
        eval_indices = [i for i in eval_indices if i != fallback_idx]
        if fallback_idx in failure_indices and fallback_idx not in train_failure_indices:
            train_failure_indices.append(fallback_idx)

    train_failure_set = set(train_failure_indices)
    eval_failure_set = set(eval_failure_indices)
    overlap_failure = train_failure_set.intersection(eval_failure_set)
    disjoint_eval_failure_count = len(eval_failure_set - train_failure_set)
    achieved_train_failure_pct = (
        len(train_failure_indices) / total_failure if total_failure else 0.0
    )
    achieved_eval_failure_pct = (
        len(eval_failure_indices) / total_failure if total_failure else 0.0
    )

    split_summary = {
        "m_failed_state": float(m_failed_state),
        "test_size": float(test_size),
        "requested_max_percentage_of_failure_states_train": float(max_percentage_of_failure_states),
        "requested_max_percentage_of_failure_states_eval": float(max_percentage_of_failure_states_test),
        "num_frontiers_total": len(samples),
        "num_frontiers_train": len(train_indices),
        "num_frontiers_eval": len(eval_indices),
        "num_failure_frontiers_total": total_failure,
        "num_failure_frontiers_train": len(train_failure_indices),
        "num_failure_frontiers_eval": len(eval_failure_indices),
        "achieved_failure_percentage_train": float(achieved_train_failure_pct),
        "achieved_failure_percentage_eval": float(achieved_eval_failure_pct),
        "num_disjoint_eval_failure_frontiers": int(disjoint_eval_failure_count),
        "num_reused_training_failure_frontiers_in_eval": int(len(overlap_failure)),
        "num_failure_overlap_train_eval": int(len(overlap_failure)),
        # Failure frontier == frontier containing at least one failure candidate.
        "failure_frontier_definition": "contains at least one candidate with Distance From Goal == m_failed_state",
    }
    return train_indices, eval_indices, split_summary


def build_frontier_samples(
    folder_data: str,
    list_subset_train: Sequence[str],
    kind_of_data: str,
    dataset_type: str,
    test_size: float,
    seed: int,
    m_failed_state: int = 100000,
    max_percentage_of_failure_states: float = 0.1,
    max_percentage_of_failure_states_test: float = 0.2,
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

    train_idx, eval_idx, split_summary = _sample_failure_frontier_indices(
        samples=all_frontiers,
        test_size=test_size,
        seed=seed,
        m_failed_state=float(m_failed_state),
        max_percentage_of_failure_states=max_percentage_of_failure_states,
        max_percentage_of_failure_states_test=max_percentage_of_failure_states_test,
    )
    train_meta = [all_frontiers[i] for i in train_idx]
    eval_meta = [all_frontiers[i] for i in eval_idx]
    train_samples = [
        _materialize_frontier(
            frontier,
            kind_of_data=kind_of_data,
            bitmask=bitmask,
            m_failed_state=float(m_failed_state),
        )
        for frontier in tqdm(train_meta, desc="Building train frontiers")
    ]
    eval_samples = [
        _materialize_frontier(
            frontier,
            kind_of_data=kind_of_data,
            bitmask=bitmask,
            m_failed_state=float(m_failed_state),
        )
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
        "m_failed_state": float(m_failed_state),
        "max_percentage_of_failure_states": float(max_percentage_of_failure_states),
        "max_percentage_of_failure_states_test": float(max_percentage_of_failure_states_test),
        "split_summary": split_summary,
    }
    return train_samples, eval_samples, params


def _materialize_frontier(
    frontier: Dict[str, Any],
    kind_of_data: str,
    bitmask: bool,
    m_failed_state: float,
) -> Dict[str, Any]:
    if kind_of_data not in {"merged", "separated"}:
        raise ValueError(f"Unsupported kind_of_data: {kind_of_data}")

    successor_graphs = [load_pyg_graph(p, bitmask=bitmask) for p in frontier["successor_paths"]]
    goal_graph = None
    if kind_of_data == "separated":
        if not frontier.get("goal_path"):
            raise ValueError(
                "Missing Goal path for kind_of_data='separated'. "
                "Each frontier row must provide a valid Goal graph path."
            )
        goal_graph = load_pyg_graph(frontier["goal_path"], bitmask=bitmask)
    # merged semantics: goal is embedded in successor graphs, Goal CSV path is ignored.

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
    frontier_has_failure = any(_is_failure_candidate(float(d), m_failed_state) for d in frontier["distances"])
    sample = {
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
        "frontier_has_failure": bool(frontier_has_failure),
    }
    if combined.pool_node_index is not None and combined.pool_membership is not None:
        sample["pool_node_index"] = combined.pool_node_index
        sample["pool_membership"] = combined.pool_membership
    if goal_graph is not None:
        sample["goal_node_features"] = goal_graph.node_features
        sample["goal_edge_index"] = goal_graph.edge_index
        sample["goal_edge_attr"] = goal_graph.edge_attr

    return sample


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
    pool_node_index_parts: List[torch.Tensor] = []
    pool_membership_parts: List[torch.Tensor] = []
    has_pool_parts = [("pool_node_index" in item and "pool_membership" in item) for item in batch]
    if any(("pool_node_index" in item) != ("pool_membership" in item) for item in batch):
        raise ValueError("Each sample must include both pool_node_index and pool_membership, or neither.")
    goal_node_parts: List[torch.Tensor] = []
    goal_edge_index_parts: List[torch.Tensor] = []
    goal_edge_attr_parts: List[torch.Tensor] = []
    goal_batch_parts: List[torch.Tensor] = []
    has_goal_parts = [("goal_node_features" in item) for item in batch]

    node_offset = 0
    membership_offset = 0
    goal_node_offset = 0
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
        if has_pool_parts[batch_idx]:
            pool_node_index = item["pool_node_index"].to(torch.long)
            pool_membership = item["pool_membership"].to(torch.long)
            if pool_node_index.dim() != 1 or pool_membership.dim() != 1:
                raise ValueError("pool_node_index and pool_membership must be 1D tensors.")
            if pool_node_index.numel() != pool_membership.numel():
                raise ValueError(
                    "pool_node_index and pool_membership must have the same number of elements."
                )
            if pool_node_index.numel() > 0:
                if int(pool_node_index.min().item()) < 0:
                    raise ValueError("pool_node_index contains negative indices.")
                if int(pool_node_index.max().item()) >= n_nodes:
                    raise ValueError(
                        f"pool_node_index out of bounds: max={int(pool_node_index.max().item())} n_nodes={n_nodes}"
                    )
            if pool_membership.numel() > 0:
                if int(pool_membership.min().item()) < 0:
                    raise ValueError("pool_membership contains negative indices.")
                if int(pool_membership.max().item()) >= n_candidates:
                    raise ValueError(
                        f"pool_membership out of bounds: max={int(pool_membership.max().item())} "
                        f"n_candidates={n_candidates}"
                    )
            pool_node_index_parts.append(pool_node_index + node_offset)
            pool_membership_parts.append(pool_membership + membership_offset)

        if has_goal_parts[batch_idx]:
            goal_nodes = item["goal_node_features"]
            goal_node_parts.append(goal_nodes)
            goal_batch_parts.append(torch.full((goal_nodes.size(0),), batch_idx, dtype=torch.long))
            if item["goal_edge_index"].numel() > 0:
                goal_edge_index_parts.append(item["goal_edge_index"] + goal_node_offset)
                goal_edge_attr_parts.append(item["goal_edge_attr"].to(torch.float32))
            goal_node_offset += int(goal_nodes.size(0))

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
    candidate_batch = out["candidate_batch"]
    frontier_ptr_tensor = out["frontier_ptr"]
    if candidate_batch.dim() != 1:
        raise ValueError(f"candidate_batch must be 1D, got shape {tuple(candidate_batch.shape)}")
    if frontier_ptr_tensor.dim() != 1:
        raise ValueError(f"frontier_ptr must be 1D, got shape {tuple(frontier_ptr_tensor.shape)}")
    if frontier_ptr_tensor.numel() != len(batch) + 1:
        raise ValueError(
            f"frontier_ptr length mismatch: got {frontier_ptr_tensor.numel()} expected {len(batch) + 1}"
        )
    if int(frontier_ptr_tensor[0].item()) != 0:
        raise ValueError("frontier_ptr must start from 0.")
    if bool((frontier_ptr_tensor[1:] < frontier_ptr_tensor[:-1]).any().item()):
        raise ValueError("frontier_ptr must be non-decreasing.")
    total_candidates = int(frontier_ptr_tensor[-1].item())
    if candidate_batch.numel() != total_candidates:
        raise ValueError(
            f"candidate_batch size mismatch: numel={candidate_batch.numel()} total_candidates={total_candidates}"
        )
    if candidate_batch.numel() > 0:
        if int(candidate_batch.min().item()) < 0:
            raise ValueError("candidate_batch contains negative frontier indices.")
        if int(candidate_batch.max().item()) >= len(batch):
            raise ValueError(
                f"candidate_batch frontier index out of range: max={int(candidate_batch.max().item())} "
                f"batch_size={len(batch)}"
            )

    if any(has_pool_parts) and not all(has_pool_parts):
        raise ValueError("Mixed batches with and without merged pool tensors are not supported.")
    if all(has_pool_parts):
        out["pool_node_index"] = (
            torch.cat(pool_node_index_parts, dim=0)
            if pool_node_index_parts
            else torch.zeros((0,), dtype=torch.long)
        )
        out["pool_membership"] = (
            torch.cat(pool_membership_parts, dim=0)
            if pool_membership_parts
            else torch.zeros((0,), dtype=torch.long)
        )
        pool_node_index = out["pool_node_index"]
        pool_membership = out["pool_membership"]
        if pool_node_index.dim() != 1 or pool_membership.dim() != 1:
            raise ValueError("pool_node_index and pool_membership must be 1D after collation.")
        if pool_node_index.numel() != pool_membership.numel():
            raise ValueError(
                "pool_node_index and pool_membership length mismatch after collation."
            )
        if pool_node_index.numel() > 0:
            if int(pool_node_index.min().item()) < 0:
                raise ValueError("pool_node_index contains negative values after collation.")
            if int(pool_node_index.max().item()) >= int(out["node_features"].size(0)):
                raise ValueError(
                    f"pool_node_index out of bounds after collation: max={int(pool_node_index.max().item())} "
                    f"n_nodes={int(out['node_features'].size(0))}"
                )
        if pool_membership.numel() > 0:
            if int(pool_membership.min().item()) < 0:
                raise ValueError("pool_membership contains negative values after collation.")
            if int(pool_membership.max().item()) >= total_candidates:
                raise ValueError(
                    f"pool_membership out of bounds after collation: max={int(pool_membership.max().item())} "
                    f"total_candidates={total_candidates}"
                )
    if any(has_goal_parts) and not all(has_goal_parts):
        raise ValueError("Mixed batches with and without goal tensors are not supported.")
    if all(has_goal_parts) and goal_node_parts:
        out["goal_node_features"] = torch.cat(goal_node_parts, dim=0)
        out["goal_edge_index"] = (
            torch.cat(goal_edge_index_parts, dim=1)
            if goal_edge_index_parts
            else torch.zeros((2, 0), dtype=torch.long)
        )
        out["goal_edge_attr"] = (
            torch.cat(goal_edge_attr_parts, dim=0)
            if goal_edge_attr_parts
            else torch.zeros((0, 1), dtype=torch.float32)
        )
        out["goal_batch"] = torch.cat(goal_batch_parts, dim=0)

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

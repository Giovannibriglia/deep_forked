from __future__ import annotations

import csv
import math
import os
import random
import torch
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from src.graph_utils import VALID_DATASET_TYPES, combine_graphs, load_pyg_graph
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

CSV_FILE_PATH = "File Path"
CSV_DEPTH = "Depth"
CSV_DISTANCE = "Distance From Goal"
CSV_GOAL = "Goal"
CSV_PREDECESSOR = "File Path Predecessor"

CSV_REQUIRED_BASE = [
    CSV_FILE_PATH,
    CSV_DEPTH,
    CSV_DISTANCE,
    CSV_PREDECESSOR,
]

RANDOM_EVAL_FRONTIER_SIZES = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    16,
    24,
    32,
]

FAILURE_EPS = 1e-9
STRESS_SCHEDULE_FIFO = "fifo"
STRESS_SCHEDULE_LIFO = "lifo"
FRONTIER_LABEL_GREEDY = "greedy"
FRONTIER_LABEL_CONSERVATIVE = "conservative"
FRONTIER_LABEL_RANDOM = "random"
FRONTIER_LABEL_COMMON = "common"
FRONTIER_LABEL_ORDER = [
    FRONTIER_LABEL_GREEDY,
    FRONTIER_LABEL_CONSERVATIVE,
    FRONTIER_LABEL_RANDOM,
    FRONTIER_LABEL_COMMON,
]


@dataclass
class FrontierRow:
    successor_path: str
    depth: int
    distance: float
    predecessor_path: str
    goal_path: str


@dataclass
class _TreeState:
    depth: int
    distance: float
    parent_path: str


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


def _is_failure_reward(reward: float, failure_reward_value: float) -> bool:
    return float(reward) <= float(failure_reward_value) + FAILURE_EPS


def normalize_distance_to_reward(
    distance: float,
    m_failed_state: Optional[float] = None,
    max_regular_distance: float = 50.0,
    failure_reward_value: float = -1.0,
) -> float:
    # m_failed_state is intentionally ignored. Reward/failure now depends only on
    # max_regular_distance as requested by the user.
    _ = m_failed_state
    max_dist = max(1e-9, float(max_regular_distance))
    dist = max(0.0, float(distance))
    if dist > max_dist:
        return float(failure_reward_value)
    return -0.9 * dist / max_dist


def refresh_frontier_sample_targets(
    samples: Sequence[Dict[str, Any]],
    m_failed_state: float,
    max_regular_distance_for_reward: float,
    failure_reward_value: float = -1.0,
) -> None:
    # m_failed_state is kept only for API compatibility.
    _ = m_failed_state
    max_dist = max(1e-9, float(max_regular_distance_for_reward))
    failure_value = float(failure_reward_value)

    def _meta_matches(sample: Dict[str, Any]) -> bool:
        prev_max = sample.get("_reward_max_regular_distance")
        prev_failure = sample.get("_reward_failure_value")
        if prev_max is None or prev_failure is None:
            return False
        try:
            return (
                abs(float(prev_max) - max_dist) <= 1e-12
                and abs(float(prev_failure) - failure_value) <= 1e-12
            )
        except (TypeError, ValueError):
            return False

    for sample in samples:
        if (
            _meta_matches(sample)
            and isinstance(sample.get("reward_target"), torch.Tensor)
            and isinstance(sample.get("is_failure"), torch.Tensor)
            and isinstance(sample.get("oracle_index"), torch.Tensor)
            and isinstance(sample.get("oracle_reward"), torch.Tensor)
        ):
            if "frontier_has_failure" not in sample:
                sample["frontier_has_failure"] = bool(sample["is_failure"].to(torch.bool).any().item())
            continue

        distances = sample.get("distance_raw", sample.get("distances"))
        if distances is None:
            continue

        if isinstance(distances, torch.Tensor):
            distances_t = distances.to(torch.float32).view(-1)
        else:
            distances_t = torch.tensor(distances, dtype=torch.float32).view(-1)
        if distances_t.numel() == 0:
            continue

        clamped_distances = torch.clamp_min(distances_t, 0.0)
        reward_targets = -0.9 * clamped_distances / float(max_dist)
        reward_targets = torch.where(
            clamped_distances > float(max_dist),
            torch.full_like(reward_targets, float(failure_value)),
            reward_targets,
        )
        is_failure = reward_targets <= float(failure_value + FAILURE_EPS)
        best_idx = int(torch.argmax(reward_targets).item())

        sample["distance_raw"] = distances_t
        sample["distances"] = distances_t
        sample["reward_target"] = reward_targets
        sample["rewards"] = reward_targets
        sample["is_failure"] = is_failure
        sample["oracle_index"] = torch.tensor(best_idx, dtype=torch.long)
        sample["oracle_reward"] = reward_targets[best_idx]
        sample["frontier_has_failure"] = bool(is_failure.any().item())
        sample["_reward_max_regular_distance"] = float(max_dist)
        sample["_reward_failure_value"] = float(failure_value)


def read_frontier_csv(csv_path: Path, kind_of_data: str) -> List[FrontierRow]:
    rows: List[FrontierRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        missing = [c for c in CSV_REQUIRED_BASE if c not in fieldnames]
        if kind_of_data == "separated" and CSV_GOAL not in fieldnames:
            missing.append(CSV_GOAL)
        if missing:
            raise ValueError(f"{csv_path}: missing CSV columns {missing}")

        for record in reader:
            goal_value = str(record.get(CSV_GOAL, "") or "").strip()
            if kind_of_data == "merged":
                goal_value = ""
            rows.append(
                FrontierRow(
                    successor_path=str(record[CSV_FILE_PATH]).strip(),
                    depth=_safe_int(record[CSV_DEPTH]),
                    distance=_safe_float(record[CSV_DISTANCE]),
                    predecessor_path=str(record[CSV_PREDECESSOR]).strip(),
                    goal_path=goal_value,
                )
            )
    return rows


def group_clean_frontiers(
    rows: Sequence[FrontierRow],
    kind_of_data: str,
    max_regular_distance_for_reward: float,
    failure_reward_value: float = -1.0,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[FrontierRow]] = {}
    for row in rows:
        grouped.setdefault(row.predecessor_path, []).append(row)

    cleaned: List[Dict[str, Any]] = []
    for predecessor, group_rows in grouped.items():
        if len(group_rows) <= 1:
            # Remove frontier with a single candidate.
            continue

        successor_paths = [str(r.successor_path) for r in group_rows]
        distances = [float(r.distance) for r in group_rows]
        depths = [int(r.depth) for r in group_rows]
        rewards = [
            normalize_distance_to_reward(
                distance=float(d),
                max_regular_distance=float(max_regular_distance_for_reward),
                failure_reward_value=float(failure_reward_value),
            )
            for d in distances
        ]

        # Remove frontier where all rewards are failures.
        if all(_is_failure_reward(r, failure_reward_value) for r in rewards):
            continue

        goal_path = ""
        if kind_of_data == "separated":
            goal_values = [r.goal_path for r in group_rows if r.goal_path]
            if not goal_values:
                continue
            goal_path = goal_values[0]
            # Skip ambiguous frontier-goal mappings.
            if any(g != goal_path for g in goal_values):
                continue

        cleaned.append(
            {
                "predecessor_path": predecessor,
                "goal_path": goal_path,
                "successor_paths": successor_paths,
                "distances": distances,
                "depths": depths,
                "rewards": rewards,
            }
        )
    return cleaned


def _ordered_unique_paths(paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in paths:
        path = str(raw).strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _resolve_dataset_goal_path(
    rows: Sequence[FrontierRow],
    kind_of_data: str,
) -> Optional[str]:
    if kind_of_data != "separated":
        return ""
    goal_values = {
        str(row.goal_path).strip()
        for row in rows
        if str(row.goal_path).strip()
    }
    if not goal_values:
        return None
    if len(goal_values) > 1:
        return None
    return next(iter(goal_values))


def _build_tree_structures(
    rows: Sequence[FrontierRow],
) -> Tuple[Dict[str, _TreeState], Dict[str, List[str]], Dict[str, str], List[str]]:
    state_by_path: Dict[str, _TreeState] = {}
    children_by_parent: Dict[str, List[str]] = {}
    parent_by_child: Dict[str, str] = {}
    all_nodes: List[str] = []

    for row in rows:
        child = str(row.successor_path).strip()
        parent = str(row.predecessor_path).strip()
        if not child:
            continue

        all_nodes.append(child)
        if parent:
            all_nodes.append(parent)
            children = children_by_parent.setdefault(parent, [])
            if child not in children:
                children.append(child)

        prev_state = state_by_path.get(child)
        depth_i = int(row.depth)
        distance_f = float(row.distance)
        if (prev_state is None) or (depth_i < int(prev_state.depth)):
            state_by_path[child] = _TreeState(
                depth=depth_i,
                distance=distance_f,
                parent_path=parent,
            )
            parent_by_child[child] = parent
        elif prev_state is not None and parent_by_child.get(child, "") == "":
            parent_by_child[child] = parent

    return state_by_path, children_by_parent, parent_by_child, _ordered_unique_paths(all_nodes)


def _find_initial_states(
    state_by_path: Dict[str, _TreeState],
    children_by_parent: Dict[str, List[str]],
    parent_by_child: Dict[str, str],
) -> List[str]:
    roots = sorted(
        _ordered_unique_paths(
            [
                parent
                for parent in children_by_parent.keys()
                if parent and parent not in parent_by_child
            ]
        )
    )
    if roots:
        return roots

    if state_by_path:
        min_depth = min(int(state.depth) for state in state_by_path.values())
        depth_roots = sorted(
            _ordered_unique_paths(
                [
                    path
                    for path, state in state_by_path.items()
                    if int(state.depth) == int(min_depth)
                ]
            )
        )
        if depth_roots:
            return depth_roots

    return sorted(_ordered_unique_paths(children_by_parent.keys()))


def _push_selected_path(
    selected: List[str],
    selected_set: set[str],
    available_state_set: set[str],
    path: str,
) -> bool:
    path_s = str(path).strip()
    if not path_s or path_s in selected_set or path_s not in available_state_set:
        return False
    selected.append(path_s)
    selected_set.add(path_s)
    return True


def _select_greedy_paths(
    expanded_node: str,
    target_size: int,
    expanded_state_order: Sequence[str],
    expanded_state_set: set[str],
    open_state_order: Sequence[str],
    open_state_set: set[str],
    available_state_set: set[str],
    parent_by_child: Dict[str, str],
    children_by_parent: Dict[str, List[str]],
) -> List[str]:
    k = int(target_size)
    if k <= 0:
        return []

    selected: List[str] = []
    selected_set: set[str] = set()
    expanded = str(expanded_node).strip()

    parent = str(parent_by_child.get(expanded, "")).strip()
    if parent:
        for sibling in children_by_parent.get(parent, []):
            sibling_s = str(sibling).strip()
            if sibling_s == expanded:
                continue
            _push_selected_path(selected, selected_set, available_state_set, sibling_s)
            if len(selected) >= k:
                return selected

    visited_ancestors: set[str] = set()
    current_ancestor = parent
    while current_ancestor and current_ancestor not in visited_ancestors and len(selected) < k:
        visited_ancestors.add(current_ancestor)
        if current_ancestor in expanded_state_set:
            _push_selected_path(selected, selected_set, available_state_set, current_ancestor)
        if len(selected) >= k:
            return selected
        ancestor_parent = str(parent_by_child.get(current_ancestor, "")).strip()
        if ancestor_parent:
            for sibling in children_by_parent.get(ancestor_parent, []):
                sibling_s = str(sibling).strip()
                if sibling_s == current_ancestor or sibling_s not in expanded_state_set:
                    continue
                _push_selected_path(selected, selected_set, available_state_set, sibling_s)
                if len(selected) >= k:
                    return selected
        current_ancestor = ancestor_parent

    for child in children_by_parent.get(expanded, []):
        _push_selected_path(selected, selected_set, available_state_set, str(child))
        if len(selected) >= k:
            return selected

    for path in reversed(open_state_order):
        if path not in open_state_set:
            continue
        _push_selected_path(selected, selected_set, available_state_set, path)
        if len(selected) >= k:
            return selected

    for path in reversed(expanded_state_order):
        if path == expanded:
            continue
        _push_selected_path(selected, selected_set, available_state_set, path)
        if len(selected) >= k:
            return selected
    return selected


def _select_conservative_paths(
    expanded_node: str,
    target_size: int,
    expanded_state_order: Sequence[str],
    open_state_order: Sequence[str],
    open_state_set: set[str],
    available_state_set: set[str],
    parent_by_child: Dict[str, str],
    children_by_parent: Dict[str, List[str]],
) -> List[str]:
    k = int(target_size)
    if k <= 0:
        return []

    selected: List[str] = []
    selected_set: set[str] = set()
    expanded = str(expanded_node).strip()

    ancestor = str(parent_by_child.get(expanded, "")).strip()
    visited_ancestors: set[str] = set()
    while ancestor and ancestor not in visited_ancestors and len(selected) < k:
        visited_ancestors.add(ancestor)
        _push_selected_path(selected, selected_set, available_state_set, ancestor)
        ancestor = str(parent_by_child.get(ancestor, "")).strip()
        if len(selected) >= k:
            return selected

    for child in children_by_parent.get(expanded, []):
        _push_selected_path(selected, selected_set, available_state_set, str(child))
        if len(selected) >= k:
            return selected

    for path in reversed(open_state_order):
        if path not in open_state_set:
            continue
        _push_selected_path(selected, selected_set, available_state_set, path)
        if len(selected) >= k:
            return selected

    for path in reversed(expanded_state_order):
        if path == expanded:
            continue
        _push_selected_path(selected, selected_set, available_state_set, path)
        if len(selected) >= k:
            return selected
    return selected


def _sample_random_paths_from_state_pool(
    all_paths: Sequence[str],
    non_failure_paths: Sequence[str],
    failure_paths: Sequence[str],
    size: int,
    require_failure: bool,
    rng: random.Random,
) -> List[str]:
    frontier_size = int(size)
    if frontier_size <= 0:
        return []
    if frontier_size > len(all_paths):
        return []
    if not non_failure_paths:
        return []
    if require_failure and (not failure_paths):
        return []

    selected: List[str] = []
    selected_non_failure = rng.choice(non_failure_paths)
    selected.append(selected_non_failure)
    selected_set: set[str] = {selected_non_failure}

    if require_failure:
        if not failure_paths:
            return []
        selected_failure = rng.choice(failure_paths)
        if selected_failure in selected_set:
            return []
        selected.append(selected_failure)
        selected_set.add(selected_failure)

    remaining = int(frontier_size - len(selected))
    if remaining < 0:
        return []
    if remaining > 0:
        if int(len(all_paths) - len(selected_set)) < remaining:
            return []
        extra: List[str] = []
        extra_set: set[str] = set()
        max_pick_attempts = max(32, int(8 * len(all_paths)))
        pick_attempts = 0
        while len(extra) < remaining and pick_attempts < max_pick_attempts:
            pick_attempts += 1
            candidate = rng.choice(all_paths)
            if candidate in selected_set or candidate in extra_set:
                continue
            extra.append(candidate)
            extra_set.add(candidate)
        if len(extra) < remaining:
            remaining_pool = [p for p in all_paths if p not in selected_set]
            if len(remaining_pool) < remaining:
                return []
            extra = list(rng.sample(remaining_pool, remaining))
        selected.extend(extra)

    rng.shuffle(selected)
    return selected


def _uniform_size_targets(total: int, sizes: Sequence[int]) -> Dict[int, int]:
    sizes_i = [int(s) for s in sizes]
    if total <= 0 or not sizes_i:
        return {int(s): 0 for s in sizes_i}
    base = int(total // len(sizes_i))
    rem = int(total % len(sizes_i))
    targets: Dict[int, int] = {int(s): int(base) for s in sizes_i}
    for i in range(rem):
        targets[int(sizes_i[i])] = int(targets[int(sizes_i[i])] + 1)
    return targets


def _comb_count(n: int, k: int) -> int:
    n_i = int(n)
    k_i = int(k)
    if k_i < 0 or k_i > n_i:
        return 0
    return int(math.comb(n_i, k_i))


def _clip_targets_and_redistribute(
    target_by_size: Dict[int, int],
    cap_by_size: Dict[int, int],
) -> Dict[int, int]:
    sizes = sorted(int(s) for s in target_by_size.keys())
    out = {
        int(size): int(min(int(target_by_size.get(int(size), 0)), int(cap_by_size.get(int(size), 0))))
        for size in sizes
    }
    target_total = int(sum(int(target_by_size.get(int(size), 0)) for size in sizes))
    current_total = int(sum(out.values()))
    remaining = int(max(0, target_total - current_total))
    while remaining > 0:
        expandable = [
            int(size)
            for size in sizes
            if int(out.get(int(size), 0)) < int(cap_by_size.get(int(size), 0))
        ]
        if not expandable:
            break
        for size in expandable:
            if remaining <= 0:
                break
            out[int(size)] = int(out.get(int(size), 0) + 1)
            remaining -= 1
    return out


def _random_with_failure_targets(
    target_by_size: Dict[int, int],
    total_with_failure: int,
    feasible_with_failure_sizes: Sequence[int],
    cap_by_size: Optional[Dict[int, int]] = None,
) -> Dict[int, int]:
    out = {int(size): 0 for size in target_by_size.keys()}
    feasible_sizes = [int(s) for s in feasible_with_failure_sizes]
    if total_with_failure <= 0 or not feasible_sizes:
        return out

    size_quota = _uniform_size_targets(total=int(total_with_failure), sizes=feasible_sizes)
    for size, quota in size_quota.items():
        out[int(size)] = int(min(int(quota), int(target_by_size.get(int(size), 0))))
        if cap_by_size is not None:
            out[int(size)] = int(min(int(out[int(size)]), int(cap_by_size.get(int(size), 0))))

    assigned = int(sum(out.values()))
    remaining = int(max(0, int(total_with_failure) - assigned))
    if remaining > 0:
        expandable = [
            int(size)
            for size in feasible_sizes
            if int(out.get(int(size), 0))
            < int(
                min(
                    int(target_by_size.get(int(size), 0)),
                    int(cap_by_size.get(int(size), 0))
                    if cap_by_size is not None
                    else int(target_by_size.get(int(size), 0)),
                )
            )
        ]
        idx = 0
        while remaining > 0 and expandable:
            size = int(expandable[idx % len(expandable)])
            max_cap = int(target_by_size.get(size, 0))
            if cap_by_size is not None:
                max_cap = int(min(max_cap, int(cap_by_size.get(size, 0))))
            if int(out.get(size, 0)) < max_cap:
                out[size] = int(out.get(size, 0) + 1)
                remaining -= 1
            expandable = [
                int(s)
                for s in feasible_sizes
                if int(out.get(int(s), 0))
                < int(
                    min(
                        int(target_by_size.get(int(s), 0)),
                        int(cap_by_size.get(int(s), 0))
                        if cap_by_size is not None
                        else int(target_by_size.get(int(s), 0)),
                    )
                )
            ]
            idx += 1
    return out


def _build_frontier_from_paths(
    selected_paths: Sequence[str],
    state_by_path: Dict[str, _TreeState],
    reward_by_path: Dict[str, float],
    label: str,
    predecessor_path: str,
    goal_path: str,
    dataset_id: str,
    failure_reward_value: float,
) -> Optional[Dict[str, Any]]:
    successors = [str(p) for p in selected_paths if str(p) in state_by_path]
    if not successors:
        return None

    distances = [float(state_by_path[path].distance) for path in successors]
    depths = [int(state_by_path[path].depth) for path in successors]
    rewards = [float(reward_by_path[path]) for path in successors]
    has_non_failure = any(
        not _is_failure_reward(float(reward), float(failure_reward_value))
        for reward in rewards
    )
    if not has_non_failure:
        return None

    return {
        "predecessor_path": str(predecessor_path),
        "goal_path": str(goal_path),
        "successor_paths": successors,
        "distances": distances,
        "depths": depths,
        "rewards": rewards,
        "dataset_id": str(dataset_id),
        "frontier_label": str(label),
    }


def _generate_random_frontiers_from_state_pool(
    state_by_path: Dict[str, _TreeState],
    reward_by_path: Dict[str, float],
    dataset_id: str,
    goal_path: str,
    onnx_frontier_size: int,
    random_frontier_ratio: float,
    random_frontier_with_failure_ratio: float,
    reference_frontier_count: int,
    seed: int,
    failure_reward_value: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    all_paths = sorted(_ordered_unique_paths(state_by_path.keys()))
    non_failure_paths = [
        path
        for path in all_paths
        if not _is_failure_reward(float(reward_by_path[path]), float(failure_reward_value))
    ]
    failure_paths = [
        path
        for path in all_paths
        if _is_failure_reward(float(reward_by_path[path]), float(failure_reward_value))
    ]

    max_size = min(max(1, int(onnx_frontier_size)), len(all_paths))
    size_range = [int(s) for s in range(1, max_size + 1)]
    target_total_requested = max(
        0, int(round(float(reference_frontier_count) * float(random_frontier_ratio)))
    )
    n_total = int(len(all_paths))
    n_non_failure = int(len(non_failure_paths))
    n_failure = int(len(failure_paths))
    feasible_any_by_size: Dict[int, int] = {}
    feasible_with_failure_by_size: Dict[int, int] = {}
    feasible_without_failure_by_size: Dict[int, int] = {}
    for size in size_range:
        total_combos = _comb_count(n_total, int(size))
        all_failure_combos = _comb_count(n_failure, int(size))
        all_non_failure_combos = _comb_count(n_non_failure, int(size))
        any_cap = int(max(0, total_combos - all_failure_combos))
        with_failure_cap = int(
            max(0, total_combos - all_failure_combos - all_non_failure_combos)
        )
        without_failure_cap = int(all_non_failure_combos)
        feasible_any_by_size[int(size)] = int(any_cap)
        feasible_with_failure_by_size[int(size)] = int(
            min(int(with_failure_cap), int(any_cap))
        )
        feasible_without_failure_by_size[int(size)] = int(
            min(int(without_failure_cap), int(any_cap))
        )

    max_feasible_total = int(sum(feasible_any_by_size.values()))
    target_total = int(min(int(target_total_requested), int(max_feasible_total)))
    requested_with_failure_total = max(
        0,
        min(
            int(target_total),
            int(round(float(target_total) * float(random_frontier_with_failure_ratio))),
        ),
    )
    feasible_with_failure_sizes = [
        int(s) for s in size_range if int(feasible_with_failure_by_size.get(int(s), 0)) > 0
    ]

    stats: Dict[str, Any] = {
        "reference_frontier_count": int(reference_frontier_count),
        "random_frontier_ratio": float(random_frontier_ratio),
        "random_frontier_with_failure_ratio": float(random_frontier_with_failure_ratio),
        "target_random_frontiers_requested": int(target_total_requested),
        "target_random_frontiers_capped_by_feasibility": int(target_total),
        "target_random_frontiers": int(target_total),
        "target_random_with_failure_frontiers": int(requested_with_failure_total),
        "target_random_with_failure_frontiers_effective": 0,
        "max_feasible_random_frontiers": int(max_feasible_total),
        "generated_random_frontiers": 0,
        "generated_random_with_failure_frontiers": 0,
        "generated_random_without_failure_frontiers": 0,
        "generated_by_size": {},
        "size_range": [int(s) for s in size_range],
    }
    if target_total <= 0 or not size_range or not non_failure_paths:
        return [], stats

    target_by_size = _uniform_size_targets(total=int(target_total), sizes=size_range)
    target_by_size = _clip_targets_and_redistribute(
        target_by_size=target_by_size,
        cap_by_size=feasible_any_by_size,
    )
    requested_with_failure_total = int(
        min(
            int(requested_with_failure_total),
            int(
                sum(
                    min(
                        int(target_by_size.get(int(size), 0)),
                        int(feasible_with_failure_by_size.get(int(size), 0)),
                    )
                    for size in size_range
                )
            ),
        )
    )
    target_with_failure_by_size = _random_with_failure_targets(
        target_by_size=target_by_size,
        total_with_failure=requested_with_failure_total,
        feasible_with_failure_sizes=feasible_with_failure_sizes,
        cap_by_size=feasible_with_failure_by_size,
    )
    for size in size_range:
        total_s = int(target_by_size.get(int(size), 0))
        with_s = int(
            min(
                int(target_with_failure_by_size.get(int(size), 0)),
                int(feasible_with_failure_by_size.get(int(size), 0)),
                int(total_s),
            )
        )
        without_s = int(total_s - with_s)
        without_cap = int(feasible_without_failure_by_size.get(int(size), 0))
        if without_s > without_cap:
            move = int(without_s - without_cap)
            with_s = int(with_s + move)
            without_s = int(without_s - move)
        target_with_failure_by_size[int(size)] = int(with_s)
    effective_with_failure_target = int(sum(target_with_failure_by_size.values()))
    stats["target_random_with_failure_frontiers_effective"] = int(effective_with_failure_target)

    rng = random.Random(int(seed))
    generated: List[Dict[str, Any]] = []
    seen_signatures: set[frozenset[str]] = set()
    generated_by_size: Dict[int, int] = {int(size): 0 for size in size_range}
    generated_with_failure_by_size: Dict[int, int] = {int(size): 0 for size in size_range}
    generated_without_failure_by_size: Dict[int, int] = {int(size): 0 for size in size_range}
    generated_with_failure = 0

    def _try_generate(size: int, require_failure: bool, max_attempts: int) -> bool:
        nonlocal generated_with_failure
        size_i = int(size)
        if int(generated_by_size.get(size_i, 0)) >= int(feasible_any_by_size.get(size_i, 0)):
            return False
        if bool(require_failure):
            if int(generated_with_failure_by_size.get(size_i, 0)) >= int(
                feasible_with_failure_by_size.get(size_i, 0)
            ):
                return False
        else:
            if int(generated_without_failure_by_size.get(size_i, 0)) >= int(
                feasible_without_failure_by_size.get(size_i, 0)
            ):
                return False

        attempts = int(
            max(
                16,
                min(
                    int(max_attempts),
                    int(
                        6
                        * max(
                            1,
                            int(feasible_any_by_size.get(size_i, 0))
                            - int(generated_by_size.get(size_i, 0)),
                        )
                    ),
                ),
            )
        )
        for _ in range(attempts):
            selected_paths = _sample_random_paths_from_state_pool(
                all_paths=all_paths,
                non_failure_paths=non_failure_paths,
                failure_paths=failure_paths,
                size=size_i,
                require_failure=bool(require_failure),
                rng=rng,
            )
            if not selected_paths:
                continue
            signature = frozenset(selected_paths)
            if signature in seen_signatures:
                continue

            frontier = _build_frontier_from_paths(
                selected_paths=selected_paths,
                state_by_path=state_by_path,
                reward_by_path=reward_by_path,
                label=FRONTIER_LABEL_RANDOM,
                predecessor_path=f"random_{dataset_id}_{len(generated):06d}",
                goal_path=str(goal_path),
                dataset_id=str(dataset_id),
                failure_reward_value=float(failure_reward_value),
            )
            if frontier is None:
                continue
            has_failure = any(
                _is_failure_reward(float(r), float(failure_reward_value))
                for r in frontier["rewards"]
            )
            if require_failure and not has_failure:
                continue

            seen_signatures.add(signature)
            generated.append(frontier)
            generated_by_size[size_i] = int(generated_by_size.get(size_i, 0) + 1)
            if has_failure:
                generated_with_failure += 1
                generated_with_failure_by_size[size_i] = int(
                    generated_with_failure_by_size.get(size_i, 0) + 1
                )
            else:
                generated_without_failure_by_size[size_i] = int(
                    generated_without_failure_by_size.get(size_i, 0) + 1
                )
            return True
        return False

    for size in size_range:
        size_target = int(target_by_size.get(int(size), 0))
        if size_target <= 0:
            continue

        target_with_failure = int(target_with_failure_by_size.get(int(size), 0))
        target_without_failure = int(max(0, size_target - target_with_failure))

        while int(generated_with_failure_by_size.get(int(size), 0)) < int(target_with_failure):
            ok = _try_generate(
                size=int(size),
                require_failure=True,
                max_attempts=120,
            )
            if not ok:
                break
        while int(generated_without_failure_by_size.get(int(size), 0)) < int(target_without_failure):
            ok = _try_generate(
                size=int(size),
                require_failure=False,
                max_attempts=100,
            )
            if not ok:
                break

    remaining = int(target_total - len(generated))
    if remaining > 0:
        cycle_sizes = [
            int(size)
            for size in size_range
            if int(generated_by_size.get(int(size), 0)) < int(feasible_any_by_size.get(int(size), 0))
        ]
        idx = 0
        blocked_modes: set[Tuple[int, bool]] = set()
        while remaining > 0 and cycle_sizes:
            size = int(cycle_sizes[idx % len(cycle_sizes)])
            need_failure = generated_with_failure < int(effective_with_failure_target)
            preferred_mode = bool(
                need_failure
                and int(generated_with_failure_by_size.get(size, 0))
                < int(feasible_with_failure_by_size.get(size, 0))
            )

            generated_now = False
            mode_order = [preferred_mode, (not preferred_mode)]
            for mode in mode_order:
                if (size, bool(mode)) in blocked_modes:
                    continue
                generated_now = _try_generate(
                    size=size,
                    require_failure=bool(mode),
                    max_attempts=40,
                )
                if generated_now:
                    break
                blocked_modes.add((size, bool(mode)))

            if generated_now:
                remaining -= 1
                blocked_modes.discard((size, True))
                blocked_modes.discard((size, False))

            if int(generated_by_size.get(size, 0)) >= int(feasible_any_by_size.get(size, 0)):
                cycle_sizes = [s for s in cycle_sizes if int(s) != int(size)]
                idx = 0
            elif (size, True) in blocked_modes and (size, False) in blocked_modes:
                cycle_sizes = [s for s in cycle_sizes if int(s) != int(size)]
                idx = 0
            else:
                idx += 1

    generated_without_failure = int(len(generated) - generated_with_failure)
    stats["generated_random_frontiers"] = int(len(generated))
    stats["generated_random_with_failure_frontiers"] = int(generated_with_failure)
    stats["generated_random_without_failure_frontiers"] = int(generated_without_failure)
    stats["generated_by_size"] = {
        str(size): int(generated_by_size.get(int(size), 0))
        for size in size_range
    }
    return generated, stats


def _deduplicate_strategy_frontiers(
    frontiers: Sequence[Dict[str, Any]],
    kind_of_data: str,
    failure_reward_value: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    deduped: Dict[Tuple[str, str, frozenset[str]], Dict[str, Any]] = {}
    dropped_no_non_failure = 0
    deduplicated_to_common = 0

    for frontier in frontiers:
        rewards = [float(x) for x in frontier.get("rewards", [])]
        if not rewards or all(
            _is_failure_reward(float(r), float(failure_reward_value))
            for r in rewards
        ):
            dropped_no_non_failure += 1
            continue

        dataset_id = str(frontier.get("dataset_id", ""))
        goal = str(frontier.get("goal_path", "")) if kind_of_data == "separated" else ""
        key = (
            dataset_id,
            goal,
            frozenset(frontier.get("successor_paths", [])),
        )
        current = deduped.get(key)
        if current is None:
            deduped[key] = dict(frontier)
            continue

        deduplicated_to_common += 1
        current["frontier_label"] = FRONTIER_LABEL_COMMON

    return list(deduped.values()), {
        "dropped_no_non_failure": int(dropped_no_non_failure),
        "deduplicated_to_common": int(deduplicated_to_common),
    }


def _frontier_successor_path_set(frontier: Dict[str, Any]) -> set[str]:
    successor_paths = frontier.get("successor_paths", [])
    if not isinstance(successor_paths, (list, tuple)):
        return set()
    out: set[str] = set()
    for raw in successor_paths:
        value = str(raw).strip()
        if value:
            out.add(value)
    return out


def _required_overlap_for_jaccard(size_a: int, size_b: int, threshold: float) -> int:
    if threshold <= 0.0:
        return 1
    rhs = float(threshold) * float(int(size_a) + int(size_b)) / float(1.0 + float(threshold))
    return max(1, int(math.ceil(rhs - 1e-12)))


def _prune_frontiers_by_jaccard_similarity(
    frontiers: Sequence[Dict[str, Any]],
    jaccard_similarity_threshold: float,
    failure_reward_value: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    threshold = float(jaccard_similarity_threshold)
    if not frontiers:
        return [], {
            "threshold": float(threshold),
            "n_total": 0,
            "n_kept": 0,
            "n_dropped": 0,
            "n_missing_successor_paths": 0,
            "n_groups": 0,
        }

    groups: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}
    token_to_id: Dict[str, int] = {}
    next_token_id = 0
    kept_frontiers: List[Dict[str, Any]] = []
    dropped = 0
    missing_successor_paths = 0

    for frontier in tqdm(frontiers, desc="Pruning..."):
        successor_set = _frontier_successor_path_set(frontier)
        if not successor_set:
            kept_frontiers.append(frontier)
            missing_successor_paths += 1
            continue
        successor_token_ids: set[int] = set()
        for token in successor_set:
            token_id = token_to_id.get(token)
            if token_id is None:
                token_id = int(next_token_id)
                token_to_id[token] = token_id
                next_token_id += 1
            successor_token_ids.add(int(token_id))

        rewards = [float(x) for x in frontier.get("rewards", [])]
        has_failure = int(
            any(_is_failure_reward(float(r), float(failure_reward_value)) for r in rewards)
        )
        group_key = (
            str(frontier.get("dataset_id", "")),
            str(frontier.get("frontier_label", FRONTIER_LABEL_COMMON)),
            int(len(successor_token_ids)),
            int(has_failure),
        )
        group = groups.setdefault(
            group_key,
            {
                "sizes": [],
                "inverted_index": {},
            },
        )
        sizes = group["sizes"]
        inverted_index = group["inverted_index"]

        overlap_by_candidate: Dict[int, int] = {}
        size_a = int(len(successor_token_ids))
        required_overlap = _required_overlap_for_jaccard(
            size_a=int(size_a),
            size_b=int(size_a),
            threshold=float(threshold),
        )
        seed_token = min(
            successor_token_ids,
            key=lambda token_id: len(inverted_index.get(int(token_id), [])),
        )
        ordered_tokens: List[int] = [int(seed_token)]
        for token_id in successor_token_ids:
            if int(token_id) == int(seed_token):
                continue
            ordered_tokens.append(int(token_id))

        remaining_tokens = int(size_a)
        drop_current = False
        for token_id in ordered_tokens:
            remaining_tokens = int(max(0, remaining_tokens - 1))
            postings = inverted_index.get(int(token_id), [])
            if not postings:
                continue
            for candidate_idx in postings:
                candidate_i = int(candidate_idx)
                new_overlap = int(overlap_by_candidate.get(candidate_i, 0) + 1)
                overlap_by_candidate[candidate_i] = int(new_overlap)
                if int(new_overlap) < int(required_overlap):
                    continue
                size_b = int(sizes[candidate_i])
                denom = int(size_a + size_b - int(new_overlap))
                similarity = (float(new_overlap) / float(denom)) if denom > 0 else 1.0
                if similarity >= threshold:
                    drop_current = True
                    break
            if drop_current:
                break

            if overlap_by_candidate and remaining_tokens > 0 and required_overlap > 1:
                impossible = [
                    candidate_i
                    for candidate_i, overlap in overlap_by_candidate.items()
                    if int(overlap) + int(remaining_tokens) < int(required_overlap)
                ]
                for candidate_i in impossible:
                    del overlap_by_candidate[int(candidate_i)]

        if drop_current:
            dropped += 1
            continue

        local_idx = int(len(sizes))
        sizes.append(size_a)
        for token_id in successor_token_ids:
            bucket = inverted_index.setdefault(int(token_id), [])
            bucket.append(local_idx)
        kept_frontiers.append(frontier)

    return kept_frontiers, {
        "threshold": float(threshold),
        "n_total": int(len(frontiers)),
        "n_kept": int(len(kept_frontiers)),
        "n_dropped": int(dropped),
        "n_missing_successor_paths": int(missing_successor_paths),
        "n_groups": int(len(groups)),
    }


def build_tree_strategy_frontiers_for_dataset(
    rows: Sequence[FrontierRow],
    kind_of_data: str,
    dataset_id: str,
    onnx_frontier_size: int,
    random_frontier_ratio: float,
    random_frontier_with_failure_ratio: float,
    max_regular_distance_for_reward: float,
    failure_reward_value: float = -1.0,
    seed: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    goal_path = _resolve_dataset_goal_path(rows=rows, kind_of_data=kind_of_data)
    if kind_of_data == "separated" and not goal_path:
        return [], {
            "dataset_id": str(dataset_id),
            "reason": "missing_or_ambiguous_goal_path",
            "generated_before_dedup": 0,
            "generated_after_dedup": 0,
            "deduplicated_to_common": 0,
            "dropped_no_non_failure": 0,
        }

    state_by_path, children_by_parent, parent_by_child, all_nodes = _build_tree_structures(rows)
    if not state_by_path:
        return [], {
            "dataset_id": str(dataset_id),
            "reason": "no_states",
            "generated_before_dedup": 0,
            "generated_after_dedup": 0,
            "deduplicated_to_common": 0,
            "dropped_no_non_failure": 0,
        }

    reward_by_path: Dict[str, float] = {
        path: normalize_distance_to_reward(
            distance=float(state.distance),
            max_regular_distance=float(max_regular_distance_for_reward),
            failure_reward_value=float(failure_reward_value),
        )
        for path, state in state_by_path.items()
    }

    roots = _find_initial_states(
        state_by_path=state_by_path,
        children_by_parent=children_by_parent,
        parent_by_child=parent_by_child,
    )
    if not roots:
        roots = sorted(_ordered_unique_paths(state_by_path.keys()))
    if not roots:
        return [], {
            "dataset_id": str(dataset_id),
            "reason": "no_roots",
            "generated_before_dedup": 0,
            "generated_after_dedup": 0,
            "deduplicated_to_common": 0,
            "dropped_no_non_failure": 0,
        }

    frontier_target_size = max(1, int(onnx_frontier_size))
    queue: deque[str] = deque()
    available_state_set = set(str(path) for path in state_by_path.keys())
    discovered_set: set[str] = set()
    open_state_order: List[str] = []
    open_state_set: set[str] = set()
    expanded_state_order: List[str] = []
    expanded_state_set: set[str] = set()

    def _enqueue(path: str) -> None:
        path_s = str(path).strip()
        if not path_s or path_s in discovered_set:
            return
        discovered_set.add(path_s)
        if path_s in available_state_set:
            open_state_order.append(path_s)
            open_state_set.add(path_s)
        queue.append(path_s)

    for root in roots:
        _enqueue(str(root))

    generated_frontiers_gc: List[Dict[str, Any]] = []
    expanded_nodes = 0

    while True:
        while queue:
            current = str(queue.popleft())
            expanded_nodes += 1
            if current in open_state_set:
                open_state_set.discard(current)
            if current in available_state_set and current not in expanded_state_set:
                expanded_state_set.add(current)
                expanded_state_order.append(current)
            if expanded_state_order:
                target_size = int(frontier_target_size)

                greedy_paths = _select_greedy_paths(
                    expanded_node=current,
                    target_size=target_size,
                    expanded_state_order=expanded_state_order,
                    expanded_state_set=expanded_state_set,
                    open_state_order=open_state_order,
                    open_state_set=open_state_set,
                    available_state_set=available_state_set,
                    parent_by_child=parent_by_child,
                    children_by_parent=children_by_parent,
                )
                conservative_paths = _select_conservative_paths(
                    expanded_node=current,
                    target_size=target_size,
                    expanded_state_order=expanded_state_order,
                    open_state_order=open_state_order,
                    open_state_set=open_state_set,
                    available_state_set=available_state_set,
                    parent_by_child=parent_by_child,
                    children_by_parent=children_by_parent,
                )
                if frozenset(greedy_paths) == frozenset(conservative_paths):
                    common_frontier = _build_frontier_from_paths(
                        selected_paths=greedy_paths,
                        state_by_path=state_by_path,
                        reward_by_path=reward_by_path,
                        label=FRONTIER_LABEL_COMMON,
                        predecessor_path=current,
                        goal_path=str(goal_path or ""),
                        dataset_id=str(dataset_id),
                        failure_reward_value=float(failure_reward_value),
                    )
                    if common_frontier is not None:
                        generated_frontiers_gc.append(common_frontier)
                else:
                    greedy_frontier = _build_frontier_from_paths(
                        selected_paths=greedy_paths,
                        state_by_path=state_by_path,
                        reward_by_path=reward_by_path,
                        label=FRONTIER_LABEL_GREEDY,
                        predecessor_path=current,
                        goal_path=str(goal_path or ""),
                        dataset_id=str(dataset_id),
                        failure_reward_value=float(failure_reward_value),
                    )
                    if greedy_frontier is not None:
                        generated_frontiers_gc.append(greedy_frontier)

                    conservative_frontier = _build_frontier_from_paths(
                        selected_paths=conservative_paths,
                        state_by_path=state_by_path,
                        reward_by_path=reward_by_path,
                        label=FRONTIER_LABEL_CONSERVATIVE,
                        predecessor_path=current,
                        goal_path=str(goal_path or ""),
                        dataset_id=str(dataset_id),
                        failure_reward_value=float(failure_reward_value),
                    )
                    if conservative_frontier is not None:
                        generated_frontiers_gc.append(conservative_frontier)

            for child in children_by_parent.get(current, []):
                _enqueue(str(child))

        remaining_nodes = [
            path for path in all_nodes if str(path).strip() and str(path).strip() not in discovered_set
        ]
        if not remaining_nodes:
            break
        _enqueue(str(remaining_nodes[0]))

    gc_frontiers, gc_dedup_stats = _deduplicate_strategy_frontiers(
        frontiers=generated_frontiers_gc,
        kind_of_data=kind_of_data,
        failure_reward_value=float(failure_reward_value),
    )
    reference_count = int(len(gc_frontiers))
    random_frontiers, random_stats = _generate_random_frontiers_from_state_pool(
        state_by_path=state_by_path,
        reward_by_path=reward_by_path,
        dataset_id=str(dataset_id),
        goal_path=str(goal_path or ""),
        onnx_frontier_size=int(frontier_target_size),
        random_frontier_ratio=float(random_frontier_ratio),
        random_frontier_with_failure_ratio=float(random_frontier_with_failure_ratio),
        reference_frontier_count=int(reference_count),
        seed=int(seed) + 811,
        failure_reward_value=float(failure_reward_value),
    )
    final_frontiers, final_dedup_stats = _deduplicate_strategy_frontiers(
        frontiers=[*gc_frontiers, *random_frontiers],
        kind_of_data=kind_of_data,
        failure_reward_value=float(failure_reward_value),
    )

    label_counts = {label: 0 for label in FRONTIER_LABEL_ORDER}
    size_counts: Dict[int, int] = {}
    for frontier in final_frontiers:
        label = str(frontier.get("frontier_label", FRONTIER_LABEL_COMMON))
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] = int(label_counts[label] + 1)

        size = int(len(frontier.get("successor_paths", [])))
        size_counts[size] = int(size_counts.get(size, 0) + 1)

    summary = {
        "dataset_id": str(dataset_id),
        "num_rows": int(len(rows)),
        "num_states": int(len(state_by_path)),
        "num_initial_states": int(len(roots)),
        "expanded_nodes": int(expanded_nodes),
        "onnx_frontier_size": int(frontier_target_size),
        "random_frontier_ratio": float(random_frontier_ratio),
        "random_frontier_with_failure_ratio": float(random_frontier_with_failure_ratio),
        "generated_greedy_conservative_before_dedup": int(len(generated_frontiers_gc)),
        "generated_greedy_conservative_after_dedup": int(len(gc_frontiers)),
        "gc_deduplicated_to_common": int(gc_dedup_stats["deduplicated_to_common"]),
        "gc_dropped_no_non_failure": int(gc_dedup_stats["dropped_no_non_failure"]),
        "generated_random_frontiers": int(len(random_frontiers)),
        "random_generation_stats": random_stats,
        "generated_total_after_final_dedup": int(len(final_frontiers)),
        "final_deduplicated_to_common": int(final_dedup_stats["deduplicated_to_common"]),
        "final_dropped_no_non_failure": int(final_dedup_stats["dropped_no_non_failure"]),
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "size_counts": {str(k): int(v) for k, v in sorted(size_counts.items())},
    }
    return final_frontiers, summary


def _flatten_candidates(frontiers: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for frontier in frontiers:
        goal_path = str(frontier.get("goal_path", ""))
        dataset_id = str(frontier.get("dataset_id", ""))
        paths = [str(x) for x in frontier.get("successor_paths", [])]
        distances = [float(x) for x in frontier.get("distances", [])]
        depths = [int(x) for x in frontier.get("depths", [])]
        rewards = [float(x) for x in frontier.get("rewards", [])]
        n = min(len(paths), len(distances), len(depths), len(rewards))
        for i in range(n):
            candidates.append(
                {
                    "successor_path": paths[i],
                    "distance": distances[i],
                    "depth": depths[i],
                    "reward": rewards[i],
                    "goal_path": goal_path,
                    "dataset_id": dataset_id,
                }
            )
    return candidates


def _frontier_to_candidates(
    frontier: Dict[str, Any],
    kind_of_data: str,
) -> List[Dict[str, Any]]:
    goal_path = str(frontier.get("goal_path", "")) if kind_of_data == "separated" else ""
    dataset_id = str(frontier.get("dataset_id", ""))
    paths = [str(x) for x in frontier.get("successor_paths", [])]
    distances = [float(x) for x in frontier.get("distances", [])]
    depths = [int(x) for x in frontier.get("depths", [])]
    rewards = [float(x) for x in frontier.get("rewards", [])]
    n = min(len(paths), len(distances), len(depths), len(rewards))
    out: List[Dict[str, Any]] = []
    for i in range(n):
        out.append(
            {
                "successor_path": paths[i],
                "distance": distances[i],
                "depth": depths[i],
                "reward": rewards[i],
                "goal_path": goal_path,
                "dataset_id": dataset_id,
            }
        )
    return out


def _frontier_from_candidate_window(
    candidates: Sequence[Dict[str, Any]],
    predecessor_path: str,
    kind_of_data: str,
    failure_reward_value: float,
    stress_schedule: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if len(candidates) <= 1:
        return None

    rewards = [float(c["reward"]) for c in candidates]
    if not rewards:
        return None
    if all(_is_failure_reward(r, failure_reward_value) for r in rewards):
        return None

    goal_path = ""
    if kind_of_data == "separated":
        goal_path = str(candidates[0].get("goal_path", ""))
        if not goal_path:
            return None
        if any(str(c.get("goal_path", "")) != goal_path for c in candidates):
            return None

    out = {
        "predecessor_path": str(predecessor_path),
        "goal_path": goal_path,
        "successor_paths": [str(c["successor_path"]) for c in candidates],
        "distances": [float(c["distance"]) for c in candidates],
        "depths": [int(c["depth"]) for c in candidates],
        "rewards": rewards,
    }
    if stress_schedule:
        out["stress_schedule"] = str(stress_schedule)
    return out


def _deduplicate_candidates(
    candidates: Sequence[Dict[str, Any]],
    kind_of_data: str,
) -> List[Dict[str, Any]]:
    best_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for cand in candidates:
        dataset_id = str(cand.get("dataset_id", ""))
        goal = str(cand.get("goal_path", "")) if kind_of_data == "separated" else ""
        path = str(cand["successor_path"])
        key = (dataset_id, goal, path)
        current = best_by_key.get(key)
        if current is None or float(cand["reward"]) > float(current["reward"]):
            best_by_key[key] = dict(cand)
    return list(best_by_key.values())


def build_random_eval_frontiers_for_dataset(
    clean_frontiers: Sequence[Dict[str, Any]],
    kind_of_data: str,
    n_max_dataset_queries: int,
    seed: int,
    dataset_tag: str,
    failure_reward_value: float = -1.0,
) -> List[Dict[str, Any]]:
    target = max(0, int(n_max_dataset_queries))
    if target == 0:
        return []

    rng = random.Random(seed)
    candidates = _deduplicate_candidates(
        _flatten_candidates(clean_frontiers),
        kind_of_data=kind_of_data,
    )
    if not candidates:
        return []

    pools_by_goal: Dict[str, List[Dict[str, Any]]] = {}
    if kind_of_data == "separated":
        for cand in candidates:
            goal = str(cand.get("goal_path", ""))
            if not goal:
                continue
            pools_by_goal.setdefault(goal, []).append(cand)
        if not pools_by_goal:
            return []
    else:
        pools_by_goal[""] = candidates

    random_frontiers: List[Dict[str, Any]] = []
    seen_keys = set()
    attempts = 0
    max_attempts = max(1000, target * 300)
    while len(random_frontiers) < target and attempts < max_attempts:
        attempts += 1
        size = int(rng.choice(RANDOM_EVAL_FRONTIER_SIZES))

        if kind_of_data == "separated":
            eligible_goals = [g for g, pool in pools_by_goal.items() if len(pool) >= size]
            if not eligible_goals:
                break
            goal = str(rng.choice(eligible_goals))
            pool = pools_by_goal[goal]
        else:
            goal = ""
            pool = pools_by_goal[""]
            if len(pool) < size:
                continue

        chosen = rng.sample(pool, size)
        rewards = [float(c["reward"]) for c in chosen]
        best_reward = max(rewards)
        n_best = sum(1 for r in rewards if abs(r - best_reward) <= FAILURE_EPS)
        if n_best != 1:
            continue
        if all(_is_failure_reward(r, failure_reward_value) for r in rewards):
            continue

        frontier_key = (
            goal,
            tuple(sorted(str(c["successor_path"]) for c in chosen)),
        )
        if frontier_key in seen_keys:
            continue
        seen_keys.add(frontier_key)

        random_frontiers.append(
            {
                "predecessor_path": f"random_eval_{dataset_tag}_{len(random_frontiers)}",
                "goal_path": goal,
                "successor_paths": [str(c["successor_path"]) for c in chosen],
                "distances": [float(c["distance"]) for c in chosen],
                "depths": [int(c["depth"]) for c in chosen],
                "rewards": rewards,
            }
        )

    return random_frontiers


def build_stress_eval_frontiers_for_dataset(
    clean_frontiers: Sequence[Dict[str, Any]],
    kind_of_data: str,
    n_max_dataset_queries: int,
    max_size_frontier: int,
    dataset_tag: str,
    failure_reward_value: float = -1.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    target = max(0, int(n_max_dataset_queries))
    if target == 0:
        return [], []

    max_size = max(2, int(max_size_frontier))
    frontiers_by_predecessor: Dict[str, Dict[str, Any]] = {}
    for frontier in clean_frontiers:
        predecessor_path = str(frontier.get("predecessor_path", "")).strip()
        if not predecessor_path:
            continue
        if predecessor_path not in frontiers_by_predecessor:
            frontiers_by_predecessor[predecessor_path] = frontier
    if not frontiers_by_predecessor:
        return [], []

    predecessor_paths = set(frontiers_by_predecessor.keys())
    successor_paths = {
        str(path)
        for frontier in clean_frontiers
        for path in frontier.get("successor_paths", [])
    }
    root_predecessors = sorted(predecessor_paths - successor_paths)
    if not root_predecessors:
        root_predecessors = sorted(predecessor_paths)

    fifo_frontiers: List[Dict[str, Any]] = []
    lifo_frontiers: List[Dict[str, Any]] = []
    query_idx = 0
    for root in root_predecessors:
        if query_idx >= target:
            break

        visited = set()
        current = str(root)
        cumulative_candidates: List[Dict[str, Any]] = []

        while current in frontiers_by_predecessor and current not in visited and query_idx < target:
            visited.add(current)
            frontier = frontiers_by_predecessor[current]
            layer_candidates = _frontier_to_candidates(frontier, kind_of_data=kind_of_data)
            if not layer_candidates:
                break

            cumulative_candidates.extend(layer_candidates)
            fifo_window = cumulative_candidates[:max_size]
            lifo_window = cumulative_candidates[-max_size:]

            fifo_frontier = _frontier_from_candidate_window(
                candidates=fifo_window,
                predecessor_path=f"stress_fifo_{dataset_tag}_{query_idx}",
                kind_of_data=kind_of_data,
                failure_reward_value=float(failure_reward_value),
                stress_schedule=STRESS_SCHEDULE_FIFO,
            )
            if fifo_frontier is not None:
                fifo_frontiers.append(fifo_frontier)

            lifo_frontier = _frontier_from_candidate_window(
                candidates=lifo_window,
                predecessor_path=f"stress_lifo_{dataset_tag}_{query_idx}",
                kind_of_data=kind_of_data,
                failure_reward_value=float(failure_reward_value),
                stress_schedule=STRESS_SCHEDULE_LIFO,
            )
            if lifo_frontier is not None:
                lifo_frontiers.append(lifo_frontier)

            query_idx += 1

            next_predecessor = None
            for successor_path in frontier.get("successor_paths", []):
                successor_key = str(successor_path)
                if successor_key in frontiers_by_predecessor and successor_key not in visited:
                    next_predecessor = successor_key
                    break
            if next_predecessor is None:
                break
            current = next_predecessor

    return fifo_frontiers, lifo_frontiers


def _build_graph_loader(
    dataset_type: str,
    enable_graph_cache: bool = True,
) -> Tuple[Callable[[str], Any], Dict[str, int]]:
    graph_cache: Dict[str, Any] = {}
    stats = {"requests": 0, "hits": 0, "misses": 0}

    def _load(path: str) -> Any:
        key = str(path)
        stats["requests"] += 1
        if enable_graph_cache and key in graph_cache:
            stats["hits"] += 1
            return graph_cache[key]
        graph = load_pyg_graph(key, dataset_type=dataset_type)
        stats["misses"] += 1
        if enable_graph_cache:
            graph_cache[key] = graph
        return graph

    return _load, stats


def build_frontier_samples(
    folder_data: str,
    list_subset_train: Sequence[str],
    kind_of_data: str,
    dataset_type: str,
    seed: int,
    max_regular_distance_for_reward: float = 50.0,
    failure_reward_value: float = -1.0,
    n_max_dataset_queries: int = 1000,
    max_size_frontier: int = 25,
    onnx_frontier_size: int = 32,
    train_random_frontier_ratio: float = 0.2,
    train_random_with_failure_ratio: float = 0.4,
    train_frontier_jaccard_threshold: float = 0.75,
    build_eval_data: bool = True,
    enable_graph_cache: bool = True,
    **_: Any,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    root = Path(folder_data)
    dataset_type = str(dataset_type).upper()
    if dataset_type not in VALID_DATASET_TYPES:
        raise ValueError(
            f"Unsupported dataset_type '{dataset_type}'. "
            f"Expected one of {sorted(VALID_DATASET_TYPES)}."
        )
    if kind_of_data not in {"merged", "separated"}:
        raise ValueError(f"Unsupported kind_of_data: {kind_of_data}")

    all_train_frontiers: List[Dict[str, Any]] = []
    all_eval_random_frontiers: List[Dict[str, Any]] = []
    all_eval_stress_fifo_frontiers: List[Dict[str, Any]] = []
    all_eval_stress_lifo_frontiers: List[Dict[str, Any]] = []
    dataset_summaries: List[Dict[str, Any]] = []

    dataset_entries: List[Tuple[Path, Path]] = []
    for prob_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if list_subset_train and os.path.basename(prob_dir) not in list_subset_train:
            continue
        csv_files = sorted(p for p in prob_dir.iterdir() if p.suffix.lower() == ".csv")
        for csv_path in csv_files:
            dataset_entries.append((prob_dir, csv_path))

    for dataset_counter, (prob_dir, csv_path) in enumerate(
        tqdm(
            dataset_entries,
            desc="Building strategy frontiers",
        )
    ):
        rows = read_frontier_csv(csv_path, kind_of_data=kind_of_data)
        dataset_id = f"{prob_dir.name}/{csv_path.name}"
        train_frontiers, strategy_summary = build_tree_strategy_frontiers_for_dataset(
            rows=rows,
            kind_of_data=kind_of_data,
            dataset_id=dataset_id,
            onnx_frontier_size=int(onnx_frontier_size),
            random_frontier_ratio=float(train_random_frontier_ratio),
            random_frontier_with_failure_ratio=float(train_random_with_failure_ratio),
            max_regular_distance_for_reward=float(max_regular_distance_for_reward),
            failure_reward_value=float(failure_reward_value),
            seed=int(seed) + 7919 * dataset_counter + 31,
        )

        train_frontiers = list(train_frontiers)

        if build_eval_data:
            cleaned = group_clean_frontiers(
                rows=rows,
                kind_of_data=kind_of_data,
                max_regular_distance_for_reward=float(max_regular_distance_for_reward),
                failure_reward_value=float(failure_reward_value),
            )
            eval_random_seed = int(seed + 104729 * dataset_counter + 17)
            eval_random_frontiers = build_random_eval_frontiers_for_dataset(
                clean_frontiers=cleaned,
                kind_of_data=kind_of_data,
                n_max_dataset_queries=int(n_max_dataset_queries),
                seed=eval_random_seed,
                dataset_tag=f"{prob_dir.name}_{csv_path.stem}",
                failure_reward_value=float(failure_reward_value),
            )
            eval_stress_fifo_frontiers, eval_stress_lifo_frontiers = (
                build_stress_eval_frontiers_for_dataset(
                    clean_frontiers=cleaned,
                    kind_of_data=kind_of_data,
                    n_max_dataset_queries=int(n_max_dataset_queries),
                    max_size_frontier=int(max_size_frontier),
                    dataset_tag=f"{prob_dir.name}_{csv_path.stem}",
                    failure_reward_value=float(failure_reward_value),
                )
            )
        else:
            cleaned = []
            eval_random_frontiers = []
            eval_stress_fifo_frontiers = []
            eval_stress_lifo_frontiers = []

        for frontier in train_frontiers:
            frontier["dataset_id"] = dataset_id
        for frontier in eval_random_frontiers:
            frontier["dataset_id"] = dataset_id
        for frontier in eval_stress_fifo_frontiers:
            frontier["dataset_id"] = dataset_id
            frontier["stress_schedule"] = STRESS_SCHEDULE_FIFO
        for frontier in eval_stress_lifo_frontiers:
            frontier["dataset_id"] = dataset_id
            frontier["stress_schedule"] = STRESS_SCHEDULE_LIFO

        all_train_frontiers.extend(train_frontiers)
        all_eval_random_frontiers.extend(eval_random_frontiers)
        all_eval_stress_fifo_frontiers.extend(eval_stress_fifo_frontiers)
        all_eval_stress_lifo_frontiers.extend(eval_stress_lifo_frontiers)

        dataset_summaries.append(
            {
                "dataset_id": dataset_id,
                "num_rows": int(len(rows)),
                "strategy_summary": strategy_summary,
                "frontiers_cleaned": len(cleaned),
                "train_frontiers": len(train_frontiers),
                "eval_random_frontiers": len(eval_random_frontiers),
                "eval_stress_fifo_frontiers": len(eval_stress_fifo_frontiers),
                "eval_stress_lifo_frontiers": len(eval_stress_lifo_frontiers),
            }
        )

    all_train_frontiers, train_jaccard_prune_stats = _prune_frontiers_by_jaccard_similarity(
        frontiers=all_train_frontiers,
        jaccard_similarity_threshold=float(train_frontier_jaccard_threshold),
        failure_reward_value=float(failure_reward_value),
    )

    graph_loader, graph_load_stats = _build_graph_loader(
        dataset_type=dataset_type,
        enable_graph_cache=bool(enable_graph_cache),
    )

    train_samples = [
        _materialize_frontier(
            frontier=f,
            kind_of_data=kind_of_data,
            dataset_type=dataset_type,
            max_regular_distance_for_reward=float(max_regular_distance_for_reward),
            failure_reward_value=float(failure_reward_value),
            graph_loader=graph_loader,
        )
        for f in tqdm(all_train_frontiers, desc="Materializing train frontiers")
    ]
    eval_samples = [
        _materialize_frontier(
            frontier=f,
            kind_of_data=kind_of_data,
            dataset_type=dataset_type,
            max_regular_distance_for_reward=float(max_regular_distance_for_reward),
            failure_reward_value=float(failure_reward_value),
            graph_loader=graph_loader,
        )
        for f in tqdm(all_eval_random_frontiers, desc="Materializing eval-random frontiers")
    ]
    eval2_samples = [
        _materialize_frontier(
            frontier=f,
            kind_of_data=kind_of_data,
            dataset_type=dataset_type,
            max_regular_distance_for_reward=float(max_regular_distance_for_reward),
            failure_reward_value=float(failure_reward_value),
            graph_loader=graph_loader,
        )
        for f in tqdm(all_eval_stress_fifo_frontiers, desc="Materializing eval-stress-fifo frontiers")
    ]
    eval3_samples = [
        _materialize_frontier(
            frontier=f,
            kind_of_data=kind_of_data,
            dataset_type=dataset_type,
            max_regular_distance_for_reward=float(max_regular_distance_for_reward),
            failure_reward_value=float(failure_reward_value),
            graph_loader=graph_loader,
        )
        for f in tqdm(all_eval_stress_lifo_frontiers, desc="Materializing eval-stress-lifo frontiers")
    ]

    train_label_counts = {label: 0 for label in FRONTIER_LABEL_ORDER}
    for sample in train_samples:
        label = str(sample.get("frontier_label", FRONTIER_LABEL_COMMON))
        if label not in train_label_counts:
            train_label_counts[label] = 0
        train_label_counts[label] = int(train_label_counts[label] + 1)

    params = {
        "folder_data": str(root),
        "kind_of_data": str(kind_of_data),
        "dataset_type": str(dataset_type),
        "seed": int(seed),
        "max_regular_distance_for_reward": float(max_regular_distance_for_reward),
        "failure_reward_value": float(failure_reward_value),
        "n_max_dataset_queries": int(n_max_dataset_queries),
        "max_size_frontier": int(max_size_frontier),
        "onnx_frontier_size": int(onnx_frontier_size),
        "train_random_frontier_ratio": float(train_random_frontier_ratio),
        "train_random_with_failure_ratio": float(train_random_with_failure_ratio),
        "train_frontier_jaccard_threshold": float(train_frontier_jaccard_threshold),
        "train_frontier_jaccard_pruning_pre_materialization": train_jaccard_prune_stats,
        "build_eval_data": bool(build_eval_data),
        "num_frontiers_train": len(train_samples),
        "num_frontiers_eval_random": len(eval_samples),
        "num_frontiers_eval_stress_fifo": len(eval2_samples),
        "num_frontiers_eval_stress_lifo": len(eval3_samples),
        "num_frontiers_eval": len(eval_samples),
        "num_frontiers_eval2": len(eval2_samples),
        "num_frontiers_eval3": len(eval3_samples),
        "dataset_summaries": dataset_summaries,
        "split_summary": {
            "num_datasets": len(dataset_summaries),
            "num_frontiers_train": len(train_samples),
            "num_frontiers_eval_random": len(eval_samples),
            "num_frontiers_eval_stress_fifo": len(eval2_samples),
            "num_frontiers_eval_stress_lifo": len(eval3_samples),
            "num_frontiers_eval": len(eval_samples),
            "num_frontiers_eval2": len(eval2_samples),
            "num_frontiers_eval3": len(eval3_samples),
            "onnx_frontier_size": int(onnx_frontier_size),
            "train_random_frontier_ratio": float(train_random_frontier_ratio),
            "train_random_with_failure_ratio": float(train_random_with_failure_ratio),
            "train_frontier_jaccard_threshold": float(train_frontier_jaccard_threshold),
            "train_frontier_jaccard_pruning_pre_materialization": train_jaccard_prune_stats,
            "train_frontier_label_counts": {
                str(label): int(count) for label, count in sorted(train_label_counts.items())
            },
            "dataset_summaries": dataset_summaries,
        },
        "graph_cache_requests": int(graph_load_stats["requests"]),
        "graph_cache_hits": int(graph_load_stats["hits"]),
        "graph_cache_misses": int(graph_load_stats["misses"]),
        "graph_cache_hit_rate": (
            float(graph_load_stats["hits"] / graph_load_stats["requests"])
            if int(graph_load_stats["requests"]) > 0
            else 0.0
        ),
    }
    return train_samples, eval_samples, eval2_samples, eval3_samples, params


def _materialize_frontier(
    frontier: Dict[str, Any],
    kind_of_data: str,
    dataset_type: str,
    max_regular_distance_for_reward: float,
    failure_reward_value: float = -1.0,
    graph_loader: Optional[Callable[[str], Any]] = None,
    m_failed_state: Optional[float] = None,
    **_: Any,
) -> Dict[str, Any]:
    _ = m_failed_state
    if kind_of_data not in {"merged", "separated"}:
        raise ValueError(f"Unsupported kind_of_data: {kind_of_data}")

    load_graph = (
        graph_loader
        if graph_loader is not None
        else lambda p: load_pyg_graph(p, dataset_type=dataset_type)
    )
    successor_graphs = [load_graph(p) for p in frontier["successor_paths"]]
    goal_graph = None
    if kind_of_data == "separated":
        goal_path = str(frontier.get("goal_path", ""))
        if not goal_path:
            raise ValueError(
                "Missing Goal path for kind_of_data='separated'."
            )
        goal_graph = load_graph(goal_path)

    combined = combine_graphs(
        frontier_graphs=successor_graphs,
        goal_graph=goal_graph,
        kind_of_data=kind_of_data,
        action_ids=None,
    )

    distances = torch.tensor(frontier["distances"], dtype=torch.float32)
    if "rewards" in frontier and frontier["rewards"] is not None:
        reward_values = [float(x) for x in frontier["rewards"]]
    else:
        reward_values = [
            normalize_distance_to_reward(
                distance=float(d),
                max_regular_distance=float(max_regular_distance_for_reward),
                failure_reward_value=float(failure_reward_value),
            )
            for d in frontier["distances"]
        ]
    reward_targets = torch.tensor(reward_values, dtype=torch.float32)
    is_failure = torch.tensor(
        [
            _is_failure_reward(float(r), float(failure_reward_value))
            for r in reward_values
        ],
        dtype=torch.bool,
    )
    best_idx = int(torch.argmax(reward_targets).item())

    sample = {
        "node_features": combined.node_features,
        "edge_index": combined.edge_index,
        "edge_attr": combined.edge_attr,
        "membership": combined.membership,
        "action_map": combined.action_map,
        "successor_ids": [str(p) for p in frontier["successor_paths"]],
        "reward_target": reward_targets,
        "rewards": reward_targets,
        "distance_raw": distances,
        "distances": distances,
        "is_failure": is_failure,
        "oracle_index": torch.tensor(best_idx, dtype=torch.long),
        "oracle_reward": reward_targets[best_idx],
        "goal_path": str(frontier.get("goal_path", "")),
        "predecessor_path": str(frontier.get("predecessor_path", "")),
        "dataset_id": str(frontier.get("dataset_id", "")),
        "frontier_label": str(frontier.get("frontier_label", FRONTIER_LABEL_COMMON)),
        "frontier_has_failure": bool(is_failure.any().item()),
        "_reward_max_regular_distance": float(max_regular_distance_for_reward),
        "_reward_failure_value": float(failure_reward_value),
    }
    if "stress_schedule" in frontier:
        sample["stress_schedule"] = str(frontier.get("stress_schedule", ""))
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
        self.samples = samples if isinstance(samples, list) else list(samples)

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
    successor_id_parts: List[List[str]] = []
    reward_target_parts: List[torch.Tensor] = []
    distance_raw_parts: List[torch.Tensor] = []
    failure_parts: List[torch.Tensor] = []
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
        reward_target = item.get("reward_target", item["rewards"]).to(torch.float32)
        distance_raw = item.get("distance_raw", item.get("distances")).to(torch.float32)
        if reward_target.numel() != distance_raw.numel():
            raise ValueError(
                "reward_target and distance tensors must have identical lengths per sample."
            )
        is_failure = item.get(
            "is_failure", torch.zeros((reward_target.numel(),), dtype=torch.bool)
        ).to(torch.bool)
        if is_failure.numel() != reward_target.numel():
            raise ValueError(
                "is_failure and reward_target tensors must have identical lengths per sample."
            )
        n_candidates = int(reward_target.size(0))

        node_parts.append(item["node_features"])
        membership = item["membership"].clone()
        candidate_mask = membership >= 0
        membership[candidate_mask] += membership_offset
        membership_parts.append(membership)

        if item["edge_index"].numel() > 0:
            edge_index_parts.append(item["edge_index"] + node_offset)
            edge_attr_parts.append(item["edge_attr"].to(torch.int64))

        action_parts.append(item["action_map"])
        successor_ids = item.get("successor_ids")
        if successor_ids is None:
            successor_id_parts.append([])
        else:
            if len(successor_ids) != n_candidates:
                raise ValueError(
                    "successor_ids must have the same length as candidates per sample."
                )
            successor_id_parts.append([str(x) for x in successor_ids])
        reward_target_parts.append(reward_target)
        distance_raw_parts.append(distance_raw)
        failure_parts.append(is_failure)
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
                goal_edge_attr_parts.append(item["goal_edge_attr"].to(torch.int64))
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
            else torch.zeros((0, 1), dtype=torch.int64)
        ),
        "membership": torch.cat(membership_parts, dim=0),
        "action_map": torch.cat(action_parts, dim=0),
        "successor_ids": successor_id_parts,
        "reward_target": torch.cat(reward_target_parts, dim=0),
        "rewards": torch.cat(reward_target_parts, dim=0),
        "distance_raw": torch.cat(distance_raw_parts, dim=0),
        "distances": torch.cat(distance_raw_parts, dim=0),
        "is_failure": torch.cat(failure_parts, dim=0),
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
            else torch.zeros((0, 1), dtype=torch.int64)
        )
        out["goal_batch"] = torch.cat(goal_batch_parts, dim=0)

    if pad_frontiers:
        max_n = max(x.size(0) for x in reward_target_parts)
        bsz = len(batch)
        mask = torch.zeros((bsz, max_n), dtype=torch.bool)
        padded_reward_targets = torch.zeros((bsz, max_n), dtype=torch.float32)
        padded_distances = torch.zeros((bsz, max_n), dtype=torch.float32)
        padded_is_failure = torch.zeros((bsz, max_n), dtype=torch.bool)
        padded_actions = torch.full((bsz, max_n), -1, dtype=torch.long)
        for i, (r, d, f, a) in enumerate(
            zip(reward_target_parts, distance_raw_parts, failure_parts, action_parts)
        ):
            n = int(r.size(0))
            mask[i, :n] = True
            padded_reward_targets[i, :n] = r
            padded_distances[i, :n] = d
            padded_is_failure[i, :n] = f
            padded_actions[i, :n] = a
        out["frontier_mask"] = mask
        out["padded_reward_targets"] = padded_reward_targets
        out["padded_rewards"] = padded_reward_targets
        out["padded_distances"] = padded_distances
        out["padded_is_failure"] = padded_is_failure
        out["padded_actions"] = padded_actions

    return out


def build_frontier_dataloader(
    samples: Sequence[Dict[str, Any]],
    batch_size: int = 8,
    seed: int = 42,
    num_workers: int = 0,
    pad_frontiers: bool = True,
    shuffle: bool = False,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    n_workers = int(max(0, num_workers))
    pin = bool(torch.cuda.is_available()) if pin_memory is None else bool(pin_memory)
    persistent = bool(n_workers > 0) if persistent_workers is None else bool(persistent_workers)
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": bool(shuffle),
        "num_workers": n_workers,
        "collate_fn": lambda x: frontier_collate_fn(x, pad_frontiers=pad_frontiers),
        "generator": generator,
        "pin_memory": pin,
        "persistent_workers": bool(persistent and n_workers > 0),
    }
    if n_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(max(1, prefetch_factor))
    return DataLoader(
        FrontierDataset(samples),
        **loader_kwargs,
    )


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

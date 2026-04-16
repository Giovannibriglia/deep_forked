from __future__ import annotations

import csv
import os
import random
import torch
import warnings
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


@dataclass
class FrontierRow:
    successor_path: str
    depth: int
    distance: float
    predecessor_path: str
    goal_path: str


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
    return 0.9 * dist / max_dist


def refresh_frontier_sample_targets(
    samples: Sequence[Dict[str, Any]],
    m_failed_state: float,
    max_regular_distance_for_reward: float,
    failure_reward_value: float = -1.0,
) -> None:
    # m_failed_state is kept only for API compatibility.
    _ = m_failed_state
    for sample in samples:
        distances = sample.get("distance_raw", sample.get("distances"))
        if distances is None:
            continue

        if isinstance(distances, torch.Tensor):
            distances_t = distances.to(torch.float32).view(-1)
        else:
            distances_t = torch.tensor(distances, dtype=torch.float32).view(-1)
        if distances_t.numel() == 0:
            continue

        reward_targets = torch.tensor(
            [
                normalize_distance_to_reward(
                    distance=float(d),
                    max_regular_distance=float(max_regular_distance_for_reward),
                    failure_reward_value=float(failure_reward_value),
                )
                for d in distances_t.tolist()
            ],
            dtype=torch.float32,
        )
        is_failure = torch.tensor(
            [
                _is_failure_reward(float(r), float(failure_reward_value))
                for r in reward_targets.tolist()
            ],
            dtype=torch.bool,
        )
        best_idx = int(torch.argmax(reward_targets).item())

        sample["distance_raw"] = distances_t.clone()
        sample["distances"] = distances_t.clone()
        sample["reward_target"] = reward_targets
        sample["rewards"] = reward_targets.clone()
        sample["is_failure"] = is_failure
        sample["oracle_index"] = torch.tensor(best_idx, dtype=torch.long)
        sample["oracle_reward"] = reward_targets[best_idx].clone()
        sample["frontier_has_failure"] = bool(is_failure.any().item())


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


def _flatten_candidates(frontiers: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for frontier in frontiers:
        goal_path = str(frontier.get("goal_path", ""))
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
                }
            )
    return candidates


def _frontier_to_candidates(
    frontier: Dict[str, Any],
    kind_of_data: str,
) -> List[Dict[str, Any]]:
    goal_path = str(frontier.get("goal_path", "")) if kind_of_data == "separated" else ""
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
    best_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for cand in candidates:
        goal = str(cand.get("goal_path", "")) if kind_of_data == "separated" else ""
        path = str(cand["successor_path"])
        key = (goal, path)
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

    dataset_counter = 0
    for prob_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if list_subset_train and os.path.basename(prob_dir) not in list_subset_train:
            continue
        csv_files = sorted(p for p in prob_dir.iterdir() if p.suffix.lower() == ".csv")
        for csv_path in csv_files:
            rows = read_frontier_csv(csv_path, kind_of_data=kind_of_data)
            cleaned = group_clean_frontiers(
                rows=rows,
                kind_of_data=kind_of_data,
                max_regular_distance_for_reward=float(max_regular_distance_for_reward),
                failure_reward_value=float(failure_reward_value),
            )
            dataset_id = f"{prob_dir.name}/{csv_path.name}"

            if not cleaned:
                dataset_summaries.append(
                    {
                        "dataset_id": dataset_id,
                        "frontiers_cleaned": 0,
                        "train_frontiers": 0,
                        "eval_random_frontiers": 0,
                        "eval_stress_fifo_frontiers": 0,
                        "eval_stress_lifo_frontiers": 0,
                    }
                )
                dataset_counter += 1
                continue

            train_frontiers = list(cleaned)

            if build_eval_data:
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
                    "frontiers_cleaned": len(cleaned),
                    "train_frontiers": len(train_frontiers),
                    "eval_random_frontiers": len(eval_random_frontiers),
                    "eval_stress_fifo_frontiers": len(eval_stress_fifo_frontiers),
                    "eval_stress_lifo_frontiers": len(eval_stress_lifo_frontiers),
                }
            )
            dataset_counter += 1

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

    params = {
        "folder_data": str(root),
        "kind_of_data": str(kind_of_data),
        "dataset_type": str(dataset_type),
        "seed": int(seed),
        "max_regular_distance_for_reward": float(max_regular_distance_for_reward),
        "failure_reward_value": float(failure_reward_value),
        "n_max_dataset_queries": int(n_max_dataset_queries),
        "max_size_frontier": int(max_size_frontier),
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
        "rewards": reward_targets.clone(),
        "distance_raw": distances,
        "distances": distances.clone(),
        "is_failure": is_failure,
        "oracle_index": torch.tensor(best_idx, dtype=torch.long),
        "oracle_reward": reward_targets[best_idx].clone(),
        "goal_path": str(frontier.get("goal_path", "")),
        "predecessor_path": str(frontier.get("predecessor_path", "")),
        "dataset_id": str(frontier.get("dataset_id", "")),
        "frontier_has_failure": bool(is_failure.any().item()),
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
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        FrontierDataset(samples),
        batch_size=batch_size,
        shuffle=bool(shuffle),
        num_workers=num_workers,
        collate_fn=lambda x: frontier_collate_fn(x, pad_frontiers=pad_frontiers),
        generator=generator,
    )


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

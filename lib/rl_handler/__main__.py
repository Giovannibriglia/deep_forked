from __future__ import annotations

import argparse
import json
import math
import matplotlib.pyplot as plt
import networkx as nx
import pydot
import re
import torch
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.data import (
    FRONTIER_LABEL_COMMON,
    FRONTIER_LABEL_CONSERVATIVE,
    FRONTIER_LABEL_GREEDY,
    FRONTIER_LABEL_RANDOM,
    build_frontier_dataloader,
    build_frontier_samples,
    build_tree_strategy_frontiers_for_dataset,
    normalize_distance_to_reward,
    read_frontier_csv,
    refresh_frontier_sample_targets,
    seed_everything,
)
from src.graph_utils import VALID_DATASET_TYPES, combine_graphs
from src.models.frontier_policy import FrontierPolicyNetwork
from src.trainer import (
    FAILURE_EPS,
    FAILURE_REWARD_VALUE,
    REGIME_ALL,
    RLFrontierTrainer,
)
try:
    from utils import (
        I64_MAX,
        I64_MIN,
        node_feature_dtypes_for_dataset as _node_feature_dtypes_for_dataset,
        parse_numeric_node_label_eval as _parse_numeric_node_label_eval,
        parse_onnx_frontier_sizes as _parse_onnx_frontier_sizes,
        ratio_0_1,
        str2bool,
        strip_quotes as _strip_quotes,
        validate_int64_range_eval as _validate_int64_range_eval,
    )
except ImportError:
    from .utils import (
        I64_MAX,
        I64_MIN,
        node_feature_dtypes_for_dataset as _node_feature_dtypes_for_dataset,
        parse_numeric_node_label_eval as _parse_numeric_node_label_eval,
        parse_onnx_frontier_sizes as _parse_onnx_frontier_sizes,
        ratio_0_1,
        str2bool,
        strip_quotes as _strip_quotes,
        validate_int64_range_eval as _validate_int64_range_eval,
    )

_BARE_NEGATIVE_INT_RE = re.compile(r'(?<!["\w])-([0-9]+)(?!["\w])')


@dataclass
class _EvalGraphTensors:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    node_names: torch.Tensor


@dataclass
class OnnxEvalContext:
    ort_sess: Any
    np_mod: Any
    static_onnx_len: Optional[int]
    use_goal_inputs: bool


@dataclass
class EvalSplitTracker:
    split_name: str
    eval_dir: Path
    history: Dict[str, list[float]]
    regime_history: Dict[str, Dict[str, list[float]]]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline RL frontier selector training and ONNX evaluation pipeline."
    )
    parser.add_argument("--subset-train", action="extend", nargs="+", type=str, default=[])
    parser.add_argument("--folder-raw-data", type=str, default="out/NN/Training")
    parser.add_argument(
        "--folder-test-data",
        type=str,
        default="",
        help="Optional test-data root. If empty, uses <model_root>/test_data.",
    )
    parser.add_argument("--dir-save-data", type=str, default="data")
    parser.add_argument("--dir-save-model", type=str, default="models")
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--model-name", type=str, default="frontier_policy")

    parser.add_argument("--dataset_type", choices=["MAPPED", "HASHED", "BITMASK"], default="HASHED")
    parser.add_argument(
        "--kind-of-data",
        choices=["merged", "separated"],
        default="merged",
        help=(
            "Graph composition mode. "
            "'merged': goal is encoded in successor states. "
            "'separated': goal graph is loaded from CSV Goal path."
        ),
    )
    parser.add_argument(
        "--n-max-dataset-queries",
        type=int,
        default=500,
        help="Maximum number of generated evaluation strategy frontiers per dataset.",
    )
    parser.add_argument(
        "--max-size-frontier",
        type=int,
        default=32,
        help="Maximum frontier size used by legacy auxiliary frontier generators.",
    )
    parser.add_argument(
        "--max-failure-states-per-dataset",
        type=ratio_0_1,
        default=0.3,
        help=(
            "Keep all train frontiers with no failure states, then keep at most "
            "this fraction of train frontiers that have at least one failure state."
        ),
    )
    parser.add_argument(
        "--train-frontier-jaccard-threshold",
        type=ratio_0_1,
        default=0.6,
        help=(
            "Prune near-duplicate train frontiers using Jaccard similarity over "
            "successor_ids. Keep the first frontier and drop later ones with "
            "similarity >= threshold within the same dataset and frontier size."
        ),
    )
    parser.add_argument(
        "--eval-frontier-jaccard-threshold",
        type=ratio_0_1,
        default=0.3,
        help=(
            "Prune near-duplicate evaluation frontiers using Jaccard similarity over "
            "successor paths, applied independently per source dataset and frontier size."
        ),
    )

    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Batch size to use for evaluation dataloaders. If not set, uses --batch-size")
    parser.add_argument("--eval-num-workers", type=int, default=0, help="Number of workers for evaluation dataloaders.")
    parser.add_argument("--n-train-epochs", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help="If > 0, clip gradient norm to this value before optimizer step.",
    )
    parser.add_argument(
        "--early-stopping-patience-evals",
        type=int,
        default=0,
        help=(
            "If > 0, stop training when eval reward does not improve for this many "
            "evaluation checkpoints."
        ),
    )

    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--conv-type", choices=["gine", "rgcn", "gcn"], default="gine")
    parser.add_argument("--pooling-type", choices=["mean", "sum", "max"], default="mean")
    parser.add_argument("--edge-emb-dim", type=int, default=32)
    parser.add_argument(
        "--K",
        "--edge-label-buckets",
        dest="edge_label_buckets",
        type=int,
        default=256,
        help=(
            "Fixed number of edge-label buckets (K) used by the model "
            "edge embedding table."
        ),
    )
    parser.add_argument("--num-node-labels", type=int, default=4096)
    parser.add_argument("--use-global-context", type=str2bool, default=True)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--reward-formulation", type=str, default="negative_distance")
    parser.add_argument("--max-regular-distance-for-reward", type=float, default=50.0)
    parser.add_argument("--failure-reward-value", type=float, default=-1.0)

    parser.add_argument("--build-data", type=str2bool, default=True)
    parser.add_argument(
        "--build-eval-data",
        type=str2bool,
        default=False,
        help=(
            "If true, rebuild strategy query definitions for ONNX evaluation from train/test roots; "
            "otherwise load the saved query bundle."
        ),
    )
    parser.add_argument("--train", type=str2bool, default=True)
    parser.add_argument("--evaluate", type=str2bool, default=False)
    parser.add_argument("--export-onnx", type=str2bool, default=True)
    parser.add_argument(
        "--onnx-frontier-size",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help=(
            "Frontier size used for ONNX export tracing. "
            "Provide one or more integers (e.g. 32 or 16 32 64)."
        ),
    )
    parser.add_argument(
        "--train-random-frontier-ratio",
        type=ratio_0_1,
        default=0.2,
        help=(
            "Number of random frontiers is this ratio times the count of "
            "post-processed greedy/conservative/common frontiers."
        ),
    )
    parser.add_argument(
        "--train-random-frontier-with-failure-ratio",
        type=ratio_0_1,
        default=0.4,
        help=(
            "Target fraction of random frontiers that must include at least "
            "one failure state (each frontier still includes a non-failure state)."
        ),
    )

    return parser.parse_args()


def _paths(args):
    data_root = Path(args.dir_save_data)
    model_root = Path(args.dir_save_model)
    if args.experiment_name:
        data_root = data_root / args.experiment_name
        model_root = model_root / args.experiment_name
    data_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    return data_root, model_root


def _build_or_load_train_samples(
    args,
    data_root: Path,
    onnx_frontier_size: int,
    cache_namespace: str = "",
):
    data_dir = data_root / "processed_data"
    namespace = str(cache_namespace).strip()
    if namespace:
        data_dir = data_dir / namespace
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train_samples.pt"
    params_path = data_dir / "samples_params.pt"
    legacy_train_candidates = [
        data_root / "data" / "train_samples.pt",
        data_root / "data_pytorch" / "train_samples.pt",
    ]
    legacy_params_candidates = [
        data_root / "data" / "samples_params.pt",
        data_root / "data_pytorch" / "samples_params.pt",
    ]

    if (not bool(args.build_data)) and (not train_path.exists()) and (not namespace):
        for legacy_train_path in legacy_train_candidates:
            if legacy_train_path.exists():
                train_path.write_bytes(legacy_train_path.read_bytes())
                break
        if not params_path.exists():
            for legacy_params_path in legacy_params_candidates:
                if legacy_params_path.exists():
                    params_path.write_bytes(legacy_params_path.read_bytes())
                    break

    if bool(args.build_data) or not train_path.exists():
        train_samples, _, _, _, params = build_frontier_samples(
            folder_data=args.folder_raw_data,
            list_subset_train=args.subset_train,
            kind_of_data=args.kind_of_data,
            dataset_type=args.dataset_type,
            seed=args.seed,
            max_regular_distance_for_reward=args.max_regular_distance_for_reward,
            failure_reward_value=args.failure_reward_value,
            n_max_dataset_queries=int(args.n_max_dataset_queries),
            max_size_frontier=int(args.max_size_frontier),
            onnx_frontier_size=int(onnx_frontier_size),
            train_random_frontier_ratio=float(args.train_random_frontier_ratio),
            train_random_with_failure_ratio=float(args.train_random_frontier_with_failure_ratio),
            train_frontier_jaccard_threshold=float(args.train_frontier_jaccard_threshold),
            build_eval_data=False,
        )
        train_samples = train_samples if isinstance(train_samples, list) else list(train_samples)
        torch.save(train_samples, train_path)
        torch.save(dict(params or {}), params_path)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing training samples file: {train_path}. "
            "Run with --build-data true."
        )

    train_samples = torch.load(train_path, weights_only=False)
    params = torch.load(params_path, weights_only=False) if params_path.exists() else {}
    train_samples = train_samples if isinstance(train_samples, list) else list(train_samples)
    return train_samples, dict(params or {}), {"train": train_path, "params": params_path}


def _collect_strategy_frontier_entries(
    root: Path,
    kind_of_data: str,
    subset_filter: Optional[set[str]],
    max_regular_distance_for_reward: float,
    failure_reward_value: float,
    onnx_frontier_size: int,
    train_random_frontier_ratio: float,
    train_random_frontier_with_failure_ratio: float,
    seed: int,
) -> list[Dict[str, Any]]:
    if not root.exists() or not root.is_dir():
        return []

    dataset_entries: list[tuple[Path, Path]] = []
    for prob_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if subset_filter and prob_dir.name not in subset_filter:
            continue
        for csv_path in sorted(p for p in prob_dir.iterdir() if p.suffix.lower() == ".csv"):
            dataset_entries.append((prob_dir, csv_path))

    entries: list[Dict[str, Any]] = []
    for dataset_counter, (prob_dir, csv_path) in enumerate(dataset_entries):
        rows = read_frontier_csv(csv_path=csv_path, kind_of_data=kind_of_data)
        dataset_id = f"{prob_dir.name}/{csv_path.name}"
        strategy_frontiers, strategy_summary = build_tree_strategy_frontiers_for_dataset(
            rows=rows,
            kind_of_data=kind_of_data,
            dataset_id=dataset_id,
            onnx_frontier_size=int(onnx_frontier_size),
            random_frontier_ratio=float(train_random_frontier_ratio),
            random_frontier_with_failure_ratio=float(train_random_frontier_with_failure_ratio),
            max_regular_distance_for_reward=float(max_regular_distance_for_reward),
            failure_reward_value=float(failure_reward_value),
            seed=int(seed) + 7919 * dataset_counter + 31,
        )
        strategy_frontiers = list(strategy_frontiers)
        for frontier in strategy_frontiers:
            frontier["dataset_id"] = str(dataset_id)
            frontier["frontier_label"] = _normalize_frontier_label(
                frontier.get("frontier_label", FRONTIER_LABEL_COMMON)
            )

        entries.append(
            {
                "problem_name": str(prob_dir.name),
                "dataset_id": str(dataset_id),
                "csv_name": str(csv_path.name),
                "csv_stem": str(csv_path.stem),
                "n_rows": int(len(rows)),
                "n_strategy_frontiers": int(len(strategy_frontiers)),
                "strategy_summary": dict(strategy_summary or {}),
                "strategy_frontiers": strategy_frontiers,
            }
        )
    return entries


def _eval_frontier_successor_path_set(frontier: Dict[str, Any]) -> set[str]:
    successor_paths = frontier.get("successor_paths")
    if not isinstance(successor_paths, (list, tuple)):
        return set()
    out: set[str] = set()
    for raw in successor_paths:
        value = str(raw).strip()
        if value:
            out.add(value)
    return out


def _prune_eval_frontiers_by_jaccard_similarity(
    frontiers: Sequence[Dict[str, Any]],
    jaccard_similarity_threshold: float,
) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
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

    groups: Dict[Tuple[str, int], Dict[str, Any]] = {}
    token_to_id: Dict[str, int] = {}
    next_token_id = 0
    kept_frontiers: list[Dict[str, Any]] = []
    dropped = 0
    missing_successor_paths = 0

    for frontier in frontiers:
        successor_set = _eval_frontier_successor_path_set(frontier)
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

        frontier_size = int(len(successor_token_ids))
        group_key = (
            str(frontier.get("dataset_id", "")),
            int(frontier_size),
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
        required_overlap = _required_overlap_for_jaccard(
            size_a=int(frontier_size),
            size_b=int(frontier_size),
            threshold=float(threshold),
        )
        seed_token = min(
            successor_token_ids,
            key=lambda token_id: len(inverted_index.get(int(token_id), [])),
        )
        ordered_tokens: list[int] = [int(seed_token)]
        for token_id in successor_token_ids:
            if int(token_id) == int(seed_token):
                continue
            ordered_tokens.append(int(token_id))

        remaining_tokens = int(frontier_size)
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
                denom = int(frontier_size + size_b - int(new_overlap))
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
        sizes.append(int(frontier_size))
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


def _select_size_diverse_eval_frontiers(
    frontiers: Sequence[Dict[str, Any]],
    n_max_dataset_queries: int,
    seed: int,
) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
    target = int(max(0, int(n_max_dataset_queries)))
    if target <= 0 or not frontiers:
        return [], {
            "n_candidates": int(len(frontiers)),
            "n_selected": 0,
            "n_unique_sizes_available": 0,
            "selected_sizes": [],
        }

    by_size: Dict[int, list[Dict[str, Any]]] = {}
    for frontier in frontiers:
        size = int(len(frontier.get("successor_paths", [])))
        if size <= 0:
            continue
        by_size.setdefault(size, []).append(frontier)
    if not by_size:
        return [], {
            "n_candidates": int(len(frontiers)),
            "n_selected": 0,
            "n_unique_sizes_available": 0,
            "selected_sizes": [],
        }

    rng = torch.Generator()
    rng.manual_seed(int(seed))
    for size, bucket in by_size.items():
        if len(bucket) <= 1:
            continue
        perm = torch.randperm(len(bucket), generator=rng).tolist()
        by_size[size] = [bucket[i] for i in perm]

    unique_sizes = sorted(int(s) for s in by_size.keys())
    n_sizes_to_cover = int(min(len(unique_sizes), target))

    selected_sizes: list[int] = []
    if n_sizes_to_cover > 0 and unique_sizes:
        selected_sizes.append(int(unique_sizes[0]))
    if n_sizes_to_cover > 1 and int(unique_sizes[-1]) != int(unique_sizes[0]):
        selected_sizes.append(int(unique_sizes[-1]))
    while len(selected_sizes) < n_sizes_to_cover:
        best_size = None
        best_distance = -1.0
        for size in unique_sizes:
            if int(size) in selected_sizes:
                continue
            nearest = min(abs(float(size) - float(x)) for x in selected_sizes)
            if nearest > best_distance:
                best_distance = float(nearest)
                best_size = int(size)
        if best_size is None:
            break
        selected_sizes.append(int(best_size))
    selected_sizes = sorted(set(int(s) for s in selected_sizes))

    remaining_by_size = {int(s): list(by_size[int(s)]) for s in selected_sizes}
    selected_frontiers: list[Dict[str, Any]] = []

    for size in selected_sizes:
        if len(selected_frontiers) >= target:
            break
        bucket = remaining_by_size.get(int(size), [])
        if not bucket:
            continue
        selected_frontiers.append(bucket.pop(0))

    while len(selected_frontiers) < target:
        progressed = False
        for size in selected_sizes:
            if len(selected_frontiers) >= target:
                break
            bucket = remaining_by_size.get(int(size), [])
            if not bucket:
                continue
            selected_frontiers.append(bucket.pop(0))
            progressed = True
        if not progressed:
            break

    selected_size_counter: Dict[int, int] = {}
    for frontier in selected_frontiers:
        size = int(len(frontier.get("successor_paths", [])))
        selected_size_counter[size] = int(selected_size_counter.get(size, 0) + 1)

    return selected_frontiers, {
        "n_candidates": int(len(frontiers)),
        "n_selected": int(len(selected_frontiers)),
        "n_unique_sizes_available": int(len(unique_sizes)),
        "selected_sizes": [int(s) for s in selected_sizes],
        "selected_size_counts": {str(k): int(v) for k, v in sorted(selected_size_counter.items())},
    }


def _attach_strategy_query_metadata(
    frontiers: Sequence[Dict[str, Any]],
    dataset_id: str,
    source_tag: str,
) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for frontier in frontiers:
        item = dict(frontier)
        label = _normalize_frontier_label(item.get("frontier_label", FRONTIER_LABEL_COMMON))
        item["dataset_id"] = str(dataset_id)
        item["source_tag"] = str(source_tag)
        item["frontier_label"] = str(label)
        item["query_label"] = str(label)
        item["query_type"] = str(label)
        item["frontier_size"] = int(len(item.get("successor_paths", [])))
        out.append(item)
    return out


def _build_eval_strategy_split_from_entries(
    entries: Sequence[Dict[str, Any]],
    n_max_dataset_queries: int,
    eval_frontier_jaccard_threshold: float,
    seed: int,
    source_tag: str,
) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    split_frontiers: list[Dict[str, Any]] = []
    entry_summaries: list[Dict[str, Any]] = []
    for dataset_idx, entry in enumerate(entries):
        strategy_frontiers = list(entry.get("strategy_frontiers", []))
        dataset_id = str(entry.get("dataset_id", ""))
        if not strategy_frontiers:
            entry_summaries.append(
                {
                    "dataset_id": dataset_id,
                    "n_strategy_frontiers_total": 0,
                    "eval_jaccard_pruning": {
                        "threshold": float(eval_frontier_jaccard_threshold),
                        "n_total": 0,
                        "n_kept": 0,
                        "n_dropped": 0,
                        "n_missing_successor_paths": 0,
                        "n_groups": 0,
                    },
                    "eval_selection": {
                        "n_candidates": 0,
                        "n_selected": 0,
                        "n_unique_sizes_available": 0,
                        "selected_sizes": [],
                    },
                }
            )
            continue

        pruned, prune_stats = _prune_eval_frontiers_by_jaccard_similarity(
            frontiers=strategy_frontiers,
            jaccard_similarity_threshold=float(eval_frontier_jaccard_threshold),
        )
        selected, selection_stats = _select_size_diverse_eval_frontiers(
            frontiers=pruned,
            n_max_dataset_queries=int(n_max_dataset_queries),
            seed=int(seed) + 1299709 * dataset_idx + 101,
        )
        split_frontiers.extend(
            _attach_strategy_query_metadata(
                frontiers=selected,
                dataset_id=dataset_id,
                source_tag=source_tag,
            )
        )
        entry_summaries.append(
            {
                "dataset_id": dataset_id,
                "problem_name": str(entry.get("problem_name", "")),
                "csv_name": str(entry.get("csv_name", "")),
                "n_rows": int(entry.get("n_rows", 0)),
                "n_strategy_frontiers_total": int(len(strategy_frontiers)),
                "eval_jaccard_pruning": prune_stats,
                "eval_selection": selection_stats,
            }
        )
    return split_frontiers, entry_summaries


def _resolve_test_data_root(args, model_root: Path) -> Path:
    if str(args.folder_test_data).strip():
        return Path(str(args.folder_test_data).strip())
    return model_root / "test_data"


def _build_or_load_query_bundle(
    args,
    data_root: Path,
    model_root: Path,
    onnx_frontier_size: int,
    cache_namespace: str = "",
):
    data_dir = data_root / "processed_data"
    namespace = str(cache_namespace).strip()
    if namespace:
        data_dir = data_dir / namespace
    data_dir.mkdir(parents=True, exist_ok=True)
    query_path = data_dir / "query_bundle.json"
    legacy_query_candidates = [
        data_root / "data" / "query_bundle.json",
        data_root / "data_queries" / "query_bundle.json",
    ]
    test_data_root = _resolve_test_data_root(args=args, model_root=model_root)

    if (not bool(args.build_eval_data)) and (not query_path.exists()):
        for legacy_query_path in legacy_query_candidates:
            if legacy_query_path.exists():
                query_path.write_bytes(legacy_query_path.read_bytes())
                break

    can_load = (not bool(args.build_eval_data)) and query_path.exists()
    if can_load:
        with query_path.open("r", encoding="utf-8") as fh:
            bundle = json.load(fh)
        bundle_meta = bundle.get("meta", {}) if isinstance(bundle, dict) else {}
        bundle_format = str(bundle_meta.get("eval_bundle_format", "")).strip().lower()
        if bundle_format == "strategy_v1":
            stored_eval_size = int(bundle_meta.get("onnx_frontier_size_for_eval_data", 0) or 0)
            requested_eval_size = int(max(1, int(onnx_frontier_size)))
            if stored_eval_size >= requested_eval_size:
                return bundle, query_path, test_data_root

    train_entries = _collect_strategy_frontier_entries(
        root=Path(args.folder_raw_data),
        kind_of_data=args.kind_of_data,
        subset_filter=set(args.subset_train) if args.subset_train else None,
        max_regular_distance_for_reward=float(args.max_regular_distance_for_reward),
        failure_reward_value=float(args.failure_reward_value),
        onnx_frontier_size=int(onnx_frontier_size),
        train_random_frontier_ratio=float(args.train_random_frontier_ratio),
        train_random_frontier_with_failure_ratio=float(args.train_random_frontier_with_failure_ratio),
        seed=int(args.seed),
    )
    train_queries, train_eval_entry_summaries = _build_eval_strategy_split_from_entries(
        entries=train_entries,
        n_max_dataset_queries=int(args.n_max_dataset_queries),
        eval_frontier_jaccard_threshold=float(args.eval_frontier_jaccard_threshold),
        seed=int(args.seed) + 170141183,
        source_tag="train",
    )

    if test_data_root.exists() and test_data_root.is_dir():
        test_entries = _collect_strategy_frontier_entries(
            root=test_data_root,
            kind_of_data=args.kind_of_data,
            subset_filter=None,
            max_regular_distance_for_reward=float(args.max_regular_distance_for_reward),
            failure_reward_value=float(args.failure_reward_value),
            onnx_frontier_size=int(onnx_frontier_size),
            train_random_frontier_ratio=float(args.train_random_frontier_ratio),
            train_random_frontier_with_failure_ratio=float(args.train_random_frontier_with_failure_ratio),
            seed=int(args.seed) + 73,
        )
        test_queries, test_eval_entry_summaries = _build_eval_strategy_split_from_entries(
            entries=test_entries,
            n_max_dataset_queries=int(args.n_max_dataset_queries),
            eval_frontier_jaccard_threshold=float(args.eval_frontier_jaccard_threshold),
            seed=int(args.seed) + 170141183 + 73,
            source_tag="test",
        )
    else:
        test_entries = []
        test_queries = []
        test_eval_entry_summaries = []

    bundle = {
        "meta": {
            "eval_bundle_format": "strategy_v1",
            "kind_of_data": str(args.kind_of_data),
            "dataset_type": str(args.dataset_type),
            "seed": int(args.seed),
            "n_max_dataset_queries": int(args.n_max_dataset_queries),
            "max_size_frontier": int(args.max_size_frontier),
            "onnx_frontier_size_for_eval_data": int(onnx_frontier_size),
            "eval_frontier_jaccard_threshold": float(args.eval_frontier_jaccard_threshold),
            "train_random_frontier_ratio": float(args.train_random_frontier_ratio),
            "train_random_frontier_with_failure_ratio": float(
                args.train_random_frontier_with_failure_ratio
            ),
            "folder_raw_data": str(args.folder_raw_data),
            "folder_test_data": test_data_root.as_posix(),
        },
        "train_entries": [
            {
                "dataset_id": e["dataset_id"],
                "problem_name": e["problem_name"],
                "csv_name": e["csv_name"],
                "n_rows": e["n_rows"],
                "n_strategy_frontiers": e["n_strategy_frontiers"],
                "strategy_summary": e.get("strategy_summary", {}),
            }
            for e in train_entries
        ],
        "test_entries": [
            {
                "dataset_id": e["dataset_id"],
                "problem_name": e["problem_name"],
                "csv_name": e["csv_name"],
                "n_rows": e["n_rows"],
                "n_strategy_frontiers": e["n_strategy_frontiers"],
                "strategy_summary": e.get("strategy_summary", {}),
            }
            for e in test_entries
        ],
        "train_eval_entries": train_eval_entry_summaries,
        "test_eval_entries": test_eval_entry_summaries,
        "train": train_queries,
        "test": test_queries,
    }
    with query_path.open("w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)
    return bundle, query_path, test_data_root


def _flatten_query_splits(bundle: Dict[str, Any]) -> Dict[str, list[Dict[str, Any]]]:
    out: Dict[str, list[Dict[str, Any]]] = {}
    for source in ("train", "test"):
        source_payload = bundle.get(source, [])
        if isinstance(source_payload, list):
            out[source] = list(source_payload)
            continue
        # Legacy fallback: flatten old random/fifo/stress bundle format.
        if isinstance(source_payload, dict):
            merged_values: list[Dict[str, Any]] = []
            for query_type in ("random", "fifo", "stress"):
                values = source_payload.get(query_type, [])
                if not isinstance(values, list):
                    continue
                for item in values:
                    if not isinstance(item, dict):
                        continue
                    entry = dict(item)
                    label = _normalize_frontier_label(entry.get("frontier_label", query_type))
                    entry["query_label"] = str(label)
                    entry["query_type"] = str(label)
                    entry["frontier_label"] = str(label)
                    entry["source_tag"] = str(source)
                    merged_values.append(entry)
            out[source] = merged_values
            continue
        out[source] = []
    return out


def _query_counts(flat_query_splits: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, int]:
    return {split_name: int(len(frontiers)) for split_name, frontiers in flat_query_splits.items()}


def _slice_sample_bool_vector(
    sample: Dict[str, Any],
    key: str,
    frontier_size_cap: Optional[int] = None,
) -> torch.Tensor:
    raw = sample.get(key)
    if isinstance(raw, torch.Tensor):
        vec = raw.to(torch.bool).view(-1)
    elif isinstance(raw, (list, tuple)):
        vec = torch.tensor([bool(x) for x in raw], dtype=torch.bool).view(-1)
    else:
        vec = torch.zeros((0,), dtype=torch.bool)
    if frontier_size_cap is None:
        return vec
    k = int(max(0, int(frontier_size_cap)))
    return vec[:k]


def _slice_sample_numeric_vector(
    sample: Dict[str, Any],
    preferred_keys: Sequence[str],
    dtype: torch.dtype,
    frontier_size_cap: Optional[int] = None,
) -> torch.Tensor:
    vec = torch.zeros((0,), dtype=dtype)
    for key in preferred_keys:
        raw = sample.get(key)
        if isinstance(raw, torch.Tensor):
            vec = raw.to(dtype).view(-1)
            break
        if isinstance(raw, (list, tuple)):
            vec = torch.tensor(raw, dtype=dtype).view(-1)
            break
    if frontier_size_cap is None:
        return vec
    k = int(max(0, int(frontier_size_cap)))
    return vec[:k]


def _sample_has_failure_state(
    sample: Dict[str, Any],
    frontier_size_cap: Optional[int] = None,
) -> bool:
    has_failure = sample.get("frontier_has_failure")
    if has_failure is not None and frontier_size_cap is None:
        return bool(has_failure)

    failure_vec = _slice_sample_bool_vector(
        sample=sample,
        key="is_failure",
        frontier_size_cap=frontier_size_cap,
    )
    if failure_vec.numel() > 0:
        return bool(failure_vec.any().item())

    reward_vec = _slice_sample_numeric_vector(
        sample=sample,
        preferred_keys=("reward_target", "rewards"),
        dtype=torch.float32,
        frontier_size_cap=frontier_size_cap,
    )
    if reward_vec.numel() > 0:
        return bool((reward_vec <= float(FAILURE_REWARD_VALUE + FAILURE_EPS)).any().item())
    return False


def _sample_has_failure_and_solution_state(
    sample: Dict[str, Any],
    frontier_size_cap: Optional[int] = None,
) -> bool:
    failure_mask = _slice_sample_bool_vector(
        sample=sample,
        key="is_failure",
        frontier_size_cap=frontier_size_cap,
    )
    if failure_mask.numel() <= 0:
        reward_vec = _slice_sample_numeric_vector(
            sample=sample,
            preferred_keys=("reward_target", "rewards"),
            dtype=torch.float32,
            frontier_size_cap=frontier_size_cap,
        )
        if reward_vec.numel() <= 0:
            return False
        failure_mask = reward_vec <= float(FAILURE_REWARD_VALUE + FAILURE_EPS)
    return bool(failure_mask.any().item() and (~failure_mask).any().item())


def _limit_failure_frontiers_in_train_dataset(
    train_samples: Sequence[Dict[str, Any]],
    max_failure_states_per_dataset: float,
    seed: int,
) -> Tuple[list[Dict[str, Any]], Dict[str, int]]:
    if not train_samples:
        return [], {
            "n_total": 0,
            "n_no_failure": 0,
            "n_with_failure_total": 0,
            "n_with_failure_and_solution_total": 0,
            "n_with_failure_kept": 0,
            "n_with_failure_and_solution_kept": 0,
            "n_total_kept": 0,
        }

    no_failure_indices = []
    with_failure_indices = []
    with_failure_and_solution_indices = []
    for idx, sample in enumerate(train_samples):
        if _sample_has_failure_state(sample):
            with_failure_indices.append(idx)
            if _sample_has_failure_and_solution_state(sample):
                with_failure_and_solution_indices.append(idx)
        else:
            no_failure_indices.append(idx)

    max_failure_to_keep = int(len(no_failure_indices) * float(max_failure_states_per_dataset))
    max_failure_to_keep = max(0, min(max_failure_to_keep, len(with_failure_indices)))

    if max_failure_to_keep >= len(with_failure_indices):
        selected_failure_indices = set(with_failure_indices)
    elif max_failure_to_keep <= 0:
        selected_failure_indices = set()
    else:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        selected_positions = torch.randperm(
            len(with_failure_indices), generator=generator
        )[:max_failure_to_keep].tolist()
        selected_failure_indices = {with_failure_indices[pos] for pos in selected_positions}

    kept_indices = set(no_failure_indices) | selected_failure_indices
    filtered_train_samples = [
        sample for idx, sample in enumerate(train_samples) if idx in kept_indices
    ]
    selected_failure_and_solution_indices = (
        selected_failure_indices & set(with_failure_and_solution_indices)
    )

    return filtered_train_samples, {
        "n_total": len(train_samples),
        "n_no_failure": len(no_failure_indices),
        "n_with_failure_total": len(with_failure_indices),
        "n_with_failure_and_solution_total": len(with_failure_and_solution_indices),
        "n_with_failure_kept": len(selected_failure_indices),
        "n_with_failure_and_solution_kept": len(selected_failure_and_solution_indices),
        "n_total_kept": len(filtered_train_samples),
    }


def _sample_successor_id_set(sample: Dict[str, Any]) -> set[str]:
    successor_ids = sample.get("successor_ids")
    if not isinstance(successor_ids, (list, tuple)):
        return set()
    out: set[str] = set()
    for raw in successor_ids:
        value = str(raw).strip()
        if value:
            out.add(value)
    return out


def _sample_jaccard_bucket_key(sample: Dict[str, Any], frontier_size: int) -> Tuple[str, int]:
    return (
        str(sample.get("dataset_id", "")),
        int(frontier_size),
    )


def _required_overlap_for_jaccard(size_a: int, size_b: int, threshold: float) -> int:
    if threshold <= 0.0:
        return 1
    rhs = float(threshold) * float(int(size_a) + int(size_b)) / float(1.0 + float(threshold))
    return max(1, int(math.ceil(rhs - 1e-12)))


def _prune_near_duplicate_frontiers_by_jaccard(
    train_samples: Sequence[Dict[str, Any]],
    jaccard_similarity_threshold: float,
) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
    threshold = float(jaccard_similarity_threshold)
    if not train_samples:
        return [], {
            "threshold": float(threshold),
            "n_total": 0,
            "n_kept": 0,
            "n_dropped": 0,
            "n_missing_successor_ids": 0,
            "n_groups": 0,
        }

    groups: Dict[Tuple[str, int], Dict[str, Any]] = {}
    token_to_id: Dict[str, int] = {}
    next_token_id = 0
    kept_samples: list[Dict[str, Any]] = []
    dropped = 0
    missing_successor_ids = 0

    for sample in train_samples:
        successor_set = _sample_successor_id_set(sample)
        if not successor_set:
            kept_samples.append(sample)
            missing_successor_ids += 1
            continue
        successor_token_ids: set[int] = set()
        for token in successor_set:
            token_id = token_to_id.get(token)
            if token_id is None:
                token_id = int(next_token_id)
                token_to_id[token] = token_id
                next_token_id += 1
            successor_token_ids.add(int(token_id))

        frontier_size = int(len(successor_token_ids))
        group_key = _sample_jaccard_bucket_key(
            sample=sample,
            frontier_size=int(frontier_size),
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
        required_overlap = _required_overlap_for_jaccard(
            size_a=int(frontier_size),
            size_b=int(frontier_size),
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

        remaining_tokens = int(frontier_size)
        drop_current = False
        size_a = int(frontier_size)
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
        kept_samples.append(sample)

    return kept_samples, {
        "threshold": float(threshold),
        "n_total": int(len(train_samples)),
        "n_kept": int(len(kept_samples)),
        "n_dropped": int(dropped),
        "n_missing_successor_ids": int(missing_successor_ids),
        "n_groups": int(len(groups)),
    }


def _count_sample_states(
    sample: Dict[str, Any],
    frontier_size_cap: Optional[int] = None,
) -> Tuple[int, int]:
    failure_vec = _slice_sample_bool_vector(
        sample=sample,
        key="is_failure",
        frontier_size_cap=frontier_size_cap,
    )
    if failure_vec.numel() > 0:
        all_states = int(failure_vec.numel())
        failure_states = int(failure_vec.to(torch.long).sum().item())
        return failure_states, all_states

    reward_vec = _slice_sample_numeric_vector(
        sample=sample,
        preferred_keys=("reward_target", "rewards"),
        dtype=torch.float32,
        frontier_size_cap=frontier_size_cap,
    )
    return 0, int(reward_vec.numel())


def _build_train_state_stats(
    train_samples: Sequence[Dict[str, Any]],
    frontier_size_cap: Optional[int] = None,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    overall_failure_states = 0
    overall_all_states = 0
    overall_failure_and_solution_frontiers = 0
    per_dataset: Dict[str, Dict[str, int]] = {}

    for sample in train_samples:
        dataset_id = str(sample.get("dataset_id", ""))
        failure_states, all_states = _count_sample_states(
            sample,
            frontier_size_cap=frontier_size_cap,
        )
        failure_and_solution_frontier = _sample_has_failure_and_solution_state(
            sample,
            frontier_size_cap=frontier_size_cap,
        )
        overall_failure_states += int(failure_states)
        overall_all_states += int(all_states)
        if failure_and_solution_frontier:
            overall_failure_and_solution_frontiers += 1

        stats = per_dataset.setdefault(
            dataset_id,
            {
                "train_frontiers_after_filter": 0,
                "train_with_failure_and_solution_frontiers_after_filter": 0,
                "train_n_failure_states": 0,
                "train_n_all_states": 0,
            },
        )
        stats["train_frontiers_after_filter"] += 1
        if failure_and_solution_frontier:
            stats["train_with_failure_and_solution_frontiers_after_filter"] += 1
        stats["train_n_failure_states"] += int(failure_states)
        stats["train_n_all_states"] += int(all_states)

    return (
        {
            "train_n_failure_states": int(overall_failure_states),
            "train_n_all_states": int(overall_all_states),
            "train_with_failure_and_solution_frontiers_after_filter": int(
                overall_failure_and_solution_frontiers
            ),
        },
        per_dataset,
    )


def _infer_num_edge_labels(samples: Sequence[Dict[str, Any]]) -> int:
    max_label = 0
    found_label = False
    for sample in samples:
        for key in ("edge_attr", "goal_edge_attr"):
            edge_attr = sample.get(key)
            if not isinstance(edge_attr, torch.Tensor) or edge_attr.numel() == 0:
                continue
            edge_ids = edge_attr.view(-1).to(torch.long)
            if edge_ids.numel() == 0:
                continue
            min_label = int(edge_ids.min().item())
            if min_label < 0:
                raise ValueError(
                    f"Edge labels must be non-negative categorical IDs. Got min={min_label} in '{key}'."
                )
            max_label = max(max_label, int(edge_ids.max().item()))
            found_label = True
    return (max_label + 1) if found_label else 1


def _load_graph_tensors_no_pyg(path: str, dataset_type: str) -> _EvalGraphTensors:
    dataset_type_norm = str(dataset_type).upper()
    if dataset_type_norm not in VALID_DATASET_TYPES:
        raise ValueError(
            f"Unsupported dataset_type '{dataset_type}'. "
            f"Expected one of {sorted(VALID_DATASET_TYPES)}."
        )

    dot_src = Path(path).read_text()
    # Signed-id datasets may contain bare negative node IDs (e.g. -123).
    # pydot expects quoted negatives, so sanitize first.
    dot_src = _BARE_NEGATIVE_INT_RE.sub(r'"-\1"', dot_src)
    parsed = pydot.graph_from_dot_data(dot_src)
    if not parsed:
        raise ValueError(f"Failed to parse DOT graph: {path}")
    dot = parsed[0]
    graph_nx = nx.nx_pydot.from_pydot(dot)

    nodes = list(graph_nx.nodes())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(nodes)}

    if dataset_type_norm == "BITMASK":
        explicit_len = graph_nx.graph.get("bit_len", None)
        if explicit_len is not None:
            explicit_len = int(_strip_quotes(explicit_len))

        first = nodes[0] if nodes else None
        inferred_len = len(first) if isinstance(first, (str, list, tuple)) else None
        bit_len = explicit_len if explicit_len is not None else inferred_len
        if bit_len is None:
            raise ValueError("Cannot infer bit length from graph nodes.")

        def _to_bits(node_obj: object, expected_len: int | None) -> list[int]:
            if isinstance(node_obj, str):
                s = node_obj.strip()
                if not set(s) <= {"0", "1"}:
                    raise ValueError(f"Node '{node_obj}' is not a bitstring.")
                if expected_len is not None and len(s) != expected_len:
                    raise ValueError(f"Inconsistent bit length for node '{node_obj}'.")
                return [int(ch) for ch in s]
            if isinstance(node_obj, (list, tuple)):
                bits = [int(x) for x in node_obj]
                if not set(bits) <= {0, 1}:
                    raise ValueError(f"Node '{node_obj}' has non-binary values.")
                if expected_len is not None and len(bits) != expected_len:
                    raise ValueError(f"Inconsistent bit length for node '{node_obj}'.")
                return bits
            if isinstance(node_obj, int):
                if expected_len is None:
                    raise ValueError("bit_len is required for int node labels.")
                return [int(ch) for ch in format(node_obj, f"0{expected_len}b")]
            raise TypeError(f"Unsupported node label type: {type(node_obj)}")

        rows = [_to_bits(node, bit_len) for node in nodes]
        node_bits = torch.tensor(rows, dtype=torch.bool)
        node_features = node_bits.to(torch.float32)
        node_names = node_features.clone()
    elif dataset_type_norm == "HASHED":
        raw_ids = [
            _validate_int64_range_eval(
                _parse_numeric_node_label_eval(node),
                context=f"Node '{node}'",
            )
            for node in nodes
        ]
        node_names = torch.tensor(raw_ids, dtype=torch.int64)
        node_features = node_names.view(-1, 1).to(torch.int64)
    else:
        raw_ids = [_parse_numeric_node_label_eval(node) for node in nodes]
        for node, raw_id in zip(nodes, raw_ids):
            if raw_id < I64_MIN or raw_id > I64_MAX:
                raise ValueError(
                    f"Node '{node}' is out of int64 range [{I64_MIN}, {I64_MAX}] "
                    "for MAPPED dataset."
                )
        node_names = torch.tensor(raw_ids, dtype=torch.int64)
        node_features = node_names.view(-1, 1).to(torch.int64)

    edge_rows: list[list[int]] = []
    edge_attrs: list[list[int]] = []
    for src, dst, edge_data in graph_nx.edges(data=True):
        src_idx = int(node_to_idx[src])
        dst_idx = int(node_to_idx[dst])
        edge_label = int(_strip_quotes(edge_data.get("label", "0")))
        edge_rows.append([src_idx, dst_idx])
        edge_attrs.append([edge_label])

    if edge_rows:
        edge_index = torch.tensor(edge_rows, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.int64)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.int64)

    return _EvalGraphTensors(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_names=node_names,
    )


def _materialize_query_frontier_no_pyg(
    frontier: Dict[str, Any],
    kind_of_data: str,
    dataset_type: str,
    max_regular_distance_for_reward: float,
    failure_reward_value: float,
    graph_loader: Any,
) -> Dict[str, Any]:
    if kind_of_data not in {"merged", "separated"}:
        raise ValueError(f"Unsupported kind_of_data: {kind_of_data}")

    successor_graphs = [graph_loader(path) for path in frontier["successor_paths"]]
    goal_graph = None
    if kind_of_data == "separated":
        goal_path = str(frontier.get("goal_path", ""))
        if not goal_path:
            raise ValueError("Missing Goal path for kind_of_data='separated'.")
        goal_graph = graph_loader(goal_path)

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
        [float(r) <= float(failure_reward_value + FAILURE_EPS) for r in reward_values],
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
    if goal_graph is not None:
        sample["goal_node_features"] = goal_graph.node_features
        sample["goal_edge_index"] = goal_graph.edge_index
        sample["goal_edge_attr"] = goal_graph.edge_attr
    return sample


def _infer_num_candidates(
    sample: Dict[str, Any],
    frontier_size_cap: Optional[int] = None,
) -> int:
    for key in ("reward_target", "rewards", "distance_raw", "distances"):
        values = sample.get(key)
        if values is None:
            continue
        if isinstance(values, torch.Tensor):
            n_candidates = int(values.numel())
        else:
            try:
                n_candidates = int(len(values))
            except TypeError:
                continue
        if n_candidates > 0:
            if frontier_size_cap is None:
                return n_candidates
            return int(min(n_candidates, max(0, int(frontier_size_cap))))

    membership = sample.get("membership")
    if isinstance(membership, torch.Tensor) and membership.numel() > 0:
        n_candidates = int(membership.max().item()) + 1
        if frontier_size_cap is None:
            return n_candidates
        return int(min(n_candidates, max(0, int(frontier_size_cap))))
    return 0


FRONTIER_STRATEGY_LABELS = [
    FRONTIER_LABEL_GREEDY,
    FRONTIER_LABEL_CONSERVATIVE,
    FRONTIER_LABEL_RANDOM,
    FRONTIER_LABEL_COMMON,
]


def _normalize_frontier_label(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw in {
        FRONTIER_LABEL_GREEDY,
        FRONTIER_LABEL_CONSERVATIVE,
        FRONTIER_LABEL_RANDOM,
        FRONTIER_LABEL_COMMON,
    }:
        return raw
    return FRONTIER_LABEL_COMMON


def _build_train_frontier_definitions_for_dataloader(
    train_samples: Sequence[Dict[str, Any]],
    frontier_size_cap: Optional[int] = None,
) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for sample in train_samples:
        successor_ids_raw = sample.get("successor_ids")
        successor_ids = (
            [str(x) for x in successor_ids_raw]
            if isinstance(successor_ids_raw, (list, tuple))
            else []
        )
        if frontier_size_cap is not None:
            successor_ids = successor_ids[: int(max(0, int(frontier_size_cap)))]
        frontier_size = int(len(successor_ids))
        if frontier_size <= 0:
            frontier_size = int(_infer_num_candidates(sample, frontier_size_cap=frontier_size_cap))

        out.append(
            {
                "dataset_id": str(sample.get("dataset_id", "")),
                "predecessor_path": str(sample.get("predecessor_path", "")),
                "goal_path": str(sample.get("goal_path", "")),
                "frontier_label": _normalize_frontier_label(
                    sample.get("frontier_label", FRONTIER_LABEL_COMMON)
                ),
                "successor_ids": successor_ids,
                "frontier_size": int(frontier_size),
                "frontier_has_failure": int(
                    _sample_has_failure_state(
                        sample,
                        frontier_size_cap=frontier_size_cap,
                    )
                ),
            }
        )
    return out


def _build_train_frontier_size_label_distribution(
    train_samples: Sequence[Dict[str, Any]],
    frontier_size_cap: Optional[int] = None,
) -> Dict[str, Any]:
    counts_by_label: Dict[str, Dict[int, int]] = {
        label: {} for label in FRONTIER_STRATEGY_LABELS
    }
    for sample in train_samples:
        size = int(_infer_num_candidates(sample, frontier_size_cap=frontier_size_cap))
        if size <= 0:
            continue
        label = _normalize_frontier_label(sample.get("frontier_label", FRONTIER_LABEL_COMMON))
        label_counts = counts_by_label.setdefault(label, {})
        label_counts[size] = int(label_counts.get(size, 0) + 1)

    sizes = sorted(
        {
            int(size)
            for label_counts in counts_by_label.values()
            for size in label_counts.keys()
        }
    )
    by_label: Dict[str, Dict[str, Any]] = {}
    for label in FRONTIER_STRATEGY_LABELS:
        label_counts = counts_by_label.get(label, {})
        total = int(sum(label_counts.values()))
        by_size = []
        for size in sizes:
            count = int(label_counts.get(int(size), 0))
            by_size.append(
                {
                    "frontier_size": int(size),
                    "count": int(count),
                    "frequency": float(count / total) if total > 0 else 0.0,
                }
            )
        by_label[str(label)] = {
            "total": int(total),
            "by_size": by_size,
        }

    return {
        "sizes": [int(size) for size in sizes],
        "labels": by_label,
    }


def _plot_train_frontier_size_label_distribution(
    train_samples: Sequence[Dict[str, Any]],
    out_path: Path,
    frontier_size_cap: Optional[int] = None,
) -> Dict[str, Any]:
    dist = _build_train_frontier_size_label_distribution(
        train_samples,
        frontier_size_cap=frontier_size_cap,
    )
    sizes = [int(x) for x in dist.get("sizes", [])]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300, sharex=True, sharey=True)
    axis_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    for idx, label in enumerate(FRONTIER_STRATEGY_LABELS):
        ax = axis_list[idx]
        label_payload = dist.get("labels", {}).get(label, {})
        by_size = label_payload.get("by_size", [])
        y_counts = [int(item.get("count", 0)) for item in by_size]
        if sizes and any(v > 0 for v in y_counts):
            ax.plot(sizes, y_counts, marker="o", linewidth=1.8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label)
        ax.set_xlabel("Frontier Size")
        ax.set_ylabel("Frontier Count")
        ax.grid(alpha=0.2, linewidth=0.5)

    fig.suptitle("Train Frontier Size Distribution by Strategy", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    plt.savefig(out_path)
    plt.close()
    return dist


def _sample_rewards_tensor(sample: Dict[str, Any]) -> torch.Tensor:
    rewards = sample.get("reward_target", sample.get("rewards"))
    if isinstance(rewards, torch.Tensor):
        return rewards.detach().cpu().to(torch.float32).view(-1)
    if rewards is None:
        return torch.zeros((0,), dtype=torch.float32)
    try:
        return torch.tensor(rewards, dtype=torch.float32).view(-1)
    except (TypeError, ValueError):
        return torch.zeros((0,), dtype=torch.float32)


def _infer_static_onnx_output_len(ort_sess: Any) -> Optional[int]:
    try:
        out_shape = ort_sess.get_outputs()[0].shape
        if out_shape and isinstance(out_shape[0], int) and out_shape[0] > 0:
            return int(out_shape[0])
    except Exception:
        return None
    return None


def _build_metric_curve_by_size(
    trainer: RLFrontierTrainer,
    frontier_sizes: Sequence[int],
    values: Sequence[float],
) -> Dict[str, List[float]]:
    by_size: Dict[int, List[float]] = {}
    for size, value in zip(frontier_sizes, values):
        by_size.setdefault(int(size), []).append(float(value))
    ordered_sizes = sorted(by_size.keys())
    iqm: list[float] = []
    iqr_std: list[float] = []
    for size in ordered_sizes:
        stats = trainer._iqm_iqr_stats(by_size[size])
        iqm.append(float(stats["iqm"]))
        iqr_std.append(float(stats["iqr_std"]))
    return {
        "sizes": [int(s) for s in ordered_sizes],
        "iqm": iqm,
        "iqr_std": iqr_std,
    }


def _build_query_label_metrics(
    trainer: RLFrontierTrainer,
    query_labels: Sequence[str],
    chosen_rewards: Sequence[float],
    accuracies: Sequence[int],
    frontier_sizes: Sequence[int],
    abs_reward_gaps: Sequence[float],
) -> Dict[str, Dict[str, Any]]:
    labels = [_normalize_frontier_label(x) for x in query_labels]
    metrics: Dict[str, Dict[str, Any]] = {}
    for label in FRONTIER_STRATEGY_LABELS:
        mask = [lbl == label for lbl in labels]
        label_rewards = [float(r) for r, keep in zip(chosen_rewards, mask) if keep]
        label_accuracies = [int(a) for a, keep in zip(accuracies, mask) if keep]
        label_frontier_sizes = [int(s) for s, keep in zip(frontier_sizes, mask) if keep]
        label_abs_reward_gaps = [float(g) for g, keep in zip(abs_reward_gaps, mask) if keep]
        metrics[label] = {
            "label": str(label),
            "n_frontiers": int(len(label_rewards)),
            "reward_by_size": _build_metric_curve_by_size(
                trainer=trainer,
                frontier_sizes=label_frontier_sizes,
                values=label_rewards,
            ),
            "accuracy_by_size": _build_metric_curve_by_size(
                trainer=trainer,
                frontier_sizes=label_frontier_sizes,
                values=label_accuracies,
            ),
            "abs_reward_gap_by_size": _build_metric_curve_by_size(
                trainer=trainer,
                frontier_sizes=label_frontier_sizes,
                values=label_abs_reward_gaps,
            ),
            "regret_by_size": _build_metric_curve_by_size(
                trainer=trainer,
                frontier_sizes=label_frontier_sizes,
                values=label_abs_reward_gaps,
            ),
        }
    return metrics


def _build_eval_metrics_from_frontier_decisions(
    trainer: RLFrontierTrainer,
    chosen_rewards: Sequence[float],
    accuracies: Sequence[int],
    frontier_has_failure: Sequence[int],
    frontier_sizes: Sequence[int],
    abs_reward_gaps: Sequence[float],
    query_labels: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    regimes = trainer._build_regime_metrics(
        rewards=chosen_rewards,
        accuracies=accuracies,
        frontier_has_failure=frontier_has_failure,
        frontier_sizes=frontier_sizes,
        abs_reward_gaps=abs_reward_gaps,
    )
    all_reward_stats = regimes[REGIME_ALL]["reward_stats"]
    all_accuracy_stats = regimes[REGIME_ALL]["accuracy_stats"]
    all_abs_gap_stats = regimes[REGIME_ALL]["abs_reward_gap_stats"]

    labels = (
        [_normalize_frontier_label(x) for x in query_labels]
        if query_labels is not None
        else [FRONTIER_LABEL_COMMON for _ in chosen_rewards]
    )

    metrics: Dict[str, object] = {
        "n_frontiers": len(chosen_rewards),
        "chosen_rewards": [float(x) for x in chosen_rewards],
        "accuracies": [int(x) for x in accuracies],
        "frontier_has_failure": [int(x) for x in frontier_has_failure],
        "frontier_sizes": [int(x) for x in frontier_sizes],
        "abs_reward_gaps": [float(x) for x in abs_reward_gaps],
        "query_labels": [str(x) for x in labels],
        "regimes": regimes,
        "mean_reward": float(all_reward_stats["mean"]),
        "std_reward": float(all_reward_stats["std"]),
        "iqm_reward": float(all_reward_stats["iqm"]),
        "iqr_reward": float(all_reward_stats["iqr"]),
        "iqr_std_reward": float(all_reward_stats["iqr_std"]),
        "mean_accuracy": float(all_accuracy_stats["mean"]),
        "std_accuracy": float(all_accuracy_stats["std"]),
        "mean_abs_reward_gap": float(all_abs_gap_stats["mean"]),
        "std_abs_reward_gap": float(all_abs_gap_stats["std"]),
        "iqm_abs_reward_gap": float(all_abs_gap_stats["iqm"]),
        "iqr_abs_reward_gap": float(all_abs_gap_stats["iqr"]),
        "iqr_std_abs_reward_gap": float(all_abs_gap_stats["iqr_std"]),
        "mean_regret": float(all_abs_gap_stats["mean"]),
        "std_regret": float(all_abs_gap_stats["std"]),
        "iqm_regret": float(all_abs_gap_stats["iqm"]),
        "iqr_regret": float(all_abs_gap_stats["iqr"]),
        "iqr_std_regret": float(all_abs_gap_stats["iqr_std"]),
        "oracle_accuracy": float(all_accuracy_stats["mean"]),
    }
    metrics["query_label_metrics"] = _build_query_label_metrics(
        trainer=trainer,
        query_labels=labels,
        chosen_rewards=chosen_rewards,
        accuracies=accuracies,
        frontier_sizes=frontier_sizes,
        abs_reward_gaps=abs_reward_gaps,
    )
    metrics["query_type_metrics"] = metrics["query_label_metrics"]

    n_frontiers = len(chosen_rewards)
    n_failure_frontiers = int(sum(frontier_has_failure))
    metrics["n_mixed_frontiers"] = n_failure_frontiers
    metrics["n_failure_frontiers"] = n_failure_frontiers
    metrics["failure_frontier_rate"] = (
        float(n_failure_frontiers / n_frontiers) if n_frontiers > 0 else 0.0
    )
    metrics["n_failure_choices_on_mixed"] = 0
    metrics["failure_choice_rate_on_mixed"] = 0.0
    metrics["normalized_regret_mean"] = 0.0
    metrics["normalized_regret_std"] = 0.0
    metrics["normalized_regrets"] = [0.0 for _ in chosen_rewards]
    return metrics


def _create_onnx_eval_context(onnx_path: Path) -> OnnxEvalContext:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "Evaluation requires numpy and onnxruntime."
        ) from exc

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 4
    ort_sess = ort.InferenceSession(
        onnx_path.as_posix(),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_names = {str(x.name) for x in ort_sess.get_inputs()}
    return OnnxEvalContext(
        ort_sess=ort_sess,
        np_mod=np,
        static_onnx_len=_infer_static_onnx_output_len(ort_sess),
        use_goal_inputs=("goal_node_features" in input_names),
    )


def _run_onnx_single_frontier(
    sample: Dict[str, Any],
    onnx_ctx: OnnxEvalContext,
    dataset_type: str,
    use_goal_inputs: bool,
) -> Dict[str, Any]:
    n_candidates = _infer_num_candidates(sample)
    if n_candidates <= 0:
        return {"status": "error", "error": "sample has no candidates."}

    rewards_t = _sample_rewards_tensor(sample)
    if rewards_t.numel() <= 0:
        return {"status": "error", "error": "sample has no reward targets."}
    n_valid = max(0, min(int(n_candidates), int(rewards_t.numel())))
    if n_valid <= 0:
        return {"status": "error", "error": "sample has no valid reward-target candidates."}

    required_base = ["node_features", "edge_index", "edge_attr", "membership"]
    missing_base = [k for k in required_base if k not in sample]
    if missing_base:
        return {
            "status": "error",
            "error": f"sample is missing required tensors {missing_base}.",
        }

    node_feature_dtype, node_feature_np_dtype = _node_feature_dtypes_for_dataset(
        dataset_type
    )
    dataset_type_norm = str(dataset_type).upper()
    node_features_np = (
        sample["node_features"]
        .detach()
        .cpu()
        .to(node_feature_dtype)
        .numpy()
        .astype(node_feature_np_dtype)
    )
    if dataset_type_norm == "HASHED":
        node_features_np = node_features_np.reshape(-1)

    base_onnx_inputs = {
        "node_features": node_features_np,
        "edge_index": (
            sample["edge_index"].detach().cpu().to(torch.int64).numpy().astype("int64")
        ),
        "edge_attr": (
            sample["edge_attr"]
            .detach()
            .cpu()
            .to(torch.int64)
            .reshape(-1)
            .numpy()
            .astype("int64")
        ),
        "membership": (
            sample["membership"].detach().cpu().to(torch.int64).numpy().astype("int64")
        ),
    }

    if use_goal_inputs:
        goal_keys = ["goal_node_features", "goal_edge_index", "goal_edge_attr"]
        goal_missing = [k for k in goal_keys if k not in sample]
        if goal_missing:
            return {
                "status": "error",
                "error": f"sample is missing goal tensors {goal_missing}.",
            }
        goal_batch = sample.get("goal_batch")
        if goal_batch is None:
            goal_batch = torch.zeros(
                (int(sample["goal_node_features"].size(0)),),
                dtype=torch.int64,
            )
        else:
            goal_batch = goal_batch.to(torch.int64)

        goal_node_features_np = (
            sample["goal_node_features"]
            .detach()
            .cpu()
            .to(node_feature_dtype)
            .numpy()
            .astype(node_feature_np_dtype)
        )
        if dataset_type_norm == "HASHED":
            goal_node_features_np = goal_node_features_np.reshape(-1)
        base_onnx_inputs["goal_node_features"] = goal_node_features_np
        base_onnx_inputs["goal_edge_index"] = (
            sample["goal_edge_index"].detach().cpu().to(torch.int64).numpy().astype("int64")
        )
        base_onnx_inputs["goal_edge_attr"] = (
            sample["goal_edge_attr"]
            .detach()
            .cpu()
            .to(torch.int64)
            .reshape(-1)
            .numpy()
            .astype("int64")
        )
        base_onnx_inputs["goal_batch"] = (
            goal_batch.detach().cpu().to(torch.int64).numpy().astype("int64")
        )

    def _mask_numpy(mask_len: int, valid_len: int):
        valid = max(0, min(int(mask_len), int(valid_len)))
        out = torch.zeros((int(mask_len),), dtype=torch.uint8)
        if valid > 0:
            out[:valid] = 1
        return out.numpy().astype("uint8")

    mask_attempts = []
    if onnx_ctx.static_onnx_len is not None and int(onnx_ctx.static_onnx_len) > 0:
        mask_attempts.append(int(onnx_ctx.static_onnx_len))
    if int(n_candidates) not in mask_attempts:
        mask_attempts.append(int(n_candidates))
    if int(n_valid) not in mask_attempts:
        mask_attempts.append(int(n_valid))

    onnx_logits_np = None
    onnx_mask_len = None
    last_onnx_error = None
    for mask_len in mask_attempts:
        try:
            onnx_inputs = dict(base_onnx_inputs)
            onnx_inputs["mask"] = _mask_numpy(mask_len=mask_len, valid_len=n_valid)
            onnx_logits_np = (
                onnx_ctx.np_mod.asarray(
                    onnx_ctx.ort_sess.run(["logits"], onnx_inputs)[0],
                    dtype=onnx_ctx.np_mod.float32,
                )
                .reshape(-1)
                .astype(onnx_ctx.np_mod.float32)
            )
            onnx_mask_len = int(mask_len)
            break
        except Exception as exc:
            last_onnx_error = exc

    if onnx_logits_np is None:
        return {"status": "error", "error": f"ONNX inference failed: {last_onnx_error}"}

    valid_for_onnx = max(0, min(int(n_valid), int(onnx_logits_np.size)))
    if valid_for_onnx <= 0:
        return {"status": "error", "error": "ONNX returned no valid logits."}

    pred_idx = int(onnx_ctx.np_mod.argmax(onnx_logits_np[:valid_for_onnx]))
    chosen_reward = float(rewards_t[pred_idx].item())
    best_reward = float(rewards_t[:n_valid].max().item())
    abs_gap = abs(best_reward - chosen_reward)
    has_failure = int(
        bool(
            (rewards_t[:n_valid] <= float(FAILURE_REWARD_VALUE + FAILURE_EPS))
            .any()
            .item()
        )
    )
    is_correct = int(abs_gap <= float(FAILURE_EPS))

    return {
        "status": "ok",
        "chosen_reward": float(chosen_reward),
        "accuracy": int(is_correct),
        "frontier_has_failure": int(has_failure),
        "frontier_size": int(n_valid),
        "abs_reward_gap": float(abs_gap),
        "onnx_prediction_index": int(pred_idx),
        "onnx_mask_len_used": int(onnx_mask_len) if onnx_mask_len is not None else None,
    }


def _run_single_sample_pytorch_and_onnx(
    trainer: RLFrontierTrainer,
    sample: Dict[str, Any],
    onnx_ctx: OnnxEvalContext,
) -> Dict[str, Any]:
    n_candidates = _infer_num_candidates(sample)
    if n_candidates <= 0:
        return {"status": "error", "error": "sample has no candidates."}

    required_base = ["node_features", "edge_index", "edge_attr", "membership"]
    missing_base = [k for k in required_base if k not in sample]
    if missing_base:
        return {"status": "error", "error": f"sample is missing required tensors {missing_base}."}

    node_feature_dtype, node_feature_np_dtype = _node_feature_dtypes_for_dataset(
        trainer.model.dataset_type
    )
    dataset_type_norm = str(trainer.model.dataset_type).upper()
    node_features = sample["node_features"].to(node_feature_dtype)
    edge_index = sample["edge_index"].to(torch.int64)
    edge_attr = sample["edge_attr"].to(torch.int64)
    membership = sample["membership"].to(torch.int64)

    model_kwargs: Dict[str, Any] = {
        "node_features": node_features.to(trainer.device),
        "edge_index": edge_index.to(trainer.device),
        "edge_attr": edge_attr.to(trainer.device),
        "membership": membership.to(trainer.device),
        "candidate_batch": None,
    }
    if onnx_ctx.use_goal_inputs:
        goal_keys = ["goal_node_features", "goal_edge_index", "goal_edge_attr"]
        goal_missing = [k for k in goal_keys if k not in sample]
        if goal_missing:
            return {
                "status": "error",
                "error": f"sample is missing goal tensors {goal_missing}.",
            }
        goal_batch = sample.get("goal_batch")
        if goal_batch is None:
            goal_batch = torch.zeros(
                (int(sample["goal_node_features"].size(0)),),
                dtype=torch.int64,
            )
        else:
            goal_batch = goal_batch.to(torch.int64)
        model_kwargs["goal_node_features"] = sample["goal_node_features"].to(
            trainer.device, dtype=node_feature_dtype
        )
        model_kwargs["goal_edge_index"] = sample["goal_edge_index"].to(
            trainer.device, dtype=torch.int64
        )
        model_kwargs["goal_edge_attr"] = sample["goal_edge_attr"].to(
            trainer.device, dtype=torch.int64
        )
        model_kwargs["goal_batch"] = goal_batch.to(trainer.device)

    try:
        with torch.no_grad():
            trainer.model.eval()
            pytorch_logits = trainer.model(**model_kwargs).detach().cpu().to(torch.float32).view(-1)
        mask = torch.ones((int(pytorch_logits.numel()),), dtype=torch.bool)
        model_kwargs["mask"] = mask.to(trainer.device)
        with torch.no_grad():
            trainer.model.eval()
            pytorch_logits = trainer.model(**model_kwargs).detach().cpu().to(torch.float32).view(-1)
    except Exception as exc:
        return {"status": "error", "error": f"PyTorch inference failed: {exc}"}

    def _mask_numpy(mask_len: int, valid_len: int):
        valid = max(0, min(int(mask_len), int(valid_len)))
        out = torch.zeros((int(mask_len),), dtype=torch.uint8)
        if valid > 0:
            out[:valid] = 1
        return out.numpy().astype("uint8")

    node_features_np = node_features.detach().cpu().numpy().astype(node_feature_np_dtype)
    if dataset_type_norm == "HASHED":
        node_features_np = node_features_np.reshape(-1)
    base_onnx_inputs = {
        "node_features": node_features_np,
        "edge_index": edge_index.detach().cpu().numpy().astype("int64"),
        "edge_attr": edge_attr.detach().cpu().reshape(-1).numpy().astype("int64"),
        "membership": membership.detach().cpu().numpy().astype("int64"),
    }
    if onnx_ctx.use_goal_inputs:
        goal_batch = sample.get("goal_batch")
        if goal_batch is None:
            goal_batch = torch.zeros(
                (int(sample["goal_node_features"].size(0)),),
                dtype=torch.int64,
            )
        else:
            goal_batch = goal_batch.to(torch.int64)
        goal_node_features_np = (
            sample["goal_node_features"]
            .detach()
            .cpu()
            .to(node_feature_dtype)
            .numpy()
            .astype(node_feature_np_dtype)
        )
        if dataset_type_norm == "HASHED":
            goal_node_features_np = goal_node_features_np.reshape(-1)
        base_onnx_inputs["goal_node_features"] = goal_node_features_np
        base_onnx_inputs["goal_edge_index"] = (
            sample["goal_edge_index"].detach().cpu().numpy().astype("int64")
        )
        base_onnx_inputs["goal_edge_attr"] = (
            sample["goal_edge_attr"]
            .detach()
            .cpu()
            .to(torch.int64)
            .reshape(-1)
            .numpy()
            .astype("int64")
        )
        base_onnx_inputs["goal_batch"] = goal_batch.detach().cpu().numpy().astype("int64")

    onnx_logits_np = None
    onnx_mask_len = None
    last_onnx_error = None
    mask_attempts = []
    if onnx_ctx.static_onnx_len is not None:
        mask_attempts.append(int(onnx_ctx.static_onnx_len))
    if int(pytorch_logits.numel()) > 0 and int(pytorch_logits.numel()) not in mask_attempts:
        mask_attempts.append(int(pytorch_logits.numel()))
    if int(n_candidates) > 0 and int(n_candidates) not in mask_attempts:
        mask_attempts.append(int(n_candidates))

    for mask_len in mask_attempts:
        try:
            onnx_inputs = dict(base_onnx_inputs)
            onnx_inputs["mask"] = _mask_numpy(mask_len=mask_len, valid_len=n_candidates)
            onnx_logits_np = (
                onnx_ctx.np_mod.asarray(
                    onnx_ctx.ort_sess.run(["logits"], onnx_inputs)[0],
                    dtype=onnx_ctx.np_mod.float32,
                )
                .reshape(-1)
                .astype(onnx_ctx.np_mod.float32)
            )
            onnx_mask_len = int(mask_len)
            break
        except Exception as exc:
            last_onnx_error = exc

    if onnx_logits_np is None:
        return {"status": "error", "error": f"ONNX inference failed: {last_onnx_error}"}

    valid_for_onnx = max(0, min(int(n_candidates), int(onnx_logits_np.size)))
    onnx_pred_idx = int(onnx_ctx.np_mod.argmax(onnx_logits_np[:valid_for_onnx])) if valid_for_onnx > 0 else -1

    valid_for_torch = max(0, min(int(n_candidates), int(pytorch_logits.numel())))
    pytorch_pred_idx = (
        int(torch.argmax(pytorch_logits[:valid_for_torch]).item()) if valid_for_torch > 0 else -1
    )

    pytorch_logits_np = pytorch_logits.detach().cpu().numpy().astype("float32")
    max_abs_diff = None
    if onnx_logits_np.shape == pytorch_logits_np.shape:
        max_abs_diff = (
            float(onnx_ctx.np_mod.max(onnx_ctx.np_mod.abs(onnx_logits_np - pytorch_logits_np)))
            if onnx_logits_np.size > 0
            else 0.0
        )

    return {
        "status": "ok",
        "num_candidates_from_sample": int(n_candidates),
        "num_candidates_from_model": int(pytorch_logits.numel()),
        "onnx_mask_len_used": int(onnx_mask_len) if onnx_mask_len is not None else None,
        "onnx_prediction_index": int(onnx_pred_idx),
        "onnx_logits": onnx_logits_np.tolist(),
        "pytorch_prediction_index": int(pytorch_pred_idx),
        "pytorch_logits": pytorch_logits_np.tolist(),
        "prediction_match": bool(pytorch_pred_idx == onnx_pred_idx),
        "max_abs_logit_diff": max_abs_diff,
    }


def _cap_query_frontier_to_size(
    frontier: Dict[str, Any],
    frontier_size_cap: Optional[int],
) -> Dict[str, Any]:
    if frontier_size_cap is None:
        return dict(frontier)
    cap = int(max(0, int(frontier_size_cap)))
    out = dict(frontier)
    for key in ("successor_paths", "distances", "depths", "rewards"):
        values = out.get(key)
        if isinstance(values, list):
            out[key] = list(values[:cap])
        elif isinstance(values, tuple):
            out[key] = list(values[:cap])
    out["frontier_size"] = int(len(out.get("successor_paths", [])))
    return out


def _evaluate_onnx_query_split(
    trainer: RLFrontierTrainer,
    split_name: str,
    frontiers: Sequence[Dict[str, Any]],
    kind_of_data: str,
    dataset_type: str,
    max_regular_distance_for_reward: float,
    failure_reward_value: float,
    onnx_ctx: OnnxEvalContext,
    graph_cache: Dict[str, _EvalGraphTensors],
    frontier_size_cap: Optional[int] = None,
) -> Dict[str, Any]:
    dataset_type_norm = str(dataset_type).upper()

    def _load_graph(path: str) -> _EvalGraphTensors:
        key = str(path)
        graph = graph_cache.get(key)
        if graph is None:
            graph = _load_graph_tensors_no_pyg(key, dataset_type=dataset_type_norm)
            graph_cache[key] = graph
        return graph

    chosen_rewards: list[float] = []
    accuracies: list[int] = []
    frontier_has_failure: list[int] = []
    frontier_sizes: list[int] = []
    abs_reward_gaps: list[float] = []
    query_labels: list[str] = []
    failed_frontiers: list[Dict[str, Any]] = []

    for idx, frontier in enumerate(
        tqdm(frontiers, desc=f"Evaluating {split_name}", leave=False)
    ):
        capped_frontier = _cap_query_frontier_to_size(
            frontier=frontier,
            frontier_size_cap=frontier_size_cap,
        )
        if int(len(capped_frontier.get("successor_paths", []))) <= 0:
            failed_frontiers.append(
                {
                    "query_index": int(idx),
                    "dataset_id": str(frontier.get("dataset_id", "")),
                    "predecessor_path": str(frontier.get("predecessor_path", "")),
                    "error": "query has no candidates after frontier-size cap.",
                }
            )
            continue
        try:
            sample = _materialize_query_frontier_no_pyg(
                frontier=capped_frontier,
                kind_of_data=kind_of_data,
                dataset_type=dataset_type_norm,
                max_regular_distance_for_reward=float(max_regular_distance_for_reward),
                failure_reward_value=float(failure_reward_value),
                graph_loader=_load_graph,
            )
        except Exception as exc:
            failed_frontiers.append(
                {
                    "query_index": int(idx),
                    "dataset_id": str(capped_frontier.get("dataset_id", "")),
                    "predecessor_path": str(capped_frontier.get("predecessor_path", "")),
                    "error": f"materialization failed: {exc}",
                }
            )
            continue

        result = _run_onnx_single_frontier(
            sample=sample,
            onnx_ctx=onnx_ctx,
            dataset_type=dataset_type_norm,
            use_goal_inputs=bool(onnx_ctx.use_goal_inputs),
        )
        if result.get("status") != "ok":
            failed_frontiers.append(
                {
                    "query_index": int(idx),
                    "dataset_id": str(capped_frontier.get("dataset_id", "")),
                    "predecessor_path": str(capped_frontier.get("predecessor_path", "")),
                    "error": str(result.get("error", "unknown error")),
                }
            )
            continue

        chosen_rewards.append(float(result["chosen_reward"]))
        accuracies.append(int(result["accuracy"]))
        frontier_has_failure.append(int(result["frontier_has_failure"]))
        frontier_sizes.append(int(result["frontier_size"]))
        abs_reward_gaps.append(float(result["abs_reward_gap"]))
        query_labels.append(
            _normalize_frontier_label(
                capped_frontier.get(
                    "query_label",
                    capped_frontier.get("frontier_label", FRONTIER_LABEL_COMMON),
                )
            )
        )

    metrics = _build_eval_metrics_from_frontier_decisions(
        trainer=trainer,
        chosen_rewards=chosen_rewards,
        accuracies=accuracies,
        frontier_has_failure=frontier_has_failure,
        frontier_sizes=frontier_sizes,
        abs_reward_gaps=abs_reward_gaps,
        query_labels=query_labels,
    )
    metrics["split_name"] = str(split_name)
    metrics["n_attempted_frontiers"] = int(len(frontiers))
    metrics["n_failed_frontiers"] = int(len(failed_frontiers))
    metrics["n_evaluated_frontiers"] = int(len(chosen_rewards))
    metrics["onnx_requires_goal_inputs"] = bool(onnx_ctx.use_goal_inputs)
    metrics["onnx_static_output_len"] = (
        int(onnx_ctx.static_onnx_len) if onnx_ctx.static_onnx_len is not None else None
    )
    metrics["frontier_size_cap_for_eval"] = (
        int(frontier_size_cap) if frontier_size_cap is not None else None
    )
    if failed_frontiers:
        metrics["failed_frontiers"] = failed_frontiers
    return metrics


def _init_eval_trackers(
    trainer: RLFrontierTrainer,
    eval_onnx_root: Path,
    query_splits: Dict[str, Sequence[Dict[str, Any]]],
) -> Dict[str, EvalSplitTracker]:
    trackers: Dict[str, EvalSplitTracker] = {}
    for split_name, queries in query_splits.items():
        if not queries:
            continue
        eval_dir = eval_onnx_root / split_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        trackers[split_name] = EvalSplitTracker(
            split_name=split_name,
            eval_dir=eval_dir,
            history={
                "epochs": [],
                "iqm_reward": [],
                "iqr_std_reward": [],
                "iqm_abs_reward_gap": [],
                "iqr_std_abs_reward_gap": [],
            },
            regime_history=trainer._empty_regime_history(),
        )
    return trackers


def _plot_query_label_metric_curves(
    out_path: Path,
    split_name: str,
    metric_title: str,
    ylabel: str,
    query_label_metrics: Dict[str, Any],
    curve_key: str,
    y_lim: Optional[Tuple[float, float]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4), dpi=300)
    plotted = False
    for label in FRONTIER_STRATEGY_LABELS:
        label_metrics = query_label_metrics.get(label, {})
        curve = label_metrics.get(curve_key, {})
        sizes = [int(x) for x in curve.get("sizes", [])]
        iqm = [float(x) for x in curve.get("iqm", [])]
        if not sizes or not iqm or len(sizes) != len(iqm):
            continue
        plotted = True
        plt.plot(sizes, iqm, marker="o", linewidth=1.8, label=str(label))

    if plotted:
        plt.legend(loc="best")
    else:
        plt.text(0.5, 0.5, "No query-label data", ha="center", va="center")
    plt.xlabel("Frontier Size")
    plt.ylabel(ylabel)
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])
    plt.title(f"{split_name.upper()} | {metric_title} by Frontier Size and Query Label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _save_eval_query_label_step_plots(
    tracker: EvalSplitTracker,
    epoch: int,
    eval_step: int,
    metrics: Dict[str, Any],
) -> None:
    query_label_metrics = metrics.get("query_label_metrics", {})
    if not isinstance(query_label_metrics, dict) or not query_label_metrics:
        return

    step_dir = tracker.eval_dir / f"step_{int(eval_step):04d}_epoch_{int(epoch):04d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    _plot_query_label_metric_curves(
        out_path=step_dir / "query_label_reward_by_frontier_size.png",
        split_name=tracker.split_name,
        metric_title="Reward (IQM)",
        ylabel="Reward",
        query_label_metrics=query_label_metrics,
        curve_key="reward_by_size",
        y_lim=(-1.0, 0.05),
    )
    _plot_query_label_metric_curves(
        out_path=step_dir / "query_label_accuracy_by_frontier_size.png",
        split_name=tracker.split_name,
        metric_title="Accuracy (IQM)",
        ylabel="Accuracy",
        query_label_metrics=query_label_metrics,
        curve_key="accuracy_by_size",
        y_lim=(-0.05, 1.05),
    )
    _plot_query_label_metric_curves(
        out_path=step_dir / "query_label_regret_by_frontier_size.png",
        split_name=tracker.split_name,
        metric_title="Regret |best reward - taken reward| (IQM)",
        ylabel="Regret",
        query_label_metrics=query_label_metrics,
        curve_key="regret_by_size",
        y_lim=(0.0, 1.05),
    )

    _plot_query_label_metric_curves(
        out_path=tracker.eval_dir / "query_label_reward_by_frontier_size_latest.png",
        split_name=tracker.split_name,
        metric_title="Reward (IQM)",
        ylabel="Reward",
        query_label_metrics=query_label_metrics,
        curve_key="reward_by_size",
        y_lim=(-1.0, 0.05),
    )
    _plot_query_label_metric_curves(
        out_path=tracker.eval_dir / "query_label_accuracy_by_frontier_size_latest.png",
        split_name=tracker.split_name,
        metric_title="Accuracy (IQM)",
        ylabel="Accuracy",
        query_label_metrics=query_label_metrics,
        curve_key="accuracy_by_size",
        y_lim=(-0.05, 1.05),
    )
    _plot_query_label_metric_curves(
        out_path=tracker.eval_dir / "query_label_regret_by_frontier_size_latest.png",
        split_name=tracker.split_name,
        metric_title="Regret |best reward - taken reward| (IQM)",
        ylabel="Regret",
        query_label_metrics=query_label_metrics,
        curve_key="regret_by_size",
        y_lim=(0.0, 1.05),
    )


def _update_eval_tracker(
    trainer: RLFrontierTrainer,
    tracker: EvalSplitTracker,
    epoch: int,
    eval_step: int,
    metrics: Dict[str, Any],
) -> None:
    tracker.history["epochs"].append(int(epoch))
    tracker.history["iqm_reward"].append(float(metrics.get("iqm_reward", 0.0)))
    tracker.history["iqr_std_reward"].append(float(metrics.get("iqr_std_reward", 0.0)))
    tracker.history["iqm_abs_reward_gap"].append(float(metrics.get("iqm_abs_reward_gap", 0.0)))
    tracker.history["iqr_std_abs_reward_gap"].append(
        float(metrics.get("iqr_std_abs_reward_gap", 0.0))
    )

    trainer._append_regime_history(
        regime_history=tracker.regime_history,
        epoch=int(epoch),
        regimes=metrics["regimes"],
    )
    trainer._save_eval_step_regime_plots(
        eval_dir=tracker.eval_dir,
        eval_name=tracker.split_name,
        epoch=int(epoch),
        eval_step=int(eval_step),
        metrics=metrics,
    )
    trainer._plot_iqm_with_iqr_std_band(
        out_path=tracker.eval_dir / "reward_iqm_iqr_std_over_eval_steps.png",
        x=tracker.history["epochs"],
        iqm=tracker.history["iqm_reward"],
        iqr_std=tracker.history["iqr_std_reward"],
        xlabel="Epoch (eval checkpoints)",
        ylabel=f"{tracker.split_name} Reward",
        y_lim=(-1.0, 0.0),
        title=f"{tracker.split_name} Reward IQM ± IQR-STD",
    )
    trainer._plot_iqm_with_iqr_std_band(
        out_path=tracker.eval_dir / "abs_reward_gap_iqm_iqr_std_over_eval_steps.png",
        x=tracker.history["epochs"],
        iqm=tracker.history["iqm_abs_reward_gap"],
        iqr_std=tracker.history["iqr_std_abs_reward_gap"],
        xlabel="Epoch (eval checkpoints)",
        ylabel="|best reward - taken reward|",
        y_lim=(0.0, 1.05),
        title=f"{tracker.split_name} Absolute Reward Gap IQM ± IQR-STD",
    )
    trainer._save_global_regime_history_plots(
        eval_dir=tracker.eval_dir,
        eval_name=tracker.split_name,
        regime_history=tracker.regime_history,
    )
    _save_eval_query_label_step_plots(
        tracker=tracker,
        epoch=int(epoch),
        eval_step=int(eval_step),
        metrics=metrics,
    )

    with (tracker.eval_dir / "history.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "epochs": tracker.history["epochs"],
                "iqm_reward": tracker.history["iqm_reward"],
                "iqr_std_reward": tracker.history["iqr_std_reward"],
                "iqm_abs_reward_gap": tracker.history["iqm_abs_reward_gap"],
                "iqr_std_abs_reward_gap": tracker.history["iqr_std_abs_reward_gap"],
                "regimes": tracker.regime_history,
                "latest_query_label_metrics": metrics.get("query_label_metrics", {}),
            },
            fh,
            indent=2,
        )


def _select_first_query_for_example(
    query_splits: Dict[str, Sequence[Dict[str, Any]]],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    preferred_order = [
        "train",
        "test",
    ]
    for split_name in preferred_order:
        frontiers = query_splits.get(split_name, [])
        if frontiers:
            return split_name, dict(frontiers[0])
    for split_name, frontiers in query_splits.items():
        if frontiers:
            return split_name, dict(frontiers[0])
    return None, None


def _run_example_parity_check(
    trainer: RLFrontierTrainer,
    onnx_ctx: OnnxEvalContext,
    query_splits: Dict[str, Sequence[Dict[str, Any]]],
    kind_of_data: str,
    dataset_type: str,
    max_regular_distance_for_reward: float,
    failure_reward_value: float,
    frontier_size_cap: Optional[int],
    report_path: Path,
) -> None:
    split_name, query = _select_first_query_for_example(query_splits)
    if query is None:
        report = {
            "status": "error",
            "error": "No query available for example parity check.",
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        return

    dataset_type_norm = str(dataset_type).upper()
    graph_cache: Dict[str, _EvalGraphTensors] = {}

    def _load_graph(path: str) -> _EvalGraphTensors:
        key = str(path)
        graph = graph_cache.get(key)
        if graph is None:
            graph = _load_graph_tensors_no_pyg(key, dataset_type=dataset_type_norm)
            graph_cache[key] = graph
        return graph

    query_capped = _cap_query_frontier_to_size(
        frontier=query,
        frontier_size_cap=frontier_size_cap,
    )

    try:
        sample = _materialize_query_frontier_no_pyg(
            frontier=query_capped,
            kind_of_data=kind_of_data,
            dataset_type=dataset_type_norm,
            max_regular_distance_for_reward=float(max_regular_distance_for_reward),
            failure_reward_value=float(failure_reward_value),
            graph_loader=_load_graph,
        )
        parity = _run_single_sample_pytorch_and_onnx(
            trainer=trainer,
            sample=sample,
            onnx_ctx=onnx_ctx,
        )
        report = {
            "status": str(parity.get("status", "error")),
            "split_name": split_name,
            "dataset_id": str(query_capped.get("dataset_id", "")),
            "predecessor_path": str(query_capped.get("predecessor_path", "")),
            "result": parity,
        }
    except Exception as exc:
        report = {
            "status": "error",
            "split_name": split_name,
            "dataset_id": str(query_capped.get("dataset_id", "")),
            "predecessor_path": str(query_capped.get("predecessor_path", "")),
            "error": str(exc),
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


def _evaluate_all_splits_with_onnx(
    trainer: RLFrontierTrainer,
    onnx_path: Path,
    query_splits: Dict[str, Sequence[Dict[str, Any]]],
    kind_of_data: str,
    dataset_type: str,
    max_regular_distance_for_reward: float,
    failure_reward_value: float,
    frontier_size_cap: Optional[int] = None,
    trackers: Optional[Dict[str, EvalSplitTracker]] = None,
    epoch: Optional[int] = None,
    eval_step: Optional[int] = None,
    summary_path: Optional[Path] = None,
    example_report_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    onnx_ctx = _create_onnx_eval_context(onnx_path=onnx_path)
    graph_cache: Dict[str, _EvalGraphTensors] = {}
    all_metrics: Dict[str, Dict[str, Any]] = {}

    for split_name, frontiers in query_splits.items():
        if not frontiers:
            continue
        metrics = _evaluate_onnx_query_split(
            trainer=trainer,
            split_name=split_name,
            frontiers=frontiers,
            kind_of_data=kind_of_data,
            dataset_type=dataset_type,
            max_regular_distance_for_reward=float(max_regular_distance_for_reward),
            failure_reward_value=float(failure_reward_value),
            onnx_ctx=onnx_ctx,
            graph_cache=graph_cache,
            frontier_size_cap=frontier_size_cap,
        )
        all_metrics[split_name] = metrics
        if (
            trackers is not None
            and epoch is not None
            and eval_step is not None
            and split_name in trackers
        ):
            _update_eval_tracker(
                trainer=trainer,
                tracker=trackers[split_name],
                epoch=int(epoch),
                eval_step=int(eval_step),
                metrics=metrics,
            )

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "epoch": int(epoch) if epoch is not None else None,
                    "eval_step": int(eval_step) if eval_step is not None else None,
                    "onnx_path": onnx_path.as_posix(),
                    "splits": all_metrics,
                },
                fh,
                indent=2,
            )

    if example_report_path is not None:
        _run_example_parity_check(
            trainer=trainer,
            onnx_ctx=onnx_ctx,
            query_splits=query_splits,
            kind_of_data=kind_of_data,
            dataset_type=dataset_type,
            max_regular_distance_for_reward=float(max_regular_distance_for_reward),
            failure_reward_value=float(failure_reward_value),
            frontier_size_cap=frontier_size_cap,
            report_path=example_report_path,
        )

    return all_metrics


def _select_eval_score(
    eval_metrics: Dict[str, Dict[str, Any]],
    preferred_order: Sequence[str],
) -> Tuple[float, Optional[str]]:
    for split_name in preferred_order:
        metrics = eval_metrics.get(split_name)
        if not metrics:
            continue
        if int(metrics.get("n_evaluated_frontiers", 0)) <= 0:
            continue
        return float(metrics.get("mean_abs_reward_gap", float("inf"))), split_name
    for split_name, metrics in eval_metrics.items():
        if int(metrics.get("n_evaluated_frontiers", 0)) <= 0:
            continue
        return float(metrics.get("mean_abs_reward_gap", float("inf"))), split_name
    return float("inf"), None


def _save_best_model_performance(
    out_path: Path,
    best_eval_epoch: Optional[int],
    best_eval_regret: Optional[float],
    best_eval_reward: Optional[float],
    best_eval_split: Optional[str],
    best_metrics_by_split: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "best_model_name": "frontier_policy.pt",
                "selection_metric": "mean_abs_reward_gap",
                "selection_goal": "minimize",
                "best_eval_epoch": best_eval_epoch,
                "best_eval_regret": best_eval_regret,
                "best_eval_reward": best_eval_reward,
                "best_eval_split": best_eval_split,
                "best_metrics_by_split": best_metrics_by_split,
            },
            fh,
            indent=2,
        )


def _find_checkpoint_to_load(best_ckpt: Path, model_ckpt: Path) -> Optional[Path]:
    for candidate in (best_ckpt, model_ckpt):
        if candidate.exists():
            return candidate
    return None


def _prepare_train_context(
    args,
    data_root: Path,
    onnx_frontier_size: int,
    cache_namespace: str = "",
) -> Dict[str, Any]:
    train_samples, sample_params, sample_paths = _build_or_load_train_samples(
        args=args,
        data_root=data_root,
        onnx_frontier_size=int(onnx_frontier_size),
        cache_namespace=str(cache_namespace),
    )
    refresh_frontier_sample_targets(
        train_samples,
        m_failed_state=0.0,
        max_regular_distance_for_reward=float(args.max_regular_distance_for_reward),
        failure_reward_value=float(args.failure_reward_value),
    )

    train_samples, train_filter_stats = _limit_failure_frontiers_in_train_dataset(
        train_samples=train_samples,
        max_failure_states_per_dataset=float(args.max_failure_states_per_dataset),
        seed=int(args.seed),
    )
    train_samples, train_jaccard_prune_stats = _prune_near_duplicate_frontiers_by_jaccard(
        train_samples=train_samples,
        jaccard_similarity_threshold=float(args.train_frontier_jaccard_threshold),
    )
    if not train_samples:
        raise ValueError(
            f"No training frontiers loaded from {sample_paths['train']} after filtering."
        )

    train_loader = build_frontier_dataloader(
        samples=train_samples,
        batch_size=args.batch_size,
        seed=int(args.seed),
        num_workers=0,
        pad_frontiers=True,
        shuffle=True,
    )
    return {
        "train_samples": train_samples,
        "sample_params": sample_params,
        "sample_paths": sample_paths,
        "train_filter_stats": train_filter_stats,
        "train_jaccard_prune_stats": train_jaccard_prune_stats,
        "train_loader": train_loader,
        "base_onnx_frontier_size": int(onnx_frontier_size),
    }


def _prepare_eval_context(
    args,
    data_root: Path,
    model_root: Path,
    onnx_frontier_size: int,
    cache_namespace: str = "",
) -> Dict[str, Any]:
    query_bundle, query_bundle_path, test_data_root = _build_or_load_query_bundle(
        args=args,
        data_root=data_root,
        model_root=model_root,
        onnx_frontier_size=int(onnx_frontier_size),
        cache_namespace=str(cache_namespace),
    )
    flat_query_splits = _flatten_query_splits(query_bundle)

    # Materialize evaluation frontiers into samples and build DataLoaders
    eval_samples: Dict[str, list[Dict[str, Any]]] = {}
    eval_loaders: Dict[str, Any] = {}

    dataset_type_norm = str(args.dataset_type).upper()
    kind_of_data = str(args.kind_of_data)
    frontier_cap = int(max(1, int(onnx_frontier_size)))
    graph_cache: Dict[str, _EvalGraphTensors] = {}

    def _load_graph(path: str) -> _EvalGraphTensors:
        key = str(path)
        graph = graph_cache.get(key)
        if graph is None:
            graph = _load_graph_tensors_no_pyg(key, dataset_type=dataset_type_norm)
            graph_cache[key] = graph
        return graph

    for split_name in ("train", "test"):
        frontiers = flat_query_splits.get(split_name, []) or []
        samples: list[Dict[str, Any]] = []
        failed_indices: list[int] = []
        for idx, frontier in enumerate(frontiers):
            capped_frontier = _cap_query_frontier_to_size(
                frontier=frontier, frontier_size_cap=frontier_cap
            )
            if int(len(capped_frontier.get("successor_paths", []))) <= 0:
                failed_indices.append(idx)
                continue
            try:
                sample = _materialize_query_frontier_no_pyg(
                    frontier=capped_frontier,
                    kind_of_data=kind_of_data,
                    dataset_type=dataset_type_norm,
                    max_regular_distance_for_reward=float(args.max_regular_distance_for_reward),
                    failure_reward_value=float(args.failure_reward_value),
                    graph_loader=_load_graph,
                )
                samples.append(sample)
            except Exception:
                # Skip problematic frontiers but continue materializing others
                failed_indices.append(idx)
                continue

        eval_samples[split_name] = samples
        if samples:
            eval_batch_size = int(args.eval_batch_size) if getattr(args, "eval_batch_size", None) else int(args.batch_size)
            eval_num_workers = int(getattr(args, "eval_num_workers", 0))
            loader = build_frontier_dataloader(
                samples=samples,
                batch_size=eval_batch_size,
                seed=int(args.seed),
                num_workers=eval_num_workers,
                pad_frontiers=True,
                shuffle=False,
            )
            eval_loaders[split_name] = loader
        else:
            eval_loaders[split_name] = None

    return {
        "query_bundle": query_bundle,
        "query_bundle_path": query_bundle_path,
        "test_data_root": test_data_root,
        "flat_query_splits": flat_query_splits,
        "base_onnx_frontier_size": int(onnx_frontier_size),
        "eval_samples": eval_samples,
        "eval_loaders": eval_loaders,
    }


def _adapt_train_batch_to_frontier_size(
    raw_batch: Dict[str, Any],
    frontier_size: int,
) -> Dict[str, Any]:
    # Keep a deterministic prefix per frontier to preserve candidate ordering
    # semantics while switching target frontier sizes at runtime.
    frontier_ptr = raw_batch.get("frontier_ptr")
    if not isinstance(frontier_ptr, torch.Tensor) or frontier_ptr.numel() <= 1:
        return raw_batch

    target_size = int(frontier_size)
    if target_size <= 0:
        raise ValueError(f"frontier_size must be > 0, got {target_size}.")

    frontier_ptr_t = frontier_ptr.to(torch.long).view(-1)
    n_frontiers = int(frontier_ptr_t.numel() - 1)
    old_total_candidates = int(frontier_ptr_t[-1].item())
    if old_total_candidates <= 0:
        return raw_batch

    keep_counts: List[int] = []
    keep_segments: List[torch.Tensor] = []
    all_unchanged = True
    for i in range(n_frontiers):
        s = int(frontier_ptr_t[i].item())
        e = int(frontier_ptr_t[i + 1].item())
        n = int(max(0, e - s))
        keep_n = int(min(n, target_size))
        keep_counts.append(keep_n)
        if keep_n != n:
            all_unchanged = False
        if keep_n > 0:
            keep_segments.append(torch.arange(s, s + keep_n, dtype=torch.long))

    if all_unchanged:
        return raw_batch

    keep_idx = (
        torch.cat(keep_segments, dim=0)
        if keep_segments
        else torch.zeros((0,), dtype=torch.long)
    )
    new_total_candidates = int(keep_idx.numel())
    remap = torch.full((old_total_candidates,), -1, dtype=torch.long)
    if new_total_candidates > 0:
        remap[keep_idx] = torch.arange(new_total_candidates, dtype=torch.long)

    out: Dict[str, Any] = dict(raw_batch)
    candidate_tensor_keys = (
        "action_map",
        "reward_target",
        "rewards",
        "distance_raw",
        "distances",
        "is_failure",
        "candidate_batch",
    )
    for key in candidate_tensor_keys:
        value = raw_batch.get(key)
        if isinstance(value, torch.Tensor) and value.dim() >= 1 and int(value.size(0)) == old_total_candidates:
            out[key] = value.index_select(0, keep_idx)

    if isinstance(raw_batch.get("membership"), torch.Tensor):
        membership = raw_batch["membership"].to(torch.long).view(-1)
        candidate_mask = membership >= 0
        membership_new = torch.full_like(membership, -1)
        if bool(candidate_mask.any().item()):
            old_membership = membership[candidate_mask]
            valid_old = (old_membership >= 0) & (old_membership < old_total_candidates)
            mapped_values = torch.full((old_membership.numel(),), -1, dtype=torch.long)
            if bool(valid_old.any().item()):
                mapped_values[valid_old] = remap[old_membership[valid_old]]
            membership_new[candidate_mask] = mapped_values
        out["membership"] = membership_new

    if isinstance(raw_batch.get("pool_membership"), torch.Tensor) and isinstance(
        raw_batch.get("pool_node_index"), torch.Tensor
    ):
        pool_membership = raw_batch["pool_membership"].to(torch.long).view(-1)
        pool_node_index = raw_batch["pool_node_index"].to(torch.long).view(-1)
        valid_old = (pool_membership >= 0) & (pool_membership < old_total_candidates)
        mapped_pool = torch.full((pool_membership.numel(),), -1, dtype=torch.long)
        if bool(valid_old.any().item()):
            mapped_pool[valid_old] = remap[pool_membership[valid_old]]
        keep_pool_mask = mapped_pool >= 0
        out["pool_membership"] = mapped_pool[keep_pool_mask]
        out["pool_node_index"] = pool_node_index[keep_pool_mask]

    counts_t = torch.tensor(keep_counts, dtype=torch.long)
    frontier_ptr_new = torch.zeros((n_frontiers + 1,), dtype=torch.long)
    if n_frontiers > 0:
        frontier_ptr_new[1:] = torch.cumsum(counts_t, dim=0)
    out["frontier_ptr"] = frontier_ptr_new
    out["candidate_batch"] = (
        torch.repeat_interleave(
            torch.arange(n_frontiers, dtype=torch.long),
            counts_t,
        )
        if n_frontiers > 0
        else torch.zeros((0,), dtype=torch.long)
    )

    reward_t = out.get("reward_target")
    if not isinstance(reward_t, torch.Tensor):
        reward_t = out.get("rewards")
    if isinstance(reward_t, torch.Tensor):
        reward_t = reward_t.to(torch.float32).view(-1)
        out["reward_target"] = reward_t
        out["rewards"] = reward_t

        oracle_indices: List[int] = []
        oracle_rewards: List[torch.Tensor] = []
        for i in range(n_frontiers):
            s = int(frontier_ptr_new[i].item())
            e = int(frontier_ptr_new[i + 1].item())
            if e <= s:
                oracle_indices.append(int(s))
                oracle_rewards.append(reward_t.new_tensor(0.0))
                continue
            local_best = int(torch.argmax(reward_t[s:e]).item())
            best_global = int(s + local_best)
            oracle_indices.append(best_global)
            oracle_rewards.append(reward_t[best_global])
        out["oracle_index"] = torch.tensor(oracle_indices, dtype=torch.long)
        out["oracle_reward"] = (
            torch.stack(oracle_rewards)
            if oracle_rewards
            else reward_t.new_zeros((0,))
        )

    if isinstance(out.get("distance_raw"), torch.Tensor):
        out["distances"] = out["distance_raw"]
    elif isinstance(out.get("distances"), torch.Tensor):
        out["distance_raw"] = out["distances"]

    successor_ids = raw_batch.get("successor_ids")
    if isinstance(successor_ids, list):
        trimmed_successor_ids: list[list[str]] = []
        for i in range(n_frontiers):
            ids_i = successor_ids[i] if i < len(successor_ids) else []
            keep_n = int(keep_counts[i])
            if isinstance(ids_i, (list, tuple)):
                trimmed_successor_ids.append([str(x) for x in ids_i[:keep_n]])
            else:
                trimmed_successor_ids.append([])
        out["successor_ids"] = trimmed_successor_ids

    if any(
        key in raw_batch
        for key in (
            "frontier_mask",
            "padded_reward_targets",
            "padded_rewards",
            "padded_distances",
            "padded_is_failure",
            "padded_actions",
        )
    ):
        bsz = int(n_frontiers)
        max_n = int(max(keep_counts)) if keep_counts else 0
        reward_vec = out.get("reward_target", torch.zeros((0,), dtype=torch.float32)).to(torch.float32)
        distance_vec = out.get("distance_raw", torch.zeros((0,), dtype=torch.float32)).to(torch.float32)
        failure_vec = out.get("is_failure", torch.zeros((0,), dtype=torch.bool)).to(torch.bool)
        action_vec = out.get("action_map", torch.zeros((0,), dtype=torch.long)).to(torch.long)

        frontier_mask = torch.zeros((bsz, max_n), dtype=torch.bool)
        padded_rewards = torch.zeros((bsz, max_n), dtype=torch.float32)
        padded_distances = torch.zeros((bsz, max_n), dtype=torch.float32)
        padded_is_failure = torch.zeros((bsz, max_n), dtype=torch.bool)
        padded_actions = torch.full((bsz, max_n), -1, dtype=torch.long)
        for i in range(bsz):
            s = int(frontier_ptr_new[i].item())
            e = int(frontier_ptr_new[i + 1].item())
            n = int(max(0, e - s))
            if n <= 0:
                continue
            frontier_mask[i, :n] = True
            padded_rewards[i, :n] = reward_vec[s:e]
            padded_distances[i, :n] = distance_vec[s:e]
            padded_is_failure[i, :n] = failure_vec[s:e]
            padded_actions[i, :n] = action_vec[s:e]
        out["frontier_mask"] = frontier_mask
        out["padded_reward_targets"] = padded_rewards
        out["padded_rewards"] = padded_rewards
        out["padded_distances"] = padded_distances
        out["padded_is_failure"] = padded_is_failure
        out["padded_actions"] = padded_actions

    return out


def _run_single_frontier_size(
    args,
    frontier_size: int,
    all_frontier_sizes: Sequence[int],
    prepared_train_context: Optional[Dict[str, Any]] = None,
    prepared_eval_context: Optional[Dict[str, Any]] = None,
):
    seed_everything(args.seed)
    data_root, model_root = _paths(args)
    frontier_size = int(frontier_size)
    is_multi_size_run = len(all_frontier_sizes) > 1
    run_tag = f"onnx_frontier_size_{frontier_size}" if is_multi_size_run else ""
    if run_tag:
        print(f"[multi-frontier] frontier_size={frontier_size}")

    if prepared_train_context is None:
        prepared_train_context = _prepare_train_context(
            args=args,
            data_root=data_root,
            onnx_frontier_size=int(frontier_size),
            cache_namespace=run_tag,
        )

    train_samples = prepared_train_context["train_samples"]
    sample_params = prepared_train_context["sample_params"]
    sample_paths = prepared_train_context["sample_paths"]
    train_filter_stats = prepared_train_context["train_filter_stats"]
    train_jaccard_prune_stats = prepared_train_context["train_jaccard_prune_stats"]
    train_loader = prepared_train_context["train_loader"]
    base_onnx_frontier_size = int(prepared_train_context["base_onnx_frontier_size"])

    model_metrics_root = model_root / "metrics"
    model_metrics_root.mkdir(parents=True, exist_ok=True)
    metrics_root = (
        model_metrics_root / str(run_tag)
        if is_multi_size_run
        else model_metrics_root
    )
    metrics_root.mkdir(parents=True, exist_ok=True)
    train_metrics_dir = metrics_root / "train"
    train_metrics_dir.mkdir(parents=True, exist_ok=True)
    train_size_dist_plot_path = train_metrics_dir / "frontier_size_distribution_by_strategy.png"
    train_size_dist_json_path = train_metrics_dir / "frontier_size_distribution_by_strategy.json"

    train_size_strategy_distribution = _plot_train_frontier_size_label_distribution(
        train_samples=train_samples,
        out_path=train_size_dist_plot_path,
        frontier_size_cap=int(frontier_size),
    )
    with train_size_dist_json_path.open("w", encoding="utf-8") as fh:
        json.dump(train_size_strategy_distribution, fh, indent=2)
    train_state_stats_overall, train_state_stats_by_dataset = _build_train_state_stats(
        train_samples,
        frontier_size_cap=int(frontier_size),
    )
    print(
        "[dataset-filter] train frontiers | no_failure="
        f"{train_filter_stats['n_no_failure']} | with_failure_total="
        f"{train_filter_stats['n_with_failure_total']} | with_failure_kept="
        f"{train_filter_stats['n_with_failure_kept']} | with_failure_and_solution_total="
        f"{train_filter_stats['n_with_failure_and_solution_total']} | "
        "with_failure_and_solution_kept="
        f"{train_filter_stats['n_with_failure_and_solution_kept']} | total_kept="
        f"{train_filter_stats['n_total_kept']}"
    )
    print(
        "[dataset-prune] train near-duplicates | threshold="
        f"{float(args.train_frontier_jaccard_threshold):.3f} | total="
        f"{train_jaccard_prune_stats['n_total']} | kept="
        f"{train_jaccard_prune_stats['n_kept']} | dropped="
        f"{train_jaccard_prune_stats['n_dropped']} | missing_successor_ids="
        f"{train_jaccard_prune_stats['n_missing_successor_ids']}"
    )
    print(
        "[dataset-filter] train states | failure/all="
        f"{train_state_stats_overall['train_n_failure_states']}/"
        f"{train_state_stats_overall['train_n_all_states']}"
    )
    if int(base_onnx_frontier_size) != int(frontier_size):
        print(
            "[multi-frontier] active frontier size="
            f"{int(frontier_size)} (base train data built at {int(base_onnx_frontier_size)})."
        )

    if not train_samples:
        raise ValueError(
            f"No training frontiers loaded from {sample_paths['train']} after filtering."
        )

    train_frontier_definitions_path = train_metrics_dir / "frontier_definitions_for_dataloader.pt"
    train_frontier_definitions_meta_path = (
        train_metrics_dir / "frontier_definitions_for_dataloader_meta.json"
    )
    train_frontier_definitions = _build_train_frontier_definitions_for_dataloader(
        train_samples,
        frontier_size_cap=int(frontier_size),
    )
    torch.save(train_frontier_definitions, train_frontier_definitions_path)
    with train_frontier_definitions_meta_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "n_frontiers": int(len(train_frontier_definitions)),
                "source_train_samples_path": sample_paths["train"].as_posix(),
                "source_samples_params_path": sample_paths["params"].as_posix(),
                "base_onnx_frontier_size_for_train_data": int(base_onnx_frontier_size),
                "effective_frontier_size_for_training": int(frontier_size),
                "max_failure_states_per_dataset": float(args.max_failure_states_per_dataset),
                "train_frontier_jaccard_threshold": float(args.train_frontier_jaccard_threshold),
                "train_frontier_jaccard_pruning": train_jaccard_prune_stats,
                "seed": int(args.seed),
                "batch_size": int(args.batch_size),
                "num_workers": 0,
            },
            fh,
            indent=2,
        )
    loader_generator = getattr(train_loader, "generator", None)
    if isinstance(loader_generator, torch.Generator):
        loader_generator.manual_seed(int(args.seed))
    # Seed eval dataloaders (if any) for reproducibility
    if prepared_eval_context is not None:
        eval_loaders = dict(prepared_eval_context.get("eval_loaders") or {})
        for _name, loader in eval_loaders.items():
            if loader is None:
                continue
            loader_generator = getattr(loader, "generator", None)
            if isinstance(loader_generator, torch.Generator):
                loader_generator.manual_seed(int(args.seed))

    query_bundle = {}
    query_bundle_path: Optional[Path] = None
    test_data_root = _resolve_test_data_root(args=args, model_root=model_root)
    flat_query_splits: Dict[str, list[Dict[str, Any]]] = {}
    base_onnx_frontier_size_for_eval = int(base_onnx_frontier_size)
    if bool(args.evaluate):
        if prepared_eval_context is None:
            prepared_eval_context = _prepare_eval_context(
                args=args,
                data_root=data_root,
                model_root=model_root,
                onnx_frontier_size=int(base_onnx_frontier_size),
                cache_namespace=run_tag,
            )
        query_bundle = dict(prepared_eval_context.get("query_bundle") or {})
        raw_query_path = prepared_eval_context.get("query_bundle_path")
        query_bundle_path = Path(raw_query_path) if raw_query_path is not None else None
        raw_test_root = prepared_eval_context.get("test_data_root")
        if raw_test_root is not None:
            test_data_root = Path(raw_test_root)
        flat_query_splits = {
            str(name): list(values)
            for name, values in dict(prepared_eval_context.get("flat_query_splits") or {}).items()
        }
        base_onnx_frontier_size_for_eval = int(
            prepared_eval_context.get("base_onnx_frontier_size", base_onnx_frontier_size)
        )
        counts = _query_counts(flat_query_splits)
        print("[queries] built/loaded query counts:")
        for split_name in sorted(counts.keys()):
            print(f"  - {split_name}: {counts[split_name]}")
        if int(base_onnx_frontier_size_for_eval) != int(frontier_size):
            print(
                "[multi-frontier] active frontier size="
                f"{int(frontier_size)} (base eval data built at {int(base_onnx_frontier_size_for_eval)})."
            )
        if all(v == 0 for v in counts.values()):
            print(
                "[warning] Evaluation is enabled but all query splits are empty. "
                "Training-time/final evaluation will be skipped."
            )

    dataset_type_norm = str(args.dataset_type).upper()
    node_input_dim = 1 if dataset_type_norm == "HASHED" else int(train_samples[0]["node_features"].size(1))
    use_goal_separate_input = args.kind_of_data == "separated"
    inferred_num_edge_labels = _infer_num_edge_labels(train_samples)
    num_edge_labels = int(args.edge_label_buckets)
    if num_edge_labels <= 0:
        raise ValueError("--K/--edge-label-buckets must be > 0.")

    print(
        "[model] edge labels | inferred_train="
        f"{inferred_num_edge_labels} | K={num_edge_labels}"
    )
    print(
        "[model] edge-id bucketing: bucket = abs(edge_id) % K "
        "(applied in-model and exported to ONNX)."
    )
    if (not bool(args.train)):
        print(
            "[warning] --train is false: --K/--edge-label-buckets only affects "
            "newly trained checkpoints/exports. Existing checkpoints keep "
            "their saved architecture."
        )

    model = FrontierPolicyNetwork(
        node_input_dim=node_input_dim,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        conv_type=args.conv_type,
        pooling_type=args.pooling_type,
        dataset_type=args.dataset_type,
        edge_emb_dim=args.edge_emb_dim,
        num_edge_labels=num_edge_labels,
        num_node_labels=args.num_node_labels,
        use_global_context=args.use_global_context,
        mlp_depth=args.mlp_depth,
        use_goal_separate_input=use_goal_separate_input,
    )
    trainer = RLFrontierTrainer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        reward_formulation=args.reward_formulation,
        kind_of_data=args.kind_of_data,
        max_grad_norm=args.max_grad_norm,
    )

    eval_onnx_root = metrics_root / "eval_onnx"
    eval_onnx_root.mkdir(parents=True, exist_ok=True)
    onnx_reports_dir = metrics_root / "onnx"
    onnx_reports_dir.mkdir(parents=True, exist_ok=True)
    eval_step_reports_dir = metrics_root / "eval_reports"
    eval_step_reports_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = model_metrics_root / f"best_{int(frontier_size)}.pt"
    model_ckpt = model_metrics_root / f"{args.model_name}_{int(frontier_size)}.pt"
    onnx_path = model_root / f"{args.model_name}_{int(frontier_size)}.onnx"

    eval_trackers: Dict[str, EvalSplitTracker] = {}
    if bool(args.evaluate) and flat_query_splits:
        eval_trackers = _init_eval_trackers(
            trainer=trainer,
            eval_onnx_root=eval_onnx_root,
            query_splits=flat_query_splits,
        )

    history: Dict[str, Any] = {
        "train_epochs": [],
        "train_iqm_reward": [],
        "train_iqr_std_reward": [],
        "train_mean_reward": [],
        "train_std_reward": [],
        "train_mean_loss": [],
        "eval_epochs": [],
        "eval_selection_regret": [],
        "eval_selection_reward": [],
        "eval_selection_split": [],
    }

    best_eval_regret = float("inf")
    best_eval_reward: Optional[float] = None
    best_eval_epoch: Optional[int] = None
    best_eval_split: Optional[str] = None
    best_metrics_by_split: Dict[str, Any] = {}
    no_improve_eval_steps = 0
    eval_step = 0
    selection_order = [
        "test",
        "train",
    ]

    if bool(args.train):
        epoch_pbar = tqdm(range(int(args.n_train_epochs)), desc="training", leave=True)
        for epoch in epoch_pbar:
            trainer.model.train()
            epoch_batch_rewards: list[float] = []
            epoch_batch_losses: list[float] = []

            for raw_batch in train_loader:
                adapted_raw_batch = _adapt_train_batch_to_frontier_size(
                    raw_batch=raw_batch,
                    frontier_size=int(frontier_size),
                )
                batch = trainer._move_to_device(adapted_raw_batch)
                trainer.optimizer.zero_grad()
                loss_terms = trainer.compute_loss(batch)
                loss = loss_terms["total_loss"]
                loss.backward()
                if trainer.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.max_grad_norm)
                trainer.optimizer.step()

                epoch_batch_rewards.append(float(loss_terms["expected_reward"].detach().item()))
                epoch_batch_losses.append(float(loss_terms["total_loss"].detach().item()))

            train_stats = trainer._iqm_iqr_stats(epoch_batch_rewards)
            history["train_epochs"].append(epoch + 1)
            history["train_iqm_reward"].append(float(train_stats["iqm"]))
            history["train_iqr_std_reward"].append(float(train_stats["iqr_std"]))
            history["train_mean_reward"].append(float(train_stats["mean"]))
            history["train_std_reward"].append(float(train_stats["std"]))
            history["train_mean_loss"].append(trainer._safe_mean(epoch_batch_losses))

            trainer._plot_iqm_with_iqr_std_band(
                out_path=train_metrics_dir / "reward_iqm_iqr_std_over_epochs.png",
                x=history["train_epochs"],
                iqm=history["train_iqm_reward"],
                iqr_std=history["train_iqr_std_reward"],
                xlabel="Epoch",
                ylabel="Train Reward",
                y_lim=(-1.0, 0.0),
                title="Train Reward IQM ± IQR-STD",
            )

            should_eval = (
                bool(args.evaluate)
                and bool(flat_query_splits)
                and ((epoch + 1) % int(args.eval_every) == 0)
            )
            if should_eval:
                eval_step += 1
                trainer.to_onnx(
                    onnx_path,
                    node_input_dim=node_input_dim,
                    onnx_frontier_size=int(frontier_size),
                )
                step_summary_path = (
                    eval_step_reports_dir / f"step_{eval_step:04d}_epoch_{epoch + 1:04d}.json"
                )
                step_example_path = (
                    onnx_reports_dir
                    / f"{args.model_name}_onnx_first_example_step_{eval_step:04d}.json"
                )
                step_metrics = _evaluate_all_splits_with_onnx(
                    trainer=trainer,
                    onnx_path=onnx_path,
                    query_splits=flat_query_splits,
                    kind_of_data=args.kind_of_data,
                    dataset_type=args.dataset_type,
                    max_regular_distance_for_reward=float(args.max_regular_distance_for_reward),
                    failure_reward_value=float(args.failure_reward_value),
                    frontier_size_cap=int(frontier_size),
                    trackers=eval_trackers,
                    epoch=epoch + 1,
                    eval_step=eval_step,
                    summary_path=step_summary_path,
                    example_report_path=step_example_path,
                )
                eval_regret, eval_split = _select_eval_score(
                    eval_metrics=step_metrics,
                    preferred_order=selection_order,
                )
                eval_reward = (
                    float(step_metrics.get(str(eval_split), {}).get("mean_reward", 0.0))
                    if eval_split is not None
                    else 0.0
                )
                history["eval_epochs"].append(epoch + 1)
                history["eval_selection_regret"].append(float(eval_regret))
                history["eval_selection_reward"].append(float(eval_reward))
                history["eval_selection_split"].append(str(eval_split or ""))

                if float(eval_regret) < float(best_eval_regret):
                    best_eval_regret = float(eval_regret)
                    best_eval_reward = float(eval_reward)
                    best_eval_epoch = int(epoch + 1)
                    best_eval_split = str(eval_split) if eval_split else None
                    best_metrics_by_split = dict(step_metrics)
                    no_improve_eval_steps = 0
                    trainer.save_model(
                        best_ckpt,
                        metrics={
                            "best_eval_epoch": best_eval_epoch,
                            "best_eval_regret": best_eval_regret,
                            "best_eval_reward": best_eval_reward,
                            "best_eval_split": best_eval_split,
                        },
                    )
                    trainer.save_model(
                        model_ckpt,
                        metrics={
                            "best_eval_epoch": best_eval_epoch,
                            "best_eval_regret": best_eval_regret,
                            "best_eval_reward": best_eval_reward,
                            "best_eval_split": best_eval_split,
                        },
                    )
                    _save_best_model_performance(
                        out_path=metrics_root / "best_model_performance.json",
                        best_eval_epoch=best_eval_epoch,
                        best_eval_regret=best_eval_regret,
                        best_eval_reward=best_eval_reward,
                        best_eval_split=best_eval_split,
                        best_metrics_by_split=best_metrics_by_split,
                    )
                else:
                    no_improve_eval_steps += 1

                epoch_pbar.set_postfix(
                    train_iqm=f"{float(train_stats['iqm']):.4f}",
                    eval_regret=f"{float(eval_regret):.4f}",
                    eval_split=str(eval_split or "n/a"),
                )

                if (
                    int(args.early_stopping_patience_evals) > 0
                    and no_improve_eval_steps >= int(args.early_stopping_patience_evals)
                ):
                    print(
                        "Early stopping: "
                        f"no eval-regret improvement for {no_improve_eval_steps} checkpoints."
                    )
                    break
            else:
                epoch_pbar.set_postfix(
                    train_iqm=f"{float(train_stats['iqm']):.4f}",
                    train_loss=f"{trainer._safe_mean(epoch_batch_losses):.4f}",
                )

        with (train_metrics_dir / "history.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "epochs": history["train_epochs"],
                    "iqm_reward": history["train_iqm_reward"],
                    "iqr_std_reward": history["train_iqr_std_reward"],
                    "mean_reward": history["train_mean_reward"],
                    "std_reward": history["train_std_reward"],
                    "mean_loss": history["train_mean_loss"],
                },
                fh,
                indent=2,
            )

        if best_eval_epoch is None:
            trainer.save_model(
                best_ckpt,
                metrics={
                    "best_eval_epoch": None,
                    "best_eval_regret": None,
                    "best_eval_reward": None,
                    "best_eval_split": None,
                    "note": "No ONNX evaluation checkpoint selected; saved final training weights.",
                },
            )
            trainer.save_model(
                model_ckpt,
                metrics={
                    "best_eval_epoch": None,
                    "best_eval_regret": None,
                    "best_eval_reward": None,
                    "best_eval_split": None,
                    "note": "No ONNX evaluation checkpoint selected; saved final training weights.",
                },
            )
            _save_best_model_performance(
                out_path=metrics_root / "best_model_performance.json",
                best_eval_epoch=None,
                best_eval_regret=None,
                best_eval_reward=None,
                best_eval_split=None,
                best_metrics_by_split={},
            )

    load_ckpt = _find_checkpoint_to_load(best_ckpt=best_ckpt, model_ckpt=model_ckpt)
    if load_ckpt is None:
        raise FileNotFoundError(
            f"Missing checkpoint: {best_ckpt} or {model_ckpt}. "
            "Run with --train true first."
        )

    loaded = RLFrontierTrainer.load_model(load_ckpt, device=trainer.device)
    trainer.model = loaded.to(trainer.device)

    must_export_onnx = bool(args.export_onnx) or bool(args.evaluate)
    if must_export_onnx:
        trainer.to_onnx(
            onnx_path,
            node_input_dim=node_input_dim,
            onnx_frontier_size=int(frontier_size),
        )

    final_eval_metrics: Dict[str, Dict[str, Any]] = {}
    if bool(args.evaluate):
        if flat_query_splits:
            final_summary_path = eval_onnx_root / "final_summary.json"
            final_example_path = (
                onnx_reports_dir / f"{args.model_name}_onnx_first_example_check_final.json"
            )
            final_eval_metrics = _evaluate_all_splits_with_onnx(
                trainer=trainer,
                onnx_path=onnx_path,
                query_splits=flat_query_splits,
                kind_of_data=args.kind_of_data,
                dataset_type=args.dataset_type,
                max_regular_distance_for_reward=float(args.max_regular_distance_for_reward),
                failure_reward_value=float(args.failure_reward_value),
                frontier_size_cap=int(frontier_size),
                trackers=None,
                epoch=best_eval_epoch,
                eval_step=None,
                summary_path=final_summary_path,
                example_report_path=final_example_path,
            )
            for split_name, metrics in final_eval_metrics.items():
                split_dir = eval_onnx_root / split_name
                split_dir.mkdir(parents=True, exist_ok=True)
                with (split_dir / "final_metrics.json").open("w", encoding="utf-8") as fh:
                    json.dump(metrics, fh, indent=2)
                query_label_metrics = metrics.get("query_label_metrics", {})
                if isinstance(query_label_metrics, dict) and query_label_metrics:
                    _plot_query_label_metric_curves(
                        out_path=split_dir / "query_label_reward_by_frontier_size_final.png",
                        split_name=str(split_name),
                        metric_title="Reward (IQM)",
                        ylabel="Reward",
                        query_label_metrics=query_label_metrics,
                        curve_key="reward_by_size",
                        y_lim=(-1.0, 0.05),
                    )
                    _plot_query_label_metric_curves(
                        out_path=split_dir / "query_label_accuracy_by_frontier_size_final.png",
                        split_name=str(split_name),
                        metric_title="Accuracy (IQM)",
                        ylabel="Accuracy",
                        query_label_metrics=query_label_metrics,
                        curve_key="accuracy_by_size",
                        y_lim=(-0.05, 1.05),
                    )
                    _plot_query_label_metric_curves(
                        out_path=split_dir / "query_label_regret_by_frontier_size_final.png",
                        split_name=str(split_name),
                        metric_title="Regret |best reward - taken reward| (IQM)",
                        ylabel="Regret",
                        query_label_metrics=query_label_metrics,
                        curve_key="regret_by_size",
                        y_lim=(0.0, 1.05),
                    )
        else:
            print(
                "[warning] Evaluation requested but query splits are empty. "
                "Skipping final ONNX evaluation."
            )

    split_summary = dict(sample_params.get("split_summary") or {})
    split_summary["num_frontiers_train"] = int(len(train_samples))
    split_summary["train_no_failure_frontiers"] = int(train_filter_stats["n_no_failure"])
    split_summary["train_with_failure_frontiers_total"] = int(
        train_filter_stats["n_with_failure_total"]
    )
    split_summary["train_with_failure_frontiers_kept"] = int(
        train_filter_stats["n_with_failure_kept"]
    )
    split_summary["train_with_failure_and_solution_frontiers_total"] = int(
        train_filter_stats["n_with_failure_and_solution_total"]
    )
    split_summary["train_with_failure_and_solution_frontiers_kept"] = int(
        train_filter_stats["n_with_failure_and_solution_kept"]
    )
    split_summary["train_with_failure_and_solution_frontiers_after_filter"] = int(
        train_state_stats_overall["train_with_failure_and_solution_frontiers_after_filter"]
    )
    split_summary["train_n_failure_states"] = int(
        train_state_stats_overall["train_n_failure_states"]
    )
    split_summary["train_n_all_states"] = int(train_state_stats_overall["train_n_all_states"])
    split_summary["train_failure_states_over_all_states"] = (
        f"{train_state_stats_overall['train_n_failure_states']}/"
        f"{train_state_stats_overall['train_n_all_states']}"
    )
    split_summary["train_failure_state_rate"] = (
        float(
            train_state_stats_overall["train_n_failure_states"]
            / train_state_stats_overall["train_n_all_states"]
        )
        if int(train_state_stats_overall["train_n_all_states"]) > 0
        else 0.0
    )
    split_summary["max_failure_states_per_dataset"] = float(args.max_failure_states_per_dataset)
    split_summary["train_frontier_jaccard_threshold"] = float(args.train_frontier_jaccard_threshold)
    split_summary["train_frontier_jaccard_pruning"] = train_jaccard_prune_stats
    split_summary["eval_frontier_jaccard_threshold"] = float(args.eval_frontier_jaccard_threshold)
    split_summary["onnx_frontier_size"] = int(frontier_size)
    split_summary["base_onnx_frontier_size_for_train_data"] = int(base_onnx_frontier_size)
    split_summary["base_onnx_frontier_size_for_eval_data"] = int(base_onnx_frontier_size_for_eval)
    split_summary["train_random_frontier_ratio"] = float(args.train_random_frontier_ratio)
    split_summary["train_random_frontier_with_failure_ratio"] = float(
        args.train_random_frontier_with_failure_ratio
    )
    split_summary["train_frontier_size_distribution"] = train_size_strategy_distribution
    split_summary["train_frontier_size_distribution_plot"] = train_size_dist_plot_path.as_posix()
    split_summary["train_frontier_size_distribution_json"] = train_size_dist_json_path.as_posix()
    split_summary["train_frontier_definitions_for_dataloader"] = (
        train_frontier_definitions_path.as_posix()
    )
    split_summary["train_frontier_definitions_for_dataloader_meta"] = (
        train_frontier_definitions_meta_path.as_posix()
    )
    split_summary["query_counts"] = _query_counts(flat_query_splits)
    split_summary["query_bundle_path"] = (
        query_bundle_path.as_posix() if query_bundle_path is not None else ""
    )
    split_summary["test_data_root"] = test_data_root.as_posix()
    split_summary["train_dataset_state_summaries"] = [
        {
            "dataset_id": dataset_id,
            "train_frontiers_after_filter": int(stats["train_frontiers_after_filter"]),
            "train_with_failure_and_solution_frontiers_after_filter": int(
                stats["train_with_failure_and_solution_frontiers_after_filter"]
            ),
            "train_n_failure_states": int(stats["train_n_failure_states"]),
            "train_n_all_states": int(stats["train_n_all_states"]),
            "train_failure_states_over_all_states": (
                f"{stats['train_n_failure_states']}/{stats['train_n_all_states']}"
            ),
        }
        for dataset_id, stats in sorted(train_state_stats_by_dataset.items())
    ]
    with (metrics_root / "dataset_split_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(split_summary, fh, indent=2)

    with (model_root / f"{args.model_name}_info_{int(frontier_size)}.txt").open("w", encoding="utf-8") as fh:
        for key, value in vars(args).items():
            fh.write(f"{key} = {value}\n")
        fh.write(f"effective_onnx_frontier_size = {int(frontier_size)}\n")
        fh.write(f"base_onnx_frontier_size_for_train_data = {int(base_onnx_frontier_size)}\n")
        fh.write(f"base_onnx_frontier_size_for_eval_data = {int(base_onnx_frontier_size_for_eval)}\n")
        fh.write(f"inferred_num_edge_labels = {inferred_num_edge_labels}\n")
        fh.write(f"effective_num_edge_labels = {num_edge_labels}\n")
        fh.write("edge_id_bucket_mapping = abs(edge_id) % effective_num_edge_labels\n")
        fh.write(f"samples_train_path = {sample_paths['train']}\n")
        fh.write(f"samples_params_path = {sample_paths['params']}\n")
        fh.write(f"query_bundle_path = {query_bundle_path.as_posix() if query_bundle_path else ''}\n")
        fh.write(f"test_data_root = {test_data_root.as_posix()}\n")
        fh.write(f"n_train_samples = {len(train_samples)}\n")
        for split_name, count in _query_counts(flat_query_splits).items():
            fh.write(f"n_queries_{split_name} = {count}\n")
        fh.write(f"model_checkpoint_loaded = {load_ckpt.as_posix()}\n")
        fh.write(f"best_checkpoint = {best_ckpt.as_posix()}\n")
        fh.write(f"best_eval_epoch = {best_eval_epoch}\n")
        fh.write(f"best_eval_regret = {best_eval_regret if best_eval_epoch is not None else None}\n")
        fh.write(f"best_eval_reward = {best_eval_reward if best_eval_epoch is not None else None}\n")
        fh.write(f"best_eval_split = {best_eval_split}\n")
        fh.write(f"onnx_path = {onnx_path.as_posix()}\n")
        fh.write(f"final_eval_splits = {sorted(final_eval_metrics.keys())}\n")

    return onnx_path.as_posix()


def main(args):
    try:
        frontier_sizes = _parse_onnx_frontier_sizes(args.onnx_frontier_size)
    except ValueError as exc:
        raise ValueError(f"Invalid --onnx-frontier-size value: {exc}") from exc

    if len(frontier_sizes) == 1:
        args.onnx_frontier_size = int(frontier_sizes[0])
        return _run_single_frontier_size(
            args=args,
            frontier_size=int(frontier_sizes[0]),
            all_frontier_sizes=frontier_sizes,
        )

    max_frontier_size = int(max(frontier_sizes))
    data_root, _ = _paths(args)
    shared_args = argparse.Namespace(**vars(args))
    shared_args.onnx_frontier_size = int(max_frontier_size)
    shared_cache_namespace = f"onnx_frontier_size_{int(max_frontier_size)}_shared"
    shared_train_context = _prepare_train_context(
        args=shared_args,
        data_root=data_root,
        onnx_frontier_size=int(max_frontier_size),
        cache_namespace=shared_cache_namespace,
    )
    shared_eval_context: Optional[Dict[str, Any]] = None
    if bool(args.evaluate):
        _, model_root = _paths(args)
        shared_eval_context = _prepare_eval_context(
            args=shared_args,
            data_root=data_root,
            model_root=model_root,
            onnx_frontier_size=int(max_frontier_size),
            cache_namespace=shared_cache_namespace,
        )

    run_summaries: list[Dict[str, Any]] = []
    for frontier_size in frontier_sizes:
        run_args = argparse.Namespace(**vars(args))
        run_args.onnx_frontier_size = int(frontier_size)
        run_onnx_path = _run_single_frontier_size(
            args=run_args,
            frontier_size=int(frontier_size),
            all_frontier_sizes=frontier_sizes,
            prepared_train_context=shared_train_context,
            prepared_eval_context=shared_eval_context,
        )
        run_summaries.append(
            {
                "onnx_frontier_size": int(frontier_size),
                "model_name": f"{str(run_args.model_name)}_{int(frontier_size)}",
                "onnx_path": str(run_onnx_path),
            }
        )

    _, model_root = _paths(args)
    metrics_root = model_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    multi_summary_path = metrics_root / "multi_frontier_training_summary.json"
    with multi_summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "onnx_frontier_sizes": [int(v) for v in frontier_sizes],
                "base_onnx_frontier_size_for_train_data": int(max_frontier_size),
                "base_onnx_frontier_size_for_eval_data": int(max_frontier_size),
                "single_dataloader_reused_across_frontier_sizes": True,
                "single_eval_bundle_reused_across_frontier_sizes": bool(args.evaluate),
                "runs": run_summaries,
            },
            fh,
            indent=2,
        )
    print(f"[multi-frontier] Summary saved to: {multi_summary_path.as_posix()}")
    return multi_summary_path.as_posix()


if __name__ == "__main__":
    main(parse_args())

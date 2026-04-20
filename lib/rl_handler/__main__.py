from __future__ import annotations

import argparse
import json
import networkx as nx
import pydot
import re
import torch
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence, Tuple

from src.data import (
    build_frontier_dataloader,
    build_frontier_samples,
    build_random_eval_frontiers_for_dataset,
    build_stress_eval_frontiers_for_dataset,
    group_clean_frontiers,
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

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "y", "true", "t"):
        return True
    if v in ("no", "n", "false", "f"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def ratio_0_1(v):
    try:
        ratio = float(v)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Expected a float value.") from exc
    if ratio < 0.0 or ratio > 1.0:
        raise argparse.ArgumentTypeError("Expected a value in [0.0, 1.0].")
    return ratio


def _node_feature_dtypes_for_dataset(dataset_type: str) -> tuple[torch.dtype, str]:
    if str(dataset_type).upper() == "BITMASK":
        return torch.float32, "float32"
    return torch.int64, "int64"


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
        default=1000,
        help="Maximum number of generated random/fifo/stress queries per dataset.",
    )
    parser.add_argument(
        "--max-size-frontier",
        type=int,
        default=32,
        help="Maximum frontier size used by stress-query generation.",
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

    parser.add_argument("--batch-size", type=int, default=9092)
    parser.add_argument("--n-train-epochs", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
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
            "If true, rebuild query definitions for ONNX evaluation from train/test roots; "
            "otherwise load the saved query bundle."
        ),
    )
    parser.add_argument("--train", type=str2bool, default=True)
    parser.add_argument("--evaluate", type=str2bool, default=False)
    parser.add_argument("--export-onnx", type=str2bool, default=True)

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


def _dataset_tag(dataset_id: str) -> str:
    return (
        str(dataset_id)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )


def _build_or_load_train_samples(args, data_root: Path):
    data_dir = data_root / "processed_data"
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

    if (not bool(args.build_data)) and (not train_path.exists()):
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
            build_eval_data=False,
        )
        torch.save(list(train_samples), train_path)
        torch.save(dict(params or {}), params_path)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing training samples file: {train_path}. "
            "Run with --build-data true."
        )

    train_samples = torch.load(train_path, weights_only=False)
    params = torch.load(params_path, weights_only=False) if params_path.exists() else {}
    return list(train_samples), dict(params or {}), {"train": train_path, "params": params_path}


def _collect_clean_frontier_entries(
    root: Path,
    kind_of_data: str,
    subset_filter: Optional[set[str]],
    max_regular_distance_for_reward: float,
    failure_reward_value: float,
) -> list[Dict[str, Any]]:
    if not root.exists() or not root.is_dir():
        return []

    entries: list[Dict[str, Any]] = []
    for prob_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if subset_filter and prob_dir.name not in subset_filter:
            continue
        for csv_path in sorted(p for p in prob_dir.iterdir() if p.suffix.lower() == ".csv"):
            rows = read_frontier_csv(csv_path=csv_path, kind_of_data=kind_of_data)
            clean_frontiers = group_clean_frontiers(
                rows=rows,
                kind_of_data=kind_of_data,
                max_regular_distance_for_reward=float(max_regular_distance_for_reward),
                failure_reward_value=float(failure_reward_value),
            )
            dataset_id = f"{prob_dir.name}/{csv_path.name}"
            for frontier in clean_frontiers:
                frontier["dataset_id"] = dataset_id
            entries.append(
                {
                    "problem_name": str(prob_dir.name),
                    "dataset_id": str(dataset_id),
                    "csv_name": str(csv_path.name),
                    "csv_stem": str(csv_path.stem),
                    "n_rows": int(len(rows)),
                    "n_clean_frontiers": int(len(clean_frontiers)),
                    "clean_frontiers": clean_frontiers,
                }
            )
    return entries


def _attach_query_metadata(
    frontiers: Sequence[Dict[str, Any]],
    dataset_id: str,
    source_tag: str,
    query_type: str,
) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for frontier in frontiers:
        item = dict(frontier)
        item["dataset_id"] = str(dataset_id)
        item["source_tag"] = str(source_tag)
        item["query_type"] = str(query_type)
        if query_type == "stress":
            item["stress_schedule"] = "stress"
        out.append(item)
    return out


def _build_query_splits_from_entries(
    entries: Sequence[Dict[str, Any]],
    kind_of_data: str,
    n_max_dataset_queries: int,
    max_size_frontier: int,
    seed: int,
    source_tag: str,
    failure_reward_value: float,
) -> Dict[str, list[Dict[str, Any]]]:
    splits: Dict[str, list[Dict[str, Any]]] = {
        "random": [],
        "fifo": [],
        "stress": [],
    }
    for dataset_idx, entry in enumerate(entries):
        clean_frontiers = list(entry.get("clean_frontiers", []))
        if not clean_frontiers:
            continue

        dataset_id = str(entry["dataset_id"])
        dataset_tag = _dataset_tag(dataset_id)
        dataset_seed = int(seed + 104729 * dataset_idx + 17)

        random_frontiers = build_random_eval_frontiers_for_dataset(
            clean_frontiers=clean_frontiers,
            kind_of_data=kind_of_data,
            n_max_dataset_queries=int(n_max_dataset_queries),
            seed=dataset_seed,
            dataset_tag=dataset_tag,
            failure_reward_value=float(failure_reward_value),
        )
        fifo_frontiers, lifo_frontiers = build_stress_eval_frontiers_for_dataset(
            clean_frontiers=clean_frontiers,
            kind_of_data=kind_of_data,
            n_max_dataset_queries=int(n_max_dataset_queries),
            max_size_frontier=int(max_size_frontier),
            dataset_tag=dataset_tag,
            failure_reward_value=float(failure_reward_value),
        )

        splits["random"].extend(
            _attach_query_metadata(
                random_frontiers,
                dataset_id=dataset_id,
                source_tag=source_tag,
                query_type="random",
            )
        )
        splits["fifo"].extend(
            _attach_query_metadata(
                fifo_frontiers,
                dataset_id=dataset_id,
                source_tag=source_tag,
                query_type="fifo",
            )
        )
        # "stress" maps to the second stress schedule (LIFO-based windowing).
        splits["stress"].extend(
            _attach_query_metadata(
                lifo_frontiers,
                dataset_id=dataset_id,
                source_tag=source_tag,
                query_type="stress",
            )
        )
    return splits


def _resolve_test_data_root(args, model_root: Path) -> Path:
    if str(args.folder_test_data).strip():
        return Path(str(args.folder_test_data).strip())
    return model_root / "test_data"


def _build_or_load_query_bundle(args, data_root: Path, model_root: Path):
    data_dir = data_root / "processed_data"
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
        return bundle, query_path, test_data_root

    train_entries = _collect_clean_frontier_entries(
        root=Path(args.folder_raw_data),
        kind_of_data=args.kind_of_data,
        subset_filter=set(args.subset_train) if args.subset_train else None,
        max_regular_distance_for_reward=float(args.max_regular_distance_for_reward),
        failure_reward_value=float(args.failure_reward_value),
    )
    train_splits = _build_query_splits_from_entries(
        entries=train_entries,
        kind_of_data=args.kind_of_data,
        n_max_dataset_queries=int(args.n_max_dataset_queries),
        max_size_frontier=int(args.max_size_frontier),
        seed=int(args.seed),
        source_tag="train",
        failure_reward_value=float(args.failure_reward_value),
    )

    if test_data_root.exists() and test_data_root.is_dir():
        test_entries = _collect_clean_frontier_entries(
            root=test_data_root,
            kind_of_data=args.kind_of_data,
            subset_filter=None,
            max_regular_distance_for_reward=float(args.max_regular_distance_for_reward),
            failure_reward_value=float(args.failure_reward_value),
        )
        test_splits = _build_query_splits_from_entries(
            entries=test_entries,
            kind_of_data=args.kind_of_data,
            n_max_dataset_queries=int(args.n_max_dataset_queries),
            max_size_frontier=int(args.max_size_frontier),
            seed=int(args.seed) + 73,
            source_tag="test",
            failure_reward_value=float(args.failure_reward_value),
        )
    else:
        test_entries = []
        test_splits = {"random": [], "fifo": [], "stress": []}

    bundle = {
        "meta": {
            "kind_of_data": str(args.kind_of_data),
            "dataset_type": str(args.dataset_type),
            "seed": int(args.seed),
            "n_max_dataset_queries": int(args.n_max_dataset_queries),
            "max_size_frontier": int(args.max_size_frontier),
            "folder_raw_data": str(args.folder_raw_data),
            "folder_test_data": test_data_root.as_posix(),
        },
        "train_entries": [
            {
                "dataset_id": e["dataset_id"],
                "problem_name": e["problem_name"],
                "csv_name": e["csv_name"],
                "n_rows": e["n_rows"],
                "n_clean_frontiers": e["n_clean_frontiers"],
            }
            for e in train_entries
        ],
        "test_entries": [
            {
                "dataset_id": e["dataset_id"],
                "problem_name": e["problem_name"],
                "csv_name": e["csv_name"],
                "n_rows": e["n_rows"],
                "n_clean_frontiers": e["n_clean_frontiers"],
            }
            for e in test_entries
        ],
        "train": train_splits,
        "test": test_splits,
    }
    with query_path.open("w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)
    return bundle, query_path, test_data_root


def _flatten_query_splits(bundle: Dict[str, Any]) -> Dict[str, list[Dict[str, Any]]]:
    out: Dict[str, list[Dict[str, Any]]] = {}
    for source in ("train", "test"):
        source_payload = bundle.get(source, {})
        if not isinstance(source_payload, dict):
            source_payload = {}
        for query_type in ("random", "fifo", "stress"):
            key = f"{source}_{query_type}"
            values = source_payload.get(query_type, [])
            out[key] = list(values) if isinstance(values, list) else []
    return out


def _query_counts(flat_query_splits: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, int]:
    return {split_name: int(len(frontiers)) for split_name, frontiers in flat_query_splits.items()}


def _sample_has_failure_state(sample: Dict[str, Any]) -> bool:
    has_failure = sample.get("frontier_has_failure")
    if has_failure is not None:
        return bool(has_failure)

    failure_vec = sample.get("is_failure")
    if isinstance(failure_vec, torch.Tensor):
        if failure_vec.numel() == 0:
            return False
        return bool(failure_vec.to(torch.bool).any().item())
    if isinstance(failure_vec, (list, tuple)):
        return any(bool(x) for x in failure_vec)
    return False


def _sample_has_failure_and_solution_state(sample: Dict[str, Any]) -> bool:
    failure_vec = sample.get("is_failure")
    if isinstance(failure_vec, torch.Tensor):
        if failure_vec.numel() == 0:
            return False
        failure_mask = failure_vec.to(torch.bool).view(-1)
        return bool(failure_mask.any().item() and (~failure_mask).any().item())
    if isinstance(failure_vec, (list, tuple)):
        if not failure_vec:
            return False
        has_failure = any(bool(x) for x in failure_vec)
        has_solution = any(not bool(x) for x in failure_vec)
        return bool(has_failure and has_solution)
    return False


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


def _count_sample_states(sample: Dict[str, Any]) -> Tuple[int, int]:
    is_failure = sample.get("is_failure")
    if isinstance(is_failure, torch.Tensor):
        all_states = int(is_failure.numel())
        failure_states = int(is_failure.to(torch.long).sum().item()) if all_states > 0 else 0
        return failure_states, all_states
    if isinstance(is_failure, (list, tuple)):
        all_states = int(len(is_failure))
        failure_states = int(sum(1 for x in is_failure if bool(x)))
        return failure_states, all_states

    reward_target = sample.get("reward_target", sample.get("rewards"))
    if isinstance(reward_target, torch.Tensor):
        return 0, int(reward_target.numel())
    if isinstance(reward_target, (list, tuple)):
        return 0, int(len(reward_target))
    return 0, 0


def _build_train_state_stats(
    train_samples: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    overall_failure_states = 0
    overall_all_states = 0
    overall_failure_and_solution_frontiers = 0
    per_dataset: Dict[str, Dict[str, int]] = {}

    for sample in train_samples:
        dataset_id = str(sample.get("dataset_id", ""))
        failure_states, all_states = _count_sample_states(sample)
        failure_and_solution_frontier = _sample_has_failure_and_solution_state(sample)
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


def _strip_quotes(v: Any) -> str:
    return str(v).replace('"', "").strip()


def _parse_numeric_node_label_eval(node_obj: object) -> int:
    if isinstance(node_obj, bool):
        raise TypeError("Boolean node labels are not supported.")
    if isinstance(node_obj, int):
        return int(node_obj)
    if isinstance(node_obj, float):
        if not float(node_obj).is_integer():
            raise ValueError(f"Node label '{node_obj}' is not an integer.")
        return int(node_obj)
    if isinstance(node_obj, str):
        s = node_obj.strip()
        try:
            return int(s)
        except ValueError:
            parsed = float(s)
            if not parsed.is_integer():
                raise ValueError(f"Node label '{node_obj}' is not an integer.")
            return int(parsed)
    raise TypeError(f"Unsupported node label type: {type(node_obj)}")


def _validate_int64_range_eval(value: int, *, context: str) -> int:
    if value < I64_MIN or value > I64_MAX:
        raise ValueError(f"{context} is out of int64 range [{I64_MIN}, {I64_MAX}].")
    return int(value)


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


def _infer_num_candidates(sample: Dict[str, Any]) -> int:
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
            return n_candidates

    membership = sample.get("membership")
    if isinstance(membership, torch.Tensor) and membership.numel() > 0:
        return int(membership.max().item()) + 1
    return 0


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


def _build_eval_metrics_from_frontier_decisions(
    trainer: RLFrontierTrainer,
    chosen_rewards: Sequence[float],
    accuracies: Sequence[int],
    frontier_has_failure: Sequence[int],
    frontier_sizes: Sequence[int],
    abs_reward_gaps: Sequence[float],
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

    metrics: Dict[str, object] = {
        "n_frontiers": len(chosen_rewards),
        "chosen_rewards": [float(x) for x in chosen_rewards],
        "accuracies": [int(x) for x in accuracies],
        "frontier_has_failure": [int(x) for x in frontier_has_failure],
        "frontier_sizes": [int(x) for x in frontier_sizes],
        "abs_reward_gaps": [float(x) for x in abs_reward_gaps],
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
        "oracle_accuracy": float(all_accuracy_stats["mean"]),
    }

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
    failed_frontiers: list[Dict[str, Any]] = []

    for idx, frontier in enumerate(
        tqdm(frontiers, desc=f"Evaluating {split_name}", leave=False)
    ):
        try:
            sample = _materialize_query_frontier_no_pyg(
                frontier=frontier,
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
                    "dataset_id": str(frontier.get("dataset_id", "")),
                    "predecessor_path": str(frontier.get("predecessor_path", "")),
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
                    "dataset_id": str(frontier.get("dataset_id", "")),
                    "predecessor_path": str(frontier.get("predecessor_path", "")),
                    "error": str(result.get("error", "unknown error")),
                }
            )
            continue

        chosen_rewards.append(float(result["chosen_reward"]))
        accuracies.append(int(result["accuracy"]))
        frontier_has_failure.append(int(result["frontier_has_failure"]))
        frontier_sizes.append(int(result["frontier_size"]))
        abs_reward_gaps.append(float(result["abs_reward_gap"]))

    metrics = _build_eval_metrics_from_frontier_decisions(
        trainer=trainer,
        chosen_rewards=chosen_rewards,
        accuracies=accuracies,
        frontier_has_failure=frontier_has_failure,
        frontier_sizes=frontier_sizes,
        abs_reward_gaps=abs_reward_gaps,
    )
    metrics["split_name"] = str(split_name)
    metrics["n_attempted_frontiers"] = int(len(frontiers))
    metrics["n_failed_frontiers"] = int(len(failed_frontiers))
    metrics["n_evaluated_frontiers"] = int(len(chosen_rewards))
    metrics["onnx_requires_goal_inputs"] = bool(onnx_ctx.use_goal_inputs)
    metrics["onnx_static_output_len"] = (
        int(onnx_ctx.static_onnx_len) if onnx_ctx.static_onnx_len is not None else None
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

    with (tracker.eval_dir / "history.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "epochs": tracker.history["epochs"],
                "iqm_reward": tracker.history["iqm_reward"],
                "iqr_std_reward": tracker.history["iqr_std_reward"],
                "iqm_abs_reward_gap": tracker.history["iqm_abs_reward_gap"],
                "iqr_std_abs_reward_gap": tracker.history["iqr_std_abs_reward_gap"],
                "regimes": tracker.regime_history,
            },
            fh,
            indent=2,
        )


def _select_first_query_for_example(
    query_splits: Dict[str, Sequence[Dict[str, Any]]],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    preferred_order = [
        "train_random",
        "train_fifo",
        "train_stress",
        "test_random",
        "test_fifo",
        "test_stress",
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

    try:
        sample = _materialize_query_frontier_no_pyg(
            frontier=query,
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
            "dataset_id": str(query.get("dataset_id", "")),
            "predecessor_path": str(query.get("predecessor_path", "")),
            "result": parity,
        }
    except Exception as exc:
        report = {
            "status": "error",
            "split_name": split_name,
            "dataset_id": str(query.get("dataset_id", "")),
            "predecessor_path": str(query.get("predecessor_path", "")),
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
        return float(metrics.get("mean_reward", 0.0)), split_name
    for split_name, metrics in eval_metrics.items():
        if int(metrics.get("n_evaluated_frontiers", 0)) <= 0:
            continue
        return float(metrics.get("mean_reward", 0.0)), split_name
    return -float("inf"), None


def _save_best_model_performance(
    out_path: Path,
    best_eval_epoch: Optional[int],
    best_eval_reward: Optional[float],
    best_eval_split: Optional[str],
    best_metrics_by_split: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "best_model_name": "best.pt",
                "selection_metric": "mean_reward",
                "best_eval_epoch": best_eval_epoch,
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


def main(args):
    seed_everything(args.seed)
    data_root, model_root = _paths(args)

    train_samples, sample_params, sample_paths = _build_or_load_train_samples(
        args=args,
        data_root=data_root,
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
    train_state_stats_overall, train_state_stats_by_dataset = _build_train_state_stats(
        train_samples
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
        "[dataset-filter] train states | failure/all="
        f"{train_state_stats_overall['train_n_failure_states']}/"
        f"{train_state_stats_overall['train_n_all_states']}"
    )

    if not train_samples:
        raise ValueError(
            f"No training frontiers loaded from {sample_paths['train']} after filtering."
        )

    train_loader = build_frontier_dataloader(
        samples=train_samples,
        batch_size=args.batch_size,
        seed=int(args.seed),
        num_workers=args.num_workers,
        pad_frontiers=True,
        shuffle=True,
    )

    query_bundle = {}
    query_bundle_path: Optional[Path] = None
    test_data_root = _resolve_test_data_root(args=args, model_root=model_root)
    flat_query_splits: Dict[str, list[Dict[str, Any]]] = {}
    if bool(args.evaluate):
        query_bundle, query_bundle_path, test_data_root = _build_or_load_query_bundle(
            args=args,
            data_root=data_root,
            model_root=model_root,
        )
        flat_query_splits = _flatten_query_splits(query_bundle)
        counts = _query_counts(flat_query_splits)
        print("[queries] built/loaded query counts:")
        for split_name in sorted(counts.keys()):
            print(f"  - {split_name}: {counts[split_name]}")
        if all(v == 0 for v in counts.values()):
            print(
                "[warning] Evaluation is enabled but all query splits are empty. "
                "Training-time/final evaluation will be skipped."
            )

    dataset_type_norm = str(args.dataset_type).upper()
    node_input_dim = 1 if dataset_type_norm == "HASHED" else int(train_samples[0]["node_features"].size(1))
    use_goal_separate_input = args.kind_of_data == "separated"
    num_edge_labels = _infer_num_edge_labels(train_samples)

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

    metrics_root = model_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    train_metrics_dir = metrics_root / "train"
    train_metrics_dir.mkdir(parents=True, exist_ok=True)
    eval_onnx_root = metrics_root / "eval_onnx"
    eval_onnx_root.mkdir(parents=True, exist_ok=True)
    onnx_reports_dir = metrics_root / "onnx"
    onnx_reports_dir.mkdir(parents=True, exist_ok=True)
    eval_step_reports_dir = metrics_root / "eval_reports"
    eval_step_reports_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = model_root / "best.pt"
    model_ckpt = metrics_root / f"{args.model_name}.pt"
    onnx_path = model_root / f"{args.model_name}.onnx"

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
        "eval_selection_score": [],
        "eval_selection_split": [],
    }

    best_eval_reward = -float("inf")
    best_eval_epoch: Optional[int] = None
    best_eval_split: Optional[str] = None
    best_metrics_by_split: Dict[str, Any] = {}
    no_improve_eval_steps = 0
    eval_step = 0
    selection_order = [
        "train_random",
        "test_random",
        "train_fifo",
        "train_stress",
        "test_fifo",
        "test_stress",
    ]

    if bool(args.train):
        epoch_pbar = tqdm(range(int(args.n_train_epochs)), desc="training", leave=True)
        for epoch in epoch_pbar:
            trainer.model.train()
            epoch_batch_rewards: list[float] = []
            epoch_batch_losses: list[float] = []

            for raw_batch in train_loader:
                batch = trainer._move_to_device(raw_batch)
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
                trainer.to_onnx(onnx_path, node_input_dim=node_input_dim)
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
                    trackers=eval_trackers,
                    epoch=epoch + 1,
                    eval_step=eval_step,
                    summary_path=step_summary_path,
                    example_report_path=step_example_path,
                )
                eval_score, eval_split = _select_eval_score(
                    eval_metrics=step_metrics,
                    preferred_order=selection_order,
                )
                history["eval_epochs"].append(epoch + 1)
                history["eval_selection_score"].append(float(eval_score))
                history["eval_selection_split"].append(str(eval_split or ""))

                if eval_score > best_eval_reward:
                    best_eval_reward = float(eval_score)
                    best_eval_epoch = int(epoch + 1)
                    best_eval_split = str(eval_split) if eval_split else None
                    best_metrics_by_split = dict(step_metrics)
                    no_improve_eval_steps = 0
                    trainer.save_model(
                        best_ckpt,
                        metrics={
                            "best_eval_epoch": best_eval_epoch,
                            "best_eval_reward": best_eval_reward,
                            "best_eval_split": best_eval_split,
                        },
                    )
                    trainer.save_model(
                        model_ckpt,
                        metrics={
                            "best_eval_epoch": best_eval_epoch,
                            "best_eval_reward": best_eval_reward,
                            "best_eval_split": best_eval_split,
                        },
                    )
                    _save_best_model_performance(
                        out_path=metrics_root / "best_model_performance.json",
                        best_eval_epoch=best_eval_epoch,
                        best_eval_reward=best_eval_reward,
                        best_eval_split=best_eval_split,
                        best_metrics_by_split=best_metrics_by_split,
                    )
                else:
                    no_improve_eval_steps += 1

                epoch_pbar.set_postfix(
                    train_iqm=f"{float(train_stats['iqm']):.4f}",
                    eval_score=f"{float(eval_score):.4f}",
                    eval_split=str(eval_split or "n/a"),
                )

                if (
                    int(args.early_stopping_patience_evals) > 0
                    and no_improve_eval_steps >= int(args.early_stopping_patience_evals)
                ):
                    print(
                        "Early stopping: "
                        f"no eval-score improvement for {no_improve_eval_steps} checkpoints."
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
                    "best_eval_reward": None,
                    "best_eval_split": None,
                    "note": "No ONNX evaluation checkpoint selected; saved final training weights.",
                },
            )
            trainer.save_model(
                model_ckpt,
                metrics={
                    "best_eval_epoch": None,
                    "best_eval_reward": None,
                    "best_eval_split": None,
                    "note": "No ONNX evaluation checkpoint selected; saved final training weights.",
                },
            )
            _save_best_model_performance(
                out_path=metrics_root / "best_model_performance.json",
                best_eval_epoch=None,
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
        trainer.to_onnx(onnx_path, node_input_dim=node_input_dim)

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

    with (model_root / f"{args.model_name}_info.txt").open("w", encoding="utf-8") as fh:
        for key, value in vars(args).items():
            fh.write(f"{key} = {value}\n")
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
        fh.write(f"best_eval_reward = {best_eval_reward if best_eval_epoch is not None else None}\n")
        fh.write(f"best_eval_split = {best_eval_split}\n")
        fh.write(f"onnx_path = {onnx_path.as_posix()}\n")
        fh.write(f"final_eval_splits = {sorted(final_eval_metrics.keys())}\n")

    return onnx_path.as_posix()


if __name__ == "__main__":
    main(parse_args())

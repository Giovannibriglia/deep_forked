from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import networkx as nx
import pydot
import torch
from tqdm import tqdm

from src.data import (
    build_frontier_dataloader,
    build_frontier_samples,
    get_dataloaders,
    group_clean_frontiers,
    normalize_distance_to_reward,
    read_frontier_csv,
    refresh_frontier_sample_targets,
    seed_everything,
)
from src.graph_utils import VALID_DATASET_TYPES, combine_graphs, load_pyg_graph
from src.models.frontier_policy import FrontierPolicyNetwork
from src.trainer import (
    FAILURE_EPS,
    FAILURE_REWARD_VALUE,
    REGIME_ALL,
    REGIME_ORDER,
    RLFrontierTrainer,
)

U64_MAX = (1 << 64) - 1
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
U64_CHUNK_BITS = 32
U64_CHUNK_BASE = 1 << U64_CHUNK_BITS
U64_CHUNK_MASK = U64_CHUNK_BASE - 1


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
        description="Offline RL frontier selector training and export pipeline."
    )
    parser.add_argument("--subset-train", action="extend", nargs="+", type=str, default=[])
    parser.add_argument("--folder-raw-data", type=str, default="out/NN/Training")
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
            "'merged': Goal CSV path is ignored. "
            "'separated': Goal CSV path is loaded for each frontier."
        ),
    )
    parser.add_argument(
        "--n-max-dataset-queries",
        type=int,
        default=1000,
        help="Maximum number of random/stress evaluation queries generated per dataset.",
    )
    parser.add_argument(
        "--max-size-frontier",
        type=int,
        default=25,
        help="Maximum frontier size for stress evaluation query scheduling (FIFO/LIFO).",
    )
    parser.add_argument(
        "--max-failure-states-per-dataset",
        type=ratio_0_1,
        default=0.3,
        help=(
            "Keep all train frontiers with no failure states, then keep at most "
            "this fraction of train frontiers that have at least one failure "
            "state (ratio w.r.t. no-failure train frontiers)."
        ),
    )

    parser.add_argument("--batch-size", type=int, default=9092)
    parser.add_argument("--n-train-epochs", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=20)
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
    parser.add_argument(
        "--use-goal-separate-input",
        type=str2bool,
        default=None,
        help=(
            "Enable separate goal input branch. Recommended for kind-of-data='separated'; "
            "for 'merged' the model ignores missing goal tensors."
        ),
    )
    parser.add_argument("--reward-formulation", type=str, default="negative_distance")
    parser.add_argument("--max-regular-distance-for-reward", type=float, default=50.0)
    parser.add_argument("--failure-reward-value", type=float, default=-1.0)

    parser.add_argument("--build-data", type=str2bool, default=True)
    parser.add_argument(
        "--build-eval-data",
        type=str2bool,
        default=True,
        help=(
            "If true and --evaluate is true, rebuild evaluation samples "
            "(random + stress) before running evaluation."
        ),
    )
    parser.add_argument("--train", type=str2bool, default=True)
    parser.add_argument("--evaluate", type=str2bool, default=True)
    parser.add_argument(
        "--evaluate-new",
        type=str2bool,
        default=False,
        help=(
            "If true and <model_root>/test_data exists, evaluate the exported ONNX "
            "model per dataset and save reports under metrics/eval_test_data."
        ),
    )
    parser.add_argument("--export-onnx", type=str2bool, default=True)
    parser.add_argument(
        "--if-try-example",
        type=str2bool,
        default=True,
        help=(
            "If true and ONNX export is enabled, run one frontier sample through "
            "both PyTorch and ONNX and save parity reports."
        ),
    )
    parser.add_argument(
        "--onnx-frontier-check-size",
        type=int,
        default=5,
        help=(
            "Frontier size for the additional ONNX order/removal check. "
            "Candidates are sampled randomly from one CSV-materialized frontier."
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


def _move_file_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    src.replace(dst)


def _build_or_load_samples(args):
    data_root, _ = _paths(args)
    sample_root = data_root / "data_pytorch"
    sample_root.mkdir(parents=True, exist_ok=True)
    legacy_sample_paths = {
        data_root / "train_samples.pt": sample_root / "train_samples.pt",
        data_root / "eval_samples_random.pt": sample_root / "eval_samples_random.pt",
        data_root / "eval_samples_stress.pt": sample_root / "eval_samples_stress.pt",
        data_root / "samples_params.pt": sample_root / "samples_params.pt",
        data_root / "metrics" / "data" / "train_samples.pt": sample_root / "train_samples.pt",
        data_root / "metrics" / "data" / "eval_samples_random.pt": sample_root / "eval_samples_random.pt",
        data_root / "metrics" / "data" / "eval_samples_stress.pt": sample_root / "eval_samples_stress.pt",
        data_root / "metrics" / "data" / "samples_params.pt": sample_root / "samples_params.pt",
    }
    for src, dst in legacy_sample_paths.items():
        _move_file_if_exists(src, dst)

    sample_paths = {
        "train": sample_root / "train_samples.pt",
        "eval_random": sample_root / "eval_samples_random.pt",
        "eval_stress": sample_root / "eval_samples_stress.pt",
        "params": sample_root / "samples_params.pt",
    }

    def _save_split_payload(
        train_samples: Sequence[Dict[str, Any]],
        eval_random_samples: Sequence[Dict[str, Any]],
        eval_stress_fifo_samples: Sequence[Dict[str, Any]],
        eval_stress_lifo_samples: Sequence[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> None:
        torch.save(list(train_samples), sample_paths["train"])
        torch.save(list(eval_random_samples), sample_paths["eval_random"])
        torch.save(
            {
                "fifo_samples": list(eval_stress_fifo_samples),
                "lifo_samples": list(eval_stress_lifo_samples),
            },
            sample_paths["eval_stress"],
        )
        torch.save(dict(params or {}), sample_paths["params"])

    should_build_train_data = bool(args.build_data)
    should_build_eval_data = bool(args.evaluate and args.build_eval_data)

    if should_build_train_data or should_build_eval_data:
        (
            train_samples,
            eval_random_samples,
            eval_stress_fifo_samples,
            eval_stress_lifo_samples,
            params,
        ) = build_frontier_samples(
            folder_data=args.folder_raw_data,
            list_subset_train=args.subset_train,
            kind_of_data=args.kind_of_data,
            dataset_type=args.dataset_type,
            seed=args.seed,
            max_regular_distance_for_reward=args.max_regular_distance_for_reward,
            failure_reward_value=args.failure_reward_value,
            n_max_dataset_queries=int(args.n_max_dataset_queries),
            max_size_frontier=int(args.max_size_frontier),
            build_eval_data=bool(should_build_eval_data),
        )
        torch.save(list(train_samples), sample_paths["train"])
        torch.save(dict(params or {}), sample_paths["params"])
        if should_build_eval_data:
            _save_split_payload(
                train_samples=train_samples,
                eval_random_samples=eval_random_samples,
                eval_stress_fifo_samples=eval_stress_fifo_samples,
                eval_stress_lifo_samples=eval_stress_lifo_samples,
                params=params,
            )

    if not sample_paths["train"].exists():
        raise FileNotFoundError(
            f"Missing training samples file: {sample_paths['train']}. "
            "Run with --build-data true."
        )

    train_samples = torch.load(sample_paths["train"], weights_only=False)
    params = (
        torch.load(sample_paths["params"], weights_only=False)
        if sample_paths["params"].exists()
        else {}
    )

    if sample_paths["eval_random"].exists() and sample_paths["eval_stress"].exists():
        eval_random_samples = torch.load(sample_paths["eval_random"], weights_only=False)
        stress_payload = torch.load(sample_paths["eval_stress"], weights_only=False)
        if isinstance(stress_payload, dict):
            eval_stress_fifo_samples = list(stress_payload.get("fifo_samples", []))
            eval_stress_lifo_samples = list(stress_payload.get("lifo_samples", []))
        else:
            eval_stress_fifo_samples = list(stress_payload or [])
            eval_stress_lifo_samples = []
    else:
        eval_random_samples = []
        eval_stress_fifo_samples = []
        eval_stress_lifo_samples = []

    payload = {
        "train_samples": list(train_samples),
        "eval_random_samples": list(eval_random_samples),
        "eval_stress_fifo_samples": list(eval_stress_fifo_samples),
        "eval_stress_lifo_samples": list(eval_stress_lifo_samples),
        "params": dict(params or {}),
    }
    return payload, sample_paths


def _infer_num_edge_labels(samples):
    max_label = 0
    found_label = False
    for sample in samples:
        for key in ("edge_attr", "goal_edge_attr"):
            edge_attr = sample.get(key)
            if edge_attr is None or edge_attr.numel() == 0:
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


def _first_available_sample(
    train_samples: Sequence[Dict[str, Any]],
    eval_samples: Sequence[Dict[str, Any]],
    eval2_samples: Sequence[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if train_samples:
        return "train", train_samples[0]
    if eval_samples:
        return "eval", eval_samples[0]
    if eval2_samples:
        return "eval2", eval2_samples[0]
    return None, None


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


def _values_to_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return [float(x) for x in values.detach().cpu().view(-1).tolist()]
    if isinstance(values, (list, tuple)):
        out = []
        for value in values:
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                continue
        return out
    return []


def _extract_orderable_candidates(sample: Dict[str, Any]) -> list[Dict[str, Any]]:
    successor_ids_raw = sample.get("successor_ids")
    if not isinstance(successor_ids_raw, (list, tuple)):
        return []

    successor_ids = [str(x) for x in successor_ids_raw]
    reward_values = _values_to_float_list(sample.get("reward_target", sample.get("rewards")))
    distance_values = _values_to_float_list(sample.get("distance_raw", sample.get("distances")))
    if not reward_values:
        reward_values = [0.0 for _ in successor_ids]
    if not distance_values:
        distance_values = [0.0 for _ in successor_ids]

    n = min(len(successor_ids), len(reward_values), len(distance_values))
    out = []
    for idx in range(n):
        out.append(
            {
                "source_index": int(idx),
                "successor_id": successor_ids[idx],
                "reward": float(reward_values[idx]),
                "distance": float(distance_values[idx]),
            }
        )
    return out


def _build_frontier_sample_from_candidates(
    ordered_candidates: Sequence[Dict[str, Any]],
    successor_graphs: Dict[str, Any],
    goal_graph: Optional[Any],
    goal_path: str,
    base_sample: Dict[str, Any],
    kind_of_data: str,
    failure_reward_value: float,
) -> Dict[str, Any]:
    if not ordered_candidates:
        raise ValueError("Cannot build a frontier sample from an empty candidate list.")

    frontier_graphs = [successor_graphs[str(c["successor_id"])] for c in ordered_candidates]
    combined = combine_graphs(
        frontier_graphs=frontier_graphs,
        goal_graph=goal_graph,
        kind_of_data=kind_of_data,
        action_ids=None,
    )

    reward_values = [float(c["reward"]) for c in ordered_candidates]
    distance_values = [float(c["distance"]) for c in ordered_candidates]
    reward_target = torch.tensor(reward_values, dtype=torch.float32)
    distance_raw = torch.tensor(distance_values, dtype=torch.float32)
    is_failure = torch.tensor(
        [float(r) <= float(failure_reward_value) for r in reward_values],
        dtype=torch.bool,
    )
    oracle_index = int(torch.argmax(reward_target).item())

    sample = {
        "node_features": combined.node_features,
        "edge_index": combined.edge_index,
        "edge_attr": combined.edge_attr,
        "membership": combined.membership,
        "action_map": combined.action_map,
        "successor_ids": [str(c["successor_id"]) for c in ordered_candidates],
        "reward_target": reward_target,
        "rewards": reward_target.clone(),
        "distance_raw": distance_raw,
        "distances": distance_raw.clone(),
        "is_failure": is_failure,
        "oracle_index": torch.tensor(oracle_index, dtype=torch.long),
        "oracle_reward": reward_target[oracle_index].clone(),
        "goal_path": str(goal_path),
        "predecessor_path": str(base_sample.get("predecessor_path", "")),
        "dataset_id": str(base_sample.get("dataset_id", "")),
        "frontier_has_failure": bool(is_failure.any().item()),
    }
    if "stress_schedule" in base_sample:
        sample["stress_schedule"] = str(base_sample.get("stress_schedule", ""))
    if combined.pool_node_index is not None and combined.pool_membership is not None:
        sample["pool_node_index"] = combined.pool_node_index
        sample["pool_membership"] = combined.pool_membership
    if goal_graph is not None:
        sample["goal_node_features"] = goal_graph.node_features
        sample["goal_edge_index"] = goal_graph.edge_index
        sample["goal_edge_attr"] = goal_graph.edge_attr
    return sample


def _infer_static_onnx_output_len(ort_sess: Any) -> Optional[int]:
    try:
        out_shape = ort_sess.get_outputs()[0].shape
        if out_shape and isinstance(out_shape[0], int) and out_shape[0] > 0:
            return int(out_shape[0])
    except Exception:
        return None
    return None


def _run_single_sample_pytorch_and_onnx(
    trainer: RLFrontierTrainer,
    sample: Dict[str, Any],
    kind_of_data: str,
    ort_sess: Any,
    np_mod: Any,
    static_onnx_len: Optional[int],
) -> Dict[str, Any]:
    n_candidates = _infer_num_candidates(sample)
    if n_candidates <= 0:
        return {
            "status": "error",
            "error": "sample has no candidates.",
            "num_candidates_from_sample": int(n_candidates),
        }

    required_base = ["node_features", "edge_index", "edge_attr", "membership"]
    missing_base = [k for k in required_base if k not in sample]
    if missing_base:
        return {
            "status": "error",
            "error": f"sample is missing required tensors {missing_base}.",
            "num_candidates_from_sample": int(n_candidates),
        }

    use_goal_inputs = kind_of_data == "separated" and trainer.model.use_goal_separate_input
    if use_goal_inputs:
        goal_keys = ["goal_node_features", "goal_edge_index", "goal_edge_attr"]
        goal_missing = [k for k in goal_keys if k not in sample]
        if goal_missing:
            return {
                "status": "error",
                "error": f"sample is missing goal tensors {goal_missing}.",
                "num_candidates_from_sample": int(n_candidates),
            }

    node_feature_dtype, node_feature_np_dtype = _node_feature_dtypes_for_dataset(
        trainer.model.dataset_type
    )
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
    if use_goal_inputs:
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
        return {
            "status": "error",
            "error": f"PyTorch inference failed: {exc}",
            "num_candidates_from_sample": int(n_candidates),
        }

    def _mask_numpy(mask_len: int, valid_len: int):
        valid = max(0, min(int(mask_len), int(valid_len)))
        out = torch.zeros((int(mask_len),), dtype=torch.bool)
        if valid > 0:
            out[:valid] = True
        return out.numpy().astype("bool")

    base_onnx_inputs = {
        "node_features": node_features.detach().cpu().numpy().astype(node_feature_np_dtype),
        "edge_index": edge_index.detach().cpu().numpy().astype("int64"),
        "edge_attr": edge_attr.detach().cpu().numpy().astype("int64"),
        "membership": membership.detach().cpu().numpy().astype("int64"),
    }
    if use_goal_inputs:
        goal_batch = sample.get("goal_batch")
        if goal_batch is None:
            goal_batch = torch.zeros(
                (int(sample["goal_node_features"].size(0)),),
                dtype=torch.int64,
            )
        else:
            goal_batch = goal_batch.to(torch.int64)
        base_onnx_inputs["goal_node_features"] = (
            sample["goal_node_features"]
            .detach()
            .cpu()
            .to(node_feature_dtype)
            .numpy()
            .astype(node_feature_np_dtype)
        )
        base_onnx_inputs["goal_edge_index"] = (
            sample["goal_edge_index"].detach().cpu().numpy().astype("int64")
        )
        base_onnx_inputs["goal_edge_attr"] = (
            sample["goal_edge_attr"].detach().cpu().numpy().astype("int64")
        )
        base_onnx_inputs["goal_batch"] = goal_batch.detach().cpu().numpy().astype("int64")

    onnx_logits_np = None
    onnx_mask_len = None
    last_onnx_error = None
    mask_attempts = []
    if static_onnx_len is not None:
        mask_attempts.append(int(static_onnx_len))
    if int(pytorch_logits.numel()) > 0 and int(pytorch_logits.numel()) not in mask_attempts:
        mask_attempts.append(int(pytorch_logits.numel()))
    if int(n_candidates) > 0 and int(n_candidates) not in mask_attempts:
        mask_attempts.append(int(n_candidates))

    for mask_len in mask_attempts:
        try:
            onnx_inputs = dict(base_onnx_inputs)
            onnx_inputs["mask"] = _mask_numpy(mask_len=mask_len, valid_len=n_candidates)
            onnx_logits_np = (
                np_mod.asarray(ort_sess.run(["logits"], onnx_inputs)[0], dtype=np_mod.float32)
                .reshape(-1)
                .astype(np_mod.float32)
            )
            onnx_mask_len = int(mask_len)
            break
        except Exception as exc:
            last_onnx_error = exc

    if onnx_logits_np is None:
        return {
            "status": "error",
            "error": f"ONNX inference failed: {last_onnx_error}",
            "num_candidates_from_sample": int(n_candidates),
            "num_candidates_from_model": int(pytorch_logits.numel()),
        }

    valid_for_onnx = max(0, min(int(n_candidates), int(onnx_logits_np.size)))
    onnx_pred_idx = int(np_mod.argmax(onnx_logits_np[:valid_for_onnx])) if valid_for_onnx > 0 else -1
    onnx_logits_valid = onnx_logits_np[: min(valid_for_onnx, 5)].tolist()

    valid_for_torch = max(0, min(int(n_candidates), int(pytorch_logits.numel())))
    pytorch_pred_idx = (
        int(torch.argmax(pytorch_logits[:valid_for_torch]).item()) if valid_for_torch > 0 else -1
    )

    successor_ids_raw = sample.get("successor_ids")
    successor_ids = [str(x) for x in successor_ids_raw] if isinstance(successor_ids_raw, (list, tuple)) else []
    onnx_pred_successor = (
        successor_ids[onnx_pred_idx] if 0 <= onnx_pred_idx < len(successor_ids) else None
    )
    pytorch_pred_successor = (
        successor_ids[pytorch_pred_idx] if 0 <= pytorch_pred_idx < len(successor_ids) else None
    )

    pytorch_logits_np = pytorch_logits.detach().cpu().numpy().astype("float32")
    max_abs_diff = None
    if onnx_logits_np.shape == pytorch_logits_np.shape:
        max_abs_diff = (
            float(np_mod.max(np_mod.abs(onnx_logits_np - pytorch_logits_np)))
            if onnx_logits_np.size > 0
            else 0.0
        )

    return {
        "status": "ok",
        "num_candidates_from_sample": int(n_candidates),
        "num_candidates_from_model": int(pytorch_logits.numel()),
        "onnx_mask_len_used": int(onnx_mask_len) if onnx_mask_len is not None else None,
        "onnx_prediction_index": int(onnx_pred_idx),
        "onnx_prediction_successor_id": onnx_pred_successor,
        "onnx_logits": onnx_logits_np.tolist(),
        "onnx_logits_valid": onnx_logits_valid,
        "pytorch_prediction_index": int(pytorch_pred_idx),
        "pytorch_prediction_successor_id": pytorch_pred_successor,
        "pytorch_logits": pytorch_logits_np.tolist(),
        "max_abs_logit_diff": max_abs_diff,
    }


def _compare_random_frontier_order_and_shrinking_onnx(
    trainer: RLFrontierTrainer,
    train_samples: Sequence[Dict[str, Any]],
    eval_samples: Sequence[Dict[str, Any]],
    eval2_samples: Sequence[Dict[str, Any]],
    eval3_samples: Sequence[Dict[str, Any]],
    kind_of_data: str,
    dataset_type: str,
    seed: int,
    onnx_frontier_check_size: int,
    failure_reward_value: float,
    onnx_path: Path,
    report_path: Path,
) -> None:
    frontier_size = int(onnx_frontier_check_size)
    if frontier_size <= 0:
        print(
            "[warning] Could not run ONNX order/removal check: "
            f"invalid frontier size {frontier_size}."
        )
        return
    if frontier_size > 8:
        print(
            "[warning] Could not run ONNX order/removal check: "
            f"frontier size {frontier_size} is too large for full permutation logging."
        )
        return

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        print(
            "[warning] Skipping ONNX order/removal check: "
            f"missing dependency ({exc})."
        )
        return

    eligible: list[Tuple[str, int, Dict[str, Any], list[Dict[str, Any]]]] = []
    split_samples = [
        ("train", train_samples),
        ("eval_random", eval_samples),
        ("eval_stress_fifo", eval2_samples),
        ("eval_stress_lifo", eval3_samples),
    ]
    for split_name, split_data in split_samples:
        for sample_idx, sample in enumerate(split_data):
            candidates = _extract_orderable_candidates(sample)
            if len(candidates) < frontier_size:
                continue
            if kind_of_data == "separated" and not str(sample.get("goal_path", "")).strip():
                continue
            eligible.append((split_name, int(sample_idx), sample, candidates))

    if not eligible:
        print(
            "[warning] Could not run ONNX order/removal check: "
            f"no frontier with at least {frontier_size} candidates."
        )
        return

    rng = random.Random(int(seed) + 99173)
    split_name, split_index, base_sample, base_candidates = rng.choice(eligible)
    selected_positions = rng.sample(list(range(len(base_candidates))), frontier_size)
    selected_candidates = [dict(base_candidates[pos]) for pos in selected_positions]

    dataset_type_norm = str(dataset_type).upper()
    successor_graphs: Dict[str, Any] = {}
    try:
        for candidate in selected_candidates:
            successor_id = str(candidate["successor_id"])
            if successor_id not in successor_graphs:
                successor_graphs[successor_id] = load_pyg_graph(
                    successor_id,
                    dataset_type=dataset_type_norm,
                )
    except Exception as exc:
        print(
            "[warning] Could not run ONNX order/removal check: "
            f"failed to load candidate graph ({exc})."
        )
        return

    goal_path = str(base_sample.get("goal_path", "")).strip()
    goal_graph = None
    if kind_of_data == "separated":
        try:
            goal_graph = load_pyg_graph(goal_path, dataset_type=dataset_type_norm)
        except Exception as exc:
            print(
                "[warning] Could not run ONNX order/removal check: "
                f"failed to load goal graph ({exc})."
            )
            return

    try:
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 4
        ort_sess = ort.InferenceSession(
            onnx_path.as_posix(),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        print(
            "[warning] Could not run ONNX order/removal check: "
            f"failed to load ONNX session ({exc})."
        )
        return
    static_onnx_len = _infer_static_onnx_output_len(ort_sess)

    support_probe_results = []
    max_supported_frontier_size = 0
    for probe_size in range(frontier_size, 0, -1):
        probe_candidates = [dict(candidate) for candidate in selected_candidates[:probe_size]]
        try:
            probe_sample = _build_frontier_sample_from_candidates(
                ordered_candidates=probe_candidates,
                successor_graphs=successor_graphs,
                goal_graph=goal_graph,
                goal_path=goal_path,
                base_sample=base_sample,
                kind_of_data=kind_of_data,
                failure_reward_value=float(failure_reward_value),
            )
            probe_prediction = _run_single_sample_pytorch_and_onnx(
                trainer=trainer,
                sample=probe_sample,
                kind_of_data=kind_of_data,
                ort_sess=ort_sess,
                np_mod=np,
                static_onnx_len=static_onnx_len,
            )
        except Exception as exc:
            probe_prediction = {
                "status": "error",
                "error": f"support probe failed: {exc}",
            }

        support_probe_results.append(
            {
                "frontier_size": int(probe_size),
                "status": str(probe_prediction.get("status", "error")),
                "error": (
                    str(probe_prediction.get("error", ""))
                    if probe_prediction.get("status") != "ok"
                    else None
                ),
                "onnx_mask_len_used": probe_prediction.get("onnx_mask_len_used"),
            }
        )
        if probe_prediction.get("status") == "ok":
            max_supported_frontier_size = int(probe_size)
            break

    if max_supported_frontier_size <= 0:
        report = {
            "onnx_path": onnx_path.as_posix(),
            "seed": int(seed),
            "frontier_size_requested": int(frontier_size),
            "frontier_size_effective_for_order_check": 0,
            "selection_source": "csv_materialized_frontier_sample",
            "support_probe_results": support_probe_results,
            "selected_frontier": {
                "sample_split": split_name,
                "sample_index": int(split_index),
                "dataset_id": str(base_sample.get("dataset_id", "")),
                "predecessor_path": str(base_sample.get("predecessor_path", "")),
                "goal_path": goal_path,
                "num_candidates_in_source_frontier": int(len(base_candidates)),
                "selected_positions_in_source_frontier": [int(i) for i in selected_positions],
                "selected_source_frontier_indices": [
                    int(candidate["source_index"]) for candidate in selected_candidates
                ],
                "selected_successor_ids": [
                    str(candidate["successor_id"]) for candidate in selected_candidates
                ],
            },
            "order_permutation_check": {
                "num_permutations": 0,
                "results": [],
            },
            "shrinking_frontier_check": {
                "num_steps": 0,
                "results": [],
            },
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(
            "[warning] Could not run ONNX order/removal check: "
            "no supported frontier size found for the selected sample. "
            f"Report written to: {report_path}"
        )
        return

    if max_supported_frontier_size < frontier_size:
        print(
            "[warning] ONNX order/removal check: requested frontier size "
            f"{frontier_size} is not fully supported by current ONNX graph; "
            f"running order permutations with effective size {max_supported_frontier_size}."
        )

    permutation_candidates = [
        dict(candidate) for candidate in selected_candidates[:max_supported_frontier_size]
    ]
    permutation_results = []
    for permutation_idx, permutation in enumerate(
        itertools.permutations(range(max_supported_frontier_size))
    ):
        ordered_candidates = [permutation_candidates[pos] for pos in permutation]
        try:
            ordered_sample = _build_frontier_sample_from_candidates(
                ordered_candidates=ordered_candidates,
                successor_graphs=successor_graphs,
                goal_graph=goal_graph,
                goal_path=goal_path,
                base_sample=base_sample,
                kind_of_data=kind_of_data,
                failure_reward_value=float(failure_reward_value),
            )
            prediction = _run_single_sample_pytorch_and_onnx(
                trainer=trainer,
                sample=ordered_sample,
                kind_of_data=kind_of_data,
                ort_sess=ort_sess,
                np_mod=np,
                static_onnx_len=static_onnx_len,
            )
        except Exception as exc:
            prediction = {
                "status": "error",
                "error": f"failed to build/evaluate permutation: {exc}",
            }

        permutation_results.append(
            {
                "permutation_index": int(permutation_idx),
                "selected_positions_order": [int(i) for i in permutation],
                "source_frontier_indices_order": [
                    int(candidate["source_index"]) for candidate in ordered_candidates
                ],
                "successor_ids_order": [
                    str(candidate["successor_id"]) for candidate in ordered_candidates
                ],
                "prediction": prediction,
            }
        )

    shrinking_results = []
    shrinking_candidates = [dict(candidate) for candidate in selected_candidates]
    removed_from_previous_step = None
    step = 0
    while shrinking_candidates:
        try:
            shrinking_sample = _build_frontier_sample_from_candidates(
                ordered_candidates=shrinking_candidates,
                successor_graphs=successor_graphs,
                goal_graph=goal_graph,
                goal_path=goal_path,
                base_sample=base_sample,
                kind_of_data=kind_of_data,
                failure_reward_value=float(failure_reward_value),
            )
            prediction = _run_single_sample_pytorch_and_onnx(
                trainer=trainer,
                sample=shrinking_sample,
                kind_of_data=kind_of_data,
                ort_sess=ort_sess,
                np_mod=np,
                static_onnx_len=static_onnx_len,
            )
        except Exception as exc:
            prediction = {
                "status": "error",
                "error": f"failed to build/evaluate shrinking frontier: {exc}",
            }

        shrinking_results.append(
            {
                "step": int(step),
                "frontier_size": int(len(shrinking_candidates)),
                "removed_from_previous_step": removed_from_previous_step,
                "source_frontier_indices_order": [
                    int(candidate["source_index"]) for candidate in shrinking_candidates
                ],
                "successor_ids_order": [
                    str(candidate["successor_id"]) for candidate in shrinking_candidates
                ],
                "prediction": prediction,
            }
        )

        if len(shrinking_candidates) <= 1:
            break

        remove_pos = int(rng.randrange(len(shrinking_candidates)))
        removed_candidate = shrinking_candidates.pop(remove_pos)
        removed_from_previous_step = {
            "removed_position": int(remove_pos),
            "source_frontier_index": int(removed_candidate["source_index"]),
            "successor_id": str(removed_candidate["successor_id"]),
        }
        step += 1

    report = {
        "onnx_path": onnx_path.as_posix(),
        "seed": int(seed),
        "frontier_size_requested": int(frontier_size),
        "frontier_size_effective_for_order_check": int(max_supported_frontier_size),
        "selection_source": "csv_materialized_frontier_sample",
        "support_probe_results": support_probe_results,
        "selected_frontier": {
            "sample_split": split_name,
            "sample_index": int(split_index),
            "dataset_id": str(base_sample.get("dataset_id", "")),
            "predecessor_path": str(base_sample.get("predecessor_path", "")),
            "goal_path": goal_path,
            "num_candidates_in_source_frontier": int(len(base_candidates)),
            "selected_positions_in_source_frontier": [int(i) for i in selected_positions],
            "selected_source_frontier_indices": [
                int(candidate["source_index"]) for candidate in selected_candidates
            ],
            "selected_successor_ids": [
                str(candidate["successor_id"]) for candidate in selected_candidates
            ],
        },
        "order_permutation_check": {
            "num_permutations": int(len(permutation_results)),
            "results": permutation_results,
        },
        "shrinking_frontier_check": {
            "num_steps": int(len(shrinking_results)),
            "results": shrinking_results,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(
        "[onnx-check] frontier order/removal report written to: "
        f"{report_path}"
    )


def _compare_first_example_pytorch_vs_onnx(
    trainer: RLFrontierTrainer,
    train_samples: Sequence[Dict[str, Any]],
    eval_samples: Sequence[Dict[str, Any]],
    eval2_samples: Sequence[Dict[str, Any]],
    kind_of_data: str,
    onnx_path: Path,
    report_path: Path,
) -> None:
    split_name, sample = _first_available_sample(train_samples, eval_samples, eval2_samples)
    if sample is None:
        print("[warning] Could not run ONNX parity check: no sample available.")
        return

    n_candidates = _infer_num_candidates(sample)
    if n_candidates <= 0:
        print("[warning] Could not run ONNX parity check: sample has no candidates.")
        return

    required_base = ["node_features", "edge_index", "edge_attr", "membership"]
    missing_base = [k for k in required_base if k not in sample]
    if missing_base:
        print(
            "[warning] Could not run ONNX parity check: "
            f"sample is missing required tensors {missing_base}."
        )
        return

    use_goal_inputs = kind_of_data == "separated" and trainer.model.use_goal_separate_input
    goal_missing = []
    if use_goal_inputs:
        goal_keys = ["goal_node_features", "goal_edge_index", "goal_edge_attr"]
        goal_missing = [k for k in goal_keys if k not in sample]
    if goal_missing:
        print(
            "[warning] Could not run ONNX parity check: "
            f"sample is missing goal tensors {goal_missing}."
        )
        return

    node_feature_dtype, node_feature_np_dtype = _node_feature_dtypes_for_dataset(
        trainer.model.dataset_type
    )
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
    if use_goal_inputs:
        goal_batch = sample.get("goal_batch")
        if goal_batch is None:
            # Single-sample parity check: goal graph belongs to one frontier.
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

    with torch.no_grad():
        trainer.model.eval()
        # Derive mask shape from actual forward output to stay robust to
        # runtime/export shape differences.
        pytorch_logits = trainer.model(**model_kwargs).detach().cpu().to(torch.float32).view(-1)
    mask = torch.ones((int(pytorch_logits.numel()),), dtype=torch.bool)
    model_kwargs["mask"] = mask.to(trainer.device)
    with torch.no_grad():
        trainer.model.eval()
        pytorch_logits = trainer.model(**model_kwargs).detach().cpu().to(torch.float32).view(-1)

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        print(f"[warning] Skipping ONNX parity check: missing dependency ({exc}).")
        return

    def _mask_numpy(mask_len: int, valid_len: int):
        valid = max(0, min(int(mask_len), int(valid_len)))
        out = torch.zeros((int(mask_len),), dtype=torch.bool)
        if valid > 0:
            out[:valid] = True
        return out.numpy().astype("bool")

    base_onnx_inputs = {
        "node_features": node_features.detach().cpu().numpy().astype(node_feature_np_dtype),
        "edge_index": edge_index.detach().cpu().numpy().astype("int64"),
        "edge_attr": edge_attr.detach().cpu().numpy().astype("int64"),
        "membership": membership.detach().cpu().numpy().astype("int64"),
    }
    if use_goal_inputs:
        goal_batch = sample.get("goal_batch")
        if goal_batch is None:
            goal_batch = torch.zeros(
                (int(sample["goal_node_features"].size(0)),),
                dtype=torch.int64,
            )
        else:
            goal_batch = goal_batch.to(torch.int64)
        base_onnx_inputs["goal_node_features"] = (
            sample["goal_node_features"]
            .detach()
            .cpu()
            .to(node_feature_dtype)
            .numpy()
            .astype(node_feature_np_dtype)
        )
        base_onnx_inputs["goal_edge_index"] = (
            sample["goal_edge_index"].detach().cpu().numpy().astype("int64")
        )
        base_onnx_inputs["goal_edge_attr"] = (
            sample["goal_edge_attr"].detach().cpu().numpy().astype("int64")
        )
        base_onnx_inputs["goal_batch"] = goal_batch.detach().cpu().numpy().astype("int64")

    try:
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 4
        ort_sess = ort.InferenceSession(
            onnx_path.as_posix(),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:  # pragma: no cover - runtime dependent
        print(f"[warning] ONNX parity check failed while loading session: {exc}")
        return

    onnx_logits = None
    onnx_mask_len = None
    last_onnx_error = None
    static_onnx_len = None
    try:
        out_shape = ort_sess.get_outputs()[0].shape
        if out_shape and isinstance(out_shape[0], int) and out_shape[0] > 0:
            static_onnx_len = int(out_shape[0])
    except Exception:
        static_onnx_len = None

    mask_attempts = []
    if static_onnx_len is not None:
        mask_attempts.append(static_onnx_len)
    if int(pytorch_logits.numel()) > 0 and int(pytorch_logits.numel()) not in mask_attempts:
        mask_attempts.append(int(pytorch_logits.numel()))
    if int(n_candidates) > 0 and int(n_candidates) not in mask_attempts:
        mask_attempts.append(int(n_candidates))

    for mask_len in mask_attempts:
        try:
            onnx_inputs = dict(base_onnx_inputs)
            onnx_inputs["mask"] = _mask_numpy(mask_len=mask_len, valid_len=n_candidates)
            onnx_logits = (
                np.asarray(ort_sess.run(["logits"], onnx_inputs)[0], dtype=np.float32)
                .reshape(-1)
                .astype(np.float32)
            )
            onnx_mask_len = int(mask_len)
            break
        except Exception as exc:  # pragma: no cover - runtime dependent
            last_onnx_error = exc

    if onnx_logits is None:
        print(f"[warning] ONNX parity check failed: {last_onnx_error}")
        return

    gt_rewards = sample.get("reward_target", sample.get("rewards"))
    gt_rewards_t = None
    if isinstance(gt_rewards, torch.Tensor):
        gt_rewards_t = gt_rewards.detach().cpu().to(torch.float32).view(-1)
    elif gt_rewards is not None:
        try:
            gt_rewards_t = torch.tensor(gt_rewards, dtype=torch.float32).view(-1)
        except (TypeError, ValueError):
            gt_rewards_t = None

    gt_valid_len = 0
    ground_truth_index = None
    ground_truth_reward = None
    if gt_rewards_t is not None and gt_rewards_t.numel() > 0:
        gt_valid_len = max(0, min(int(n_candidates), int(gt_rewards_t.numel())))
        if gt_valid_len > 0:
            gt_slice = gt_rewards_t[:gt_valid_len]
            gt_local_idx = int(torch.argmax(gt_slice).item())
            ground_truth_index = gt_local_idx
            ground_truth_reward = float(gt_slice[gt_local_idx].item())

    pytorch_logits_np = pytorch_logits.numpy().astype("float32")
    max_abs_diff = None
    if onnx_logits.shape == pytorch_logits_np.shape:
        max_abs_diff = (
            float(np.max(np.abs(onnx_logits - pytorch_logits_np)))
            if onnx_logits.size > 0
            else 0.0
        )

    valid_for_torch = max(0, min(int(n_candidates), int(pytorch_logits.numel())))
    if valid_for_torch > 0:
        pytorch_pred_idx = int(torch.argmax(pytorch_logits[:valid_for_torch]).item())
    else:
        pytorch_pred_idx = -1

    valid_for_onnx = max(0, min(int(n_candidates), int(onnx_logits.size)))
    if valid_for_onnx > 0:
        onnx_pred_idx = int(np.argmax(onnx_logits[:valid_for_onnx]))
    else:
        onnx_pred_idx = -1

    sample_oracle_index = sample.get("oracle_index")
    if isinstance(sample_oracle_index, torch.Tensor) and sample_oracle_index.numel() > 0:
        sample_oracle_index = int(sample_oracle_index.view(-1)[0].item())
    elif sample_oracle_index is not None:
        try:
            sample_oracle_index = int(sample_oracle_index)
        except (TypeError, ValueError):
            sample_oracle_index = None
    else:
        sample_oracle_index = None

    sample_oracle_reward = sample.get("oracle_reward")
    if isinstance(sample_oracle_reward, torch.Tensor) and sample_oracle_reward.numel() > 0:
        sample_oracle_reward = float(sample_oracle_reward.view(-1)[0].item())
    elif sample_oracle_reward is not None:
        try:
            sample_oracle_reward = float(sample_oracle_reward)
        except (TypeError, ValueError):
            sample_oracle_reward = None
    else:
        sample_oracle_reward = None

    parity_report = {
        "sample_split": split_name,
        "num_candidates": int(pytorch_logits.numel()),
        "num_candidates_from_sample": int(n_candidates),
        "ground_truth_index": ground_truth_index,
        "ground_truth_reward": ground_truth_reward,
        "ground_truth_valid_len": int(gt_valid_len),
        "sample_oracle_index": sample_oracle_index,
        "sample_oracle_reward": sample_oracle_reward,
        "onnx_output_static_len": int(static_onnx_len) if static_onnx_len is not None else None,
        "onnx_mask_len_used": int(onnx_mask_len) if onnx_mask_len is not None else None,
        "onnx_path": onnx_path.as_posix(),
        "pytorch_prediction_index": pytorch_pred_idx,
        "onnx_prediction_index": onnx_pred_idx,
        "prediction_match": bool(pytorch_pred_idx == onnx_pred_idx),
        "pytorch_matches_ground_truth": (
            bool(pytorch_pred_idx == ground_truth_index)
            if ground_truth_index is not None
            else None
        ),
        "onnx_matches_ground_truth": (
            bool(onnx_pred_idx == ground_truth_index)
            if ground_truth_index is not None
            else None
        ),
        "max_abs_logit_diff": max_abs_diff,
        "pytorch_logits": pytorch_logits_np.tolist(),
        "onnx_logits": onnx_logits.tolist(),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(parity_report, fh, indent=2)

    print(
        "[onnx-check] first sample split="
        f"{split_name} | torch_idx={pytorch_pred_idx} | onnx_idx={onnx_pred_idx}"
    )
    if max_abs_diff is not None:
        print(f"[onnx-check] max_abs_logit_diff={max_abs_diff:.8f}")
    print(f"[onnx-check] report written to: {report_path}")


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


def _run_onnx_single_frontier(
    sample: Dict[str, Any],
    ort_sess: Any,
    np_mod: Any,
    static_onnx_len: Optional[int],
    use_goal_inputs: bool,
    dataset_type: str,
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
    base_onnx_inputs = {
        "node_features": (
            sample["node_features"]
            .detach()
            .cpu()
            .to(node_feature_dtype)
            .numpy()
            .astype(node_feature_np_dtype)
        ),
        "edge_index": (
            sample["edge_index"].detach().cpu().to(torch.int64).numpy().astype("int64")
        ),
        "edge_attr": sample["edge_attr"].detach().cpu().to(torch.int64).numpy().astype("int64"),
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

        base_onnx_inputs["goal_node_features"] = (
            sample["goal_node_features"]
            .detach()
            .cpu()
            .to(node_feature_dtype)
            .numpy()
            .astype(node_feature_np_dtype)
        )
        base_onnx_inputs["goal_edge_index"] = (
            sample["goal_edge_index"].detach().cpu().to(torch.int64).numpy().astype("int64")
        )
        base_onnx_inputs["goal_edge_attr"] = (
            sample["goal_edge_attr"].detach().cpu().to(torch.int64).numpy().astype("int64")
        )
        base_onnx_inputs["goal_batch"] = (
            goal_batch.detach().cpu().to(torch.int64).numpy().astype("int64")
        )

    def _mask_numpy(mask_len: int, valid_len: int):
        valid = max(0, min(int(mask_len), int(valid_len)))
        out = torch.zeros((int(mask_len),), dtype=torch.bool)
        if valid > 0:
            out[:valid] = True
        return out.numpy().astype("bool")

    mask_attempts = []
    if static_onnx_len is not None and int(static_onnx_len) > 0:
        mask_attempts.append(int(static_onnx_len))
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
                np_mod.asarray(ort_sess.run(["logits"], onnx_inputs)[0], dtype=np_mod.float32)
                .reshape(-1)
                .astype(np_mod.float32)
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

    pred_idx = int(np_mod.argmax(onnx_logits_np[:valid_for_onnx]))
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


def _evaluate_onnx_dataset_samples(
    trainer: RLFrontierTrainer,
    samples: Sequence[Dict[str, Any]],
    dataset_label: str,
    ort_sess: Any,
    np_mod: Any,
    static_onnx_len: Optional[int],
    use_goal_inputs: bool,
) -> Dict[str, Any]:
    chosen_rewards: list[float] = []
    accuracies: list[int] = []
    frontier_has_failure: list[int] = []
    frontier_sizes: list[int] = []
    abs_reward_gaps: list[float] = []
    failed_frontiers: list[Dict[str, Any]] = []

    for idx, sample in enumerate(
        tqdm(samples, desc=f"Evaluating {dataset_label}", leave=False)
    ):
        result = _run_onnx_single_frontier(
            sample=sample,
            ort_sess=ort_sess,
            np_mod=np_mod,
            static_onnx_len=static_onnx_len,
            use_goal_inputs=use_goal_inputs,
            dataset_type=trainer.model.dataset_type,
        )
        if result.get("status") != "ok":
            failed_frontiers.append(
                {
                    "sample_index": int(idx),
                    "dataset_id": str(sample.get("dataset_id", "")),
                    "predecessor_path": str(sample.get("predecessor_path", "")),
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
    metrics["n_attempted_frontiers"] = int(len(samples))
    metrics["n_failed_frontiers"] = int(len(failed_frontiers))
    metrics["n_evaluated_frontiers"] = int(len(chosen_rewards))
    if failed_frontiers:
        metrics["failed_frontiers"] = failed_frontiers
    return metrics


@dataclass
class _EvalGraphTensors:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    node_names: torch.Tensor


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


def _split_uint64_eval(value: int, *, context: str) -> tuple[int, int]:
    if value < 0 or value > U64_MAX:
        raise ValueError(f"{context} is out of uint64 range [0, {U64_MAX}].")
    hi = value >> U64_CHUNK_BITS
    lo = value & U64_CHUNK_MASK
    return int(hi), int(lo)


def _load_graph_tensors_no_pyg(path: str, dataset_type: str) -> _EvalGraphTensors:
    dataset_type_norm = str(dataset_type).upper()
    if dataset_type_norm not in VALID_DATASET_TYPES:
        raise ValueError(
            f"Unsupported dataset_type '{dataset_type}'. "
            f"Expected one of {sorted(VALID_DATASET_TYPES)}."
        )

    dot_src = Path(path).read_text()
    dot = pydot.graph_from_dot_data(dot_src)[0]
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
        rows: list[list[int]] = []
        for node in nodes:
            parsed = _parse_numeric_node_label_eval(node)
            hi, lo = _split_uint64_eval(parsed, context=f"Node '{node}'")
            rows.append([hi, lo])
        node_features = torch.tensor(rows, dtype=torch.int64)
        node_names = node_features.clone()
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


def _materialize_frontier_no_pyg(
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
    if combined.pool_node_index is not None and combined.pool_membership is not None:
        sample["pool_node_index"] = combined.pool_node_index
        sample["pool_membership"] = combined.pool_membership
    if goal_graph is not None:
        sample["goal_node_features"] = goal_graph.node_features
        sample["goal_edge_index"] = goal_graph.edge_index
        sample["goal_edge_attr"] = goal_graph.edge_attr
    return sample


def _build_test_data_datasets(
    test_data_root: Path,
    kind_of_data: str,
    dataset_type: str,
    max_regular_distance_for_reward: float,
    failure_reward_value: float,
) -> list[Dict[str, Any]]:
    dataset_type_norm = str(dataset_type).upper()
    dataset_entries: list[Tuple[str, Path]] = []
    for problem_dir in sorted(p for p in test_data_root.iterdir() if p.is_dir()):
        for csv_path in sorted(p for p in problem_dir.iterdir() if p.suffix.lower() == ".csv"):
            dataset_entries.append((problem_dir.name, csv_path))

    graph_cache: Dict[str, Any] = {}

    def _load_graph(graph_path: str) -> Any:
        key = str(graph_path)
        if key in graph_cache:
            return graph_cache[key]
        graph = _load_graph_tensors_no_pyg(key, dataset_type=dataset_type_norm)
        graph_cache[key] = graph
        return graph

    out: list[Dict[str, Any]] = []
    for problem_name, csv_path in dataset_entries:
        rows = read_frontier_csv(csv_path=csv_path, kind_of_data=kind_of_data)
        clean_frontiers = group_clean_frontiers(
            rows=rows,
            kind_of_data=kind_of_data,
            max_regular_distance_for_reward=float(max_regular_distance_for_reward),
            failure_reward_value=float(failure_reward_value),
        )
        dataset_id = f"{problem_name}/{csv_path.name}"
        for frontier in clean_frontiers:
            frontier["dataset_id"] = dataset_id

        samples = [
            _materialize_frontier_no_pyg(
                frontier=frontier,
                kind_of_data=kind_of_data,
                dataset_type=dataset_type_norm,
                max_regular_distance_for_reward=float(max_regular_distance_for_reward),
                failure_reward_value=float(failure_reward_value),
                graph_loader=_load_graph,
            )
            for frontier in tqdm(
                clean_frontiers,
                desc=f"Materializing {dataset_id}",
                leave=False,
            )
        ]

        out.append(
            {
                "problem_name": str(problem_name),
                "dataset_id": str(dataset_id),
                "csv_name": str(csv_path.name),
                "csv_stem": str(csv_path.stem),
                "n_rows": int(len(rows)),
                "n_frontiers_kept": int(len(samples)),
                "samples": samples,
            }
        )
    return out


def _save_test_dataset_plots(
    trainer: RLFrontierTrainer,
    problem_dir: Path,
    dataset_csv_stem: str,
    dataset_title: str,
    metrics: Dict[str, Any],
) -> list[str]:
    regimes = metrics.get("regimes")
    if not isinstance(regimes, dict):
        return []

    generated_plot_paths: list[str] = []
    for regime in REGIME_ORDER:
        regime_metrics = regimes.get(regime)
        if not isinstance(regime_metrics, dict):
            continue

        reward_curve = regime_metrics.get("reward_by_size", {})
        reward_plot = (
            problem_dir / f"{dataset_csv_stem}_{regime}_reward_by_frontier_size_iqm_iqrstd.png"
        )
        trainer._plot_frontier_size_curve_with_band(
            out_path=reward_plot,
            sizes=[int(x) for x in reward_curve.get("sizes", [])],
            iqm=[float(x) for x in reward_curve.get("iqm", [])],
            iqr_std=[float(x) for x in reward_curve.get("iqr_std", [])],
            ylabel="Reward",
            y_lim=(-1.0, 0.05),
            title=f"{dataset_title} | {regime} | Reward by Frontier Size (IQM ± IQR-STD)",
        )
        generated_plot_paths.append(reward_plot.as_posix())

        accuracy_curve = regime_metrics.get("accuracy_by_size", {})
        accuracy_plot = (
            problem_dir / f"{dataset_csv_stem}_{regime}_accuracy_by_frontier_size_iqm_iqrstd.png"
        )
        trainer._plot_frontier_size_curve_with_band(
            out_path=accuracy_plot,
            sizes=[int(x) for x in accuracy_curve.get("sizes", [])],
            iqm=[float(x) for x in accuracy_curve.get("iqm", [])],
            iqr_std=[float(x) for x in accuracy_curve.get("iqr_std", [])],
            ylabel="Accuracy",
            y_lim=(-0.05, 1.05),
            title=f"{dataset_title} | {regime} | Accuracy by Frontier Size (IQM ± IQR-STD)",
        )
        generated_plot_paths.append(accuracy_plot.as_posix())

        best_minus_achieved_curve = regime_metrics.get("abs_reward_gap_by_size", {})
        best_minus_achieved_plot = (
            problem_dir
            / f"{dataset_csv_stem}_{regime}_best_minus_achieved_reward_by_frontier_size_iqm_iqrstd.png"
        )
        trainer._plot_frontier_size_curve_with_band(
            out_path=best_minus_achieved_plot,
            sizes=[int(x) for x in best_minus_achieved_curve.get("sizes", [])],
            iqm=[float(x) for x in best_minus_achieved_curve.get("iqm", [])],
            iqr_std=[float(x) for x in best_minus_achieved_curve.get("iqr_std", [])],
            ylabel="best reward achievable - reward achieved",
            y_lim=(0.0, 1.05),
            title=(
                f"{dataset_title} | {regime} | "
                "Best Achievable Reward - Achieved Reward by Frontier Size (IQM ± IQR-STD)"
            ),
        )
        generated_plot_paths.append(best_minus_achieved_plot.as_posix())

    return generated_plot_paths


def _evaluate_onnx_on_test_data(
    trainer: RLFrontierTrainer,
    onnx_path: Path,
    test_data_root: Path,
    metrics_root: Path,
    kind_of_data: str,
    dataset_type: str,
    max_regular_distance_for_reward: float,
    failure_reward_value: float,
) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            f"--evaluate-new requires numpy and onnxruntime. Missing dependency: {exc}"
        ) from exc

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 4
    ort_sess = ort.InferenceSession(
        onnx_path.as_posix(),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    static_onnx_len = _infer_static_onnx_output_len(ort_sess)
    onnx_input_names = {str(x.name) for x in ort_sess.get_inputs()}
    use_goal_inputs = "goal_node_features" in onnx_input_names

    datasets = _build_test_data_datasets(
        test_data_root=test_data_root,
        kind_of_data=kind_of_data,
        dataset_type=dataset_type,
        max_regular_distance_for_reward=float(max_regular_distance_for_reward),
        failure_reward_value=float(failure_reward_value),
    )
    if not datasets:
        print(f"[warning] No CSV datasets found under test_data root: {test_data_root}")
        return

    eval_root = metrics_root / "eval_test_data"
    eval_root.mkdir(parents=True, exist_ok=True)
    problem_summaries: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    all_dataset_rows: list[Dict[str, Any]] = []

    for dataset in tqdm(datasets, desc="Evaluating test_data datasets"):
        problem_name = str(dataset["problem_name"])
        dataset_id = str(dataset["dataset_id"])
        dataset_label = dataset_id
        dataset_metrics = _evaluate_onnx_dataset_samples(
            trainer=trainer,
            samples=dataset["samples"],
            dataset_label=dataset_label,
            ort_sess=ort_sess,
            np_mod=np,
            static_onnx_len=static_onnx_len,
            use_goal_inputs=use_goal_inputs,
        )
        dataset_metrics["dataset_id"] = dataset_id
        dataset_metrics["problem_name"] = problem_name
        dataset_metrics["csv_name"] = str(dataset["csv_name"])
        dataset_metrics["n_rows"] = int(dataset["n_rows"])
        dataset_metrics["n_frontiers_after_cleaning"] = int(dataset["n_frontiers_kept"])
        dataset_metrics["onnx_path"] = onnx_path.as_posix()
        dataset_metrics["onnx_static_output_len"] = (
            int(static_onnx_len) if static_onnx_len is not None else None
        )
        dataset_metrics["onnx_requires_goal_inputs"] = bool(use_goal_inputs)

        problem_dir = eval_root / problem_name
        problem_dir.mkdir(parents=True, exist_ok=True)
        plot_files = _save_test_dataset_plots(
            trainer=trainer,
            problem_dir=problem_dir,
            dataset_csv_stem=str(dataset["csv_stem"]),
            dataset_title=str(dataset_id),
            metrics=dataset_metrics,
        )
        dataset_metrics["plot_files"] = plot_files
        dataset_metrics_path = problem_dir / f"{dataset['csv_stem']}_final_metrics.json"
        with dataset_metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(dataset_metrics, fh, indent=2)

        row = {
            "problem_name": problem_name,
            "dataset_id": dataset_id,
            "csv_name": str(dataset["csv_name"]),
            "metrics_path": dataset_metrics_path.as_posix(),
            "n_rows": int(dataset["n_rows"]),
            "n_frontiers_after_cleaning": int(dataset["n_frontiers_kept"]),
            "n_evaluated_frontiers": int(dataset_metrics.get("n_evaluated_frontiers", 0)),
            "n_failed_frontiers": int(dataset_metrics.get("n_failed_frontiers", 0)),
            "mean_reward": float(dataset_metrics.get("mean_reward", 0.0)),
            "mean_accuracy": float(dataset_metrics.get("mean_accuracy", 0.0)),
            "plot_files": plot_files,
        }
        problem_summaries[problem_name].append(row)
        all_dataset_rows.append(row)

    for problem_name, rows in problem_summaries.items():
        with (eval_root / problem_name / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "problem_name": problem_name,
                    "n_datasets": int(len(rows)),
                    "datasets": rows,
                },
                fh,
                indent=2,
            )

    with (eval_root / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "onnx_path": onnx_path.as_posix(),
                "test_data_root": test_data_root.as_posix(),
                "n_problems": int(len(problem_summaries)),
                "n_datasets": int(len(all_dataset_rows)),
                "datasets": all_dataset_rows,
            },
            fh,
            indent=2,
        )

    print(f"[eval-test-data] ONNX evaluation reports written to: {eval_root}")


def main(args):
    seed_everything(args.seed)
    payload, sample_paths = _build_or_load_samples(args)
    train_samples = payload.get("train_samples", [])
    eval_samples = payload.get("eval_random_samples", [])
    eval2_samples = payload.get("eval_stress_fifo_samples", [])
    eval3_samples = payload.get("eval_stress_lifo_samples", [])

    refresh_frontier_sample_targets(
        train_samples,
        m_failed_state=0.0,
        max_regular_distance_for_reward=float(args.max_regular_distance_for_reward),
        failure_reward_value=float(args.failure_reward_value),
    )
    refresh_frontier_sample_targets(
        eval_samples,
        m_failed_state=0.0,
        max_regular_distance_for_reward=float(args.max_regular_distance_for_reward),
        failure_reward_value=float(args.failure_reward_value),
    )
    refresh_frontier_sample_targets(
        eval2_samples,
        m_failed_state=0.0,
        max_regular_distance_for_reward=float(args.max_regular_distance_for_reward),
        failure_reward_value=float(args.failure_reward_value),
    )
    refresh_frontier_sample_targets(
        eval3_samples,
        m_failed_state=0.0,
        max_regular_distance_for_reward=float(args.max_regular_distance_for_reward),
        failure_reward_value=float(args.failure_reward_value),
    )
    train_samples, train_filter_stats = _limit_failure_frontiers_in_train_dataset(
        train_samples=train_samples,
        max_failure_states_per_dataset=float(args.max_failure_states_per_dataset),
        seed=int(args.seed),
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
    train_state_stats_overall, train_state_stats_by_dataset = _build_train_state_stats(
        train_samples
    )
    print(
        "[dataset-filter] train states | failure/all="
        f"{train_state_stats_overall['train_n_failure_states']}/"
        f"{train_state_stats_overall['train_n_all_states']}"
    )

    if not train_samples:
        raise ValueError(f"No training frontiers loaded from {sample_paths['train']}.")
    if not eval_samples:
        print("[warning] Random eval split is empty; random evaluation metrics will be trivial.")
    if not eval2_samples:
        print("[warning] Stress FIFO split is empty; stress FIFO evaluation metrics will be trivial.")
    if not eval3_samples:
        print("[warning] Stress LIFO split is empty; stress LIFO evaluation metrics will be trivial.")
    if args.evaluate and not args.build_eval_data:
        print("[info] --build-eval-data is false: skipping eval data rebuild, using saved eval files only.")

    dataset_type_norm = str(args.dataset_type).upper()
    if dataset_type_norm != "BITMASK":
        first_node_features = train_samples[0].get("node_features")
        if isinstance(first_node_features, torch.Tensor) and first_node_features.dtype.is_floating_point:
            print(
                "[warning] Loaded cached samples use floating node_features for HASHED/MAPPED. "
                "Regenerate data with --build-data true to avoid precision loss in node IDs."
            )
        if (
            dataset_type_norm == "HASHED"
            and isinstance(first_node_features, torch.Tensor)
            and first_node_features.dim() == 2
            and int(first_node_features.size(1)) != 2
        ):
            print(
                "[warning] HASHED samples should use hi32/lo32 encoding with width 2. "
                "Regenerate data with --build-data true for full uint64 support."
            )

    train_loader, eval_loader, eval2_loader = get_dataloaders(
        train_samples=train_samples,
        eval_samples=eval_samples,
        eval2_samples=eval2_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pad_frontiers=True,
    )
    eval3_loader = build_frontier_dataloader(
        samples=eval3_samples,
        batch_size=args.batch_size,
        seed=int(args.seed) + 3,
        num_workers=args.num_workers,
        pad_frontiers=True,
        shuffle=False,
    )

    node_input_dim = 1 if dataset_type_norm == "HASHED" else int(train_samples[0]["node_features"].size(1))
    use_goal_separate_input = (
        args.kind_of_data == "separated"
        if args.use_goal_separate_input is None
        else bool(args.use_goal_separate_input)
    )
    num_edge_labels = _infer_num_edge_labels(
        train_samples + eval_samples + eval2_samples + eval3_samples
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

    _, model_root = _paths(args)
    metrics_root = model_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    metrics_eval_reports_dir = metrics_root / "eval_reports"
    metrics_eval_reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_onnx_reports_dir = metrics_root / "onnx"
    metrics_onnx_reports_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = model_root / "best.pt"
    onnx_path = model_root / f"{args.model_name}.onnx"
    model_ckpt = metrics_root / f"{args.model_name}.pt"
    legacy_model_ckpt = model_root / f"{args.model_name}.pt"

    legacy_to_metrics = {
        legacy_model_ckpt: model_ckpt,
        model_root / "history_losses.json": metrics_root / "history_losses.json",
        model_root / "best_model_performance.json": metrics_root / "best_model_performance.json",
        model_root / "dataset_split_summary.json": metrics_root / "dataset_split_summary.json",
        model_root / f"{args.model_name}_eval_random.json": (
            metrics_eval_reports_dir / f"{args.model_name}_eval_random.json"
        ),
        model_root / f"{args.model_name}_eval_stress_fifo.json": (
            metrics_eval_reports_dir / f"{args.model_name}_eval_stress_fifo.json"
        ),
        model_root / f"{args.model_name}_eval_stress_lifo.json": (
            metrics_eval_reports_dir / f"{args.model_name}_eval_stress_lifo.json"
        ),
        model_root / f"{args.model_name}_eval_stress.json": (
            metrics_eval_reports_dir / f"{args.model_name}_eval_stress.json"
        ),
        model_root / f"{args.model_name}_onnx_first_example_check.json": (
            metrics_onnx_reports_dir / f"{args.model_name}_onnx_first_example_check.json"
        ),
        model_root / f"{args.model_name}_onnx_frontier_order_removal_check.json": (
            metrics_onnx_reports_dir / f"{args.model_name}_onnx_frontier_order_removal_check.json"
        ),
    }
    for src, dst in legacy_to_metrics.items():
        _move_file_if_exists(src, dst)

    if args.train:
        train_model_name = f"metrics/{args.model_name}"
        trainer.train(
            train_loader=train_loader,
            eval_loader=eval_loader,
            eval2_loader=eval2_loader,
            eval_loader_name="eval_random",
            eval2_loader_name="eval_stress_fifo",
            n_epochs=args.n_train_epochs,
            checkpoint_dir=model_root.as_posix(),
            model_name=train_model_name,
            eval_every=args.eval_every,
            early_stopping_patience_evals=args.early_stopping_patience_evals,
        )
        _move_file_if_exists(
            model_root / "history_losses.json",
            metrics_root / "history_losses.json",
        )
        _move_file_if_exists(
            model_root / "best_model_performance.json",
            metrics_root / "best_model_performance.json",
        )

    load_ckpt = None
    for candidate_path in (best_ckpt, model_ckpt, legacy_model_ckpt):
        if candidate_path.exists():
            load_ckpt = candidate_path
            break

    if load_ckpt is not None:
        loaded = RLFrontierTrainer.load_model(load_ckpt, device=trainer.device)
        trainer.model = loaded.to(trainer.device)
    elif not args.train:
        raise FileNotFoundError(
            f"Missing checkpoint: {best_ckpt} or {model_ckpt} or {legacy_model_ckpt}"
        )
    else:
        raise FileNotFoundError(
            "Training completed but no checkpoint was found at "
            f"{best_ckpt}, {model_ckpt}, or {legacy_model_ckpt}."
        )

    eval_data_available = bool(eval_samples or eval2_samples or eval3_samples)
    if args.evaluate and eval_data_available:
        eval_metrics = trainer.evaluate(eval_loader, verbose=True)
        eval2_metrics = (
            trainer.evaluate(eval2_loader, verbose=True) if eval2_loader is not None else {}
        )
        eval3_metrics = trainer.evaluate(eval3_loader, verbose=True) if eval3_loader is not None else {}

        with (metrics_eval_reports_dir / f"{args.model_name}_eval_random.json").open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(eval_metrics, fh, indent=2)
        with (metrics_eval_reports_dir / f"{args.model_name}_eval_stress_fifo.json").open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(eval2_metrics, fh, indent=2)
        with (metrics_eval_reports_dir / f"{args.model_name}_eval_stress_lifo.json").open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(eval3_metrics, fh, indent=2)
        with (metrics_eval_reports_dir / f"{args.model_name}_eval_stress.json").open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump({"fifo": eval2_metrics, "lifo": eval3_metrics}, fh, indent=2)

        eval_metrics_dir = model_root / "metrics" / "eval_random"
        eval2_metrics_dir = model_root / "metrics" / "eval_stress_fifo"
        eval3_metrics_dir = model_root / "metrics" / "eval_stress_lifo"
        eval_metrics_dir.mkdir(parents=True, exist_ok=True)
        eval2_metrics_dir.mkdir(parents=True, exist_ok=True)
        eval3_metrics_dir.mkdir(parents=True, exist_ok=True)
        with (eval_metrics_dir / "final_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(eval_metrics, fh, indent=2)
        with (eval2_metrics_dir / "final_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(eval2_metrics, fh, indent=2)
        with (eval3_metrics_dir / "final_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(eval3_metrics, fh, indent=2)
    elif args.evaluate:
        print(
            "[warning] Evaluation requested but no eval samples are available. "
            "Set --build-eval-data true (and provide raw data) to generate eval sets."
        )

    params = payload.get("params", {})
    split_summary = params.get("split_summary") or {
        "num_frontiers_train": len(train_samples),
        "num_frontiers_eval_random": len(eval_samples),
        "num_frontiers_eval_stress_fifo": len(eval2_samples),
        "num_frontiers_eval_stress_lifo": len(eval3_samples),
    }
    split_summary = dict(split_summary)
    split_summary["num_frontiers_train"] = len(train_samples)
    split_summary["num_frontiers_eval_random"] = len(eval_samples)
    split_summary["num_frontiers_eval_stress_fifo"] = len(eval2_samples)
    split_summary["num_frontiers_eval_stress_lifo"] = len(eval3_samples)
    # Backward-compatible counters.
    split_summary["num_frontiers_eval"] = len(eval_samples)
    split_summary["num_frontiers_eval2"] = len(eval2_samples)
    split_summary["num_frontiers_eval3"] = len(eval3_samples)
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
    existing_dataset_summaries = split_summary.get("dataset_summaries")
    if isinstance(existing_dataset_summaries, list):
        merged_dataset_summaries = []
        seen_dataset_ids = set()
        for dataset_entry in existing_dataset_summaries:
            row = dict(dataset_entry)
            dataset_id = str(row.get("dataset_id", ""))
            stats = train_state_stats_by_dataset.get(
                dataset_id,
                {
                    "train_frontiers_after_filter": 0,
                    "train_with_failure_and_solution_frontiers_after_filter": 0,
                    "train_n_failure_states": 0,
                    "train_n_all_states": 0,
                },
            )
            row.update(stats)
            row["train_failure_states_over_all_states"] = (
                f"{stats['train_n_failure_states']}/{stats['train_n_all_states']}"
            )
            merged_dataset_summaries.append(row)
            seen_dataset_ids.add(dataset_id)

        for dataset_id in sorted(train_state_stats_by_dataset.keys()):
            if dataset_id in seen_dataset_ids:
                continue
            stats = train_state_stats_by_dataset[dataset_id]
            merged_dataset_summaries.append(
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
            )
        split_summary["dataset_summaries"] = merged_dataset_summaries
    else:
        split_summary["dataset_summaries"] = [
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

    if args.export_onnx:
        trainer.to_onnx(onnx_path, node_input_dim=node_input_dim)
        if args.if_try_example:
            _compare_first_example_pytorch_vs_onnx(
                trainer=trainer,
                train_samples=train_samples,
                eval_samples=eval_samples,
                eval2_samples=eval2_samples,
                kind_of_data=args.kind_of_data,
                onnx_path=onnx_path,
                report_path=(
                    metrics_onnx_reports_dir / f"{args.model_name}_onnx_first_example_check.json"
                ),
            )
            _compare_random_frontier_order_and_shrinking_onnx(
                trainer=trainer,
                train_samples=train_samples,
                eval_samples=eval_samples,
                eval2_samples=eval2_samples,
                eval3_samples=eval3_samples,
                kind_of_data=args.kind_of_data,
                dataset_type=args.dataset_type,
                seed=int(args.seed),
                onnx_frontier_check_size=int(args.onnx_frontier_check_size),
                failure_reward_value=float(args.failure_reward_value),
                onnx_path=onnx_path,
                report_path=(
                    metrics_onnx_reports_dir
                    / f"{args.model_name}_onnx_frontier_order_removal_check.json"
                ),
            )

    if args.evaluate_new:
        test_data_root = model_root / "test_data"
        if not test_data_root.exists() or not test_data_root.is_dir():
            print(
                "[info] --evaluate-new enabled but test_data directory was not found at: "
                f"{test_data_root}"
            )
        else:
            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"--evaluate-new requires ONNX model at {onnx_path}. "
                    "Enable --export-onnx true or provide the file."
                )
            _evaluate_onnx_on_test_data(
                trainer=trainer,
                onnx_path=onnx_path,
                test_data_root=test_data_root,
                metrics_root=metrics_root,
                kind_of_data=args.kind_of_data,
                dataset_type=args.dataset_type,
                max_regular_distance_for_reward=float(
                    args.max_regular_distance_for_reward
                ),
                failure_reward_value=float(args.failure_reward_value),
            )

    best_epoch = None
    best_perf_path = metrics_root / "best_model_performance.json"
    if not best_perf_path.exists():
        legacy_best_perf = model_root / "best_model_performance.json"
        if legacy_best_perf.exists():
            best_perf_path = legacy_best_perf
    if best_perf_path.exists():
        try:
            with best_perf_path.open("r", encoding="utf-8") as fh:
                best_payload = json.load(fh)
            best_epoch = best_payload.get("best_eval_epoch")
        except (OSError, json.JSONDecodeError):
            best_epoch = None

    with (model_root / f"{args.model_name}_info.txt").open("w", encoding="utf-8") as fh:
        for key, value in vars(args).items():
            fh.write(f"{key} = {value}\n")
        fh.write(f"samples_train_path = {sample_paths['train']}\n")
        fh.write(f"samples_eval_random_path = {sample_paths['eval_random']}\n")
        fh.write(f"samples_eval_stress_path = {sample_paths['eval_stress']}\n")
        fh.write(f"samples_params_path = {sample_paths['params']}\n")
        fh.write(f"n_train_samples = {len(train_samples)}\n")
        fh.write(f"n_eval_random_samples = {len(eval_samples)}\n")
        fh.write(f"n_eval_stress_fifo_samples = {len(eval2_samples)}\n")
        fh.write(f"n_eval_stress_lifo_samples = {len(eval3_samples)}\n")
        # Backward-compatible counters.
        fh.write(f"n_eval_samples = {len(eval_samples)}\n")
        fh.write(f"n_eval2_samples = {len(eval2_samples)}\n")
        fh.write(f"n_eval3_samples = {len(eval3_samples)}\n")
        fh.write(f"model_checkpoint = {load_ckpt}\n")
        fh.write(f"best_checkpoint = {best_ckpt}\n")
        fh.write(f"best model epoch = {best_epoch}\n")
        fh.write(f"onnx_path = {onnx_path}\n")

    return onnx_path.as_posix()


if __name__ == "__main__":
    main(parse_args())

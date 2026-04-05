from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data import (
    build_frontier_samples,
    get_dataloaders,
    load_saved_samples,
    refresh_frontier_sample_targets,
    save_samples,
    seed_everything,
)
from src.models.frontier_policy import FrontierPolicyNetwork
from src.trainer import RLFrontierTrainer


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "y", "true", "t"):
        return True
    if v in ("no", "n", "false", "f"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


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
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-random-eval-frontiers-for-dataset", type=int, default=100)

    parser.add_argument("--batch-size", type=int, default=9092)
    parser.add_argument("--n-train-epochs", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=10)
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
    parser.add_argument("--train", type=str2bool, default=True)
    parser.add_argument("--evaluate", type=str2bool, default=True)
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


def _build_or_load_samples(args):
    data_root, _ = _paths(args)
    samples_path = data_root / "samples.pt"

    if args.build_data:
        train_samples, eval_samples, eval2_samples, params = build_frontier_samples(
            folder_data=args.folder_raw_data,
            list_subset_train=args.subset_train,
            kind_of_data=args.kind_of_data,
            dataset_type=args.dataset_type,
            test_size=args.test_size,
            seed=args.seed,
            max_regular_distance_for_reward=args.max_regular_distance_for_reward,
            failure_reward_value=args.failure_reward_value,
            max_random_eval_frontiers_for_dataset=args.max_random_eval_frontiers_for_dataset,
            include_eval2=True,
        )
        save_samples(
            samples_path,
            train_samples,
            eval_samples,
            eval2_samples=eval2_samples,
            params=params,
        )
    payload = load_saved_samples(samples_path)
    return payload, samples_path


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


def main(args):
    seed_everything(args.seed)
    payload, samples_path = _build_or_load_samples(args)
    train_samples = payload.get("train_samples", [])
    eval_samples = payload.get("eval_samples", [])
    eval2_samples = payload.get("eval2_samples", [])

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

    if not train_samples:
        raise ValueError(f"No training frontiers loaded from {samples_path}.")
    if not eval_samples:
        print("[warning] Eval split is empty; evaluation metrics will be trivial.")
    if not eval2_samples:
        print("[warning] Eval2 split is empty; eval2 metrics will be trivial.")

    train_loader, eval_loader, eval2_loader = get_dataloaders(
        train_samples=train_samples,
        eval_samples=eval_samples,
        eval2_samples=eval2_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pad_frontiers=True,
    )

    node_input_dim = int(train_samples[0]["node_features"].size(1))
    use_goal_separate_input = (
        args.kind_of_data == "separated"
        if args.use_goal_separate_input is None
        else bool(args.use_goal_separate_input)
    )
    num_edge_labels = _infer_num_edge_labels(train_samples + eval_samples + eval2_samples)

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
    model_ckpt = model_root / f"{args.model_name}.pt"
    onnx_path = model_root / f"{args.model_name}.onnx"

    if args.train:
        trainer.train(
            train_loader=train_loader,
            eval_loader=eval_loader,
            eval2_loader=eval2_loader,
            n_epochs=args.n_train_epochs,
            checkpoint_dir=model_root.as_posix(),
            model_name=args.model_name,
            eval_every=args.eval_every,
            early_stopping_patience_evals=args.early_stopping_patience_evals,
        )

    if model_ckpt.exists():
        loaded = RLFrontierTrainer.load_model(model_ckpt, device=trainer.device)
        trainer.model = loaded.to(trainer.device)
    elif not args.train:
        raise FileNotFoundError(f"Missing checkpoint: {model_ckpt}")

    if args.evaluate:
        eval_metrics = trainer.evaluate(eval_loader, verbose=True)
        eval2_metrics = trainer.evaluate(eval2_loader, verbose=True) if eval2_loader is not None else {}

        with (model_root / f"{args.model_name}_eval.json").open("w", encoding="utf-8") as fh:
            json.dump(eval_metrics, fh, indent=2)
        with (model_root / f"{args.model_name}_eval2.json").open("w", encoding="utf-8") as fh:
            json.dump(eval2_metrics, fh, indent=2)

        eval_metrics_dir = model_root / "metrics" / "eval"
        eval2_metrics_dir = model_root / "metrics" / "eval2"
        eval_metrics_dir.mkdir(parents=True, exist_ok=True)
        eval2_metrics_dir.mkdir(parents=True, exist_ok=True)
        with (eval_metrics_dir / "final_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(eval_metrics, fh, indent=2)
        with (eval2_metrics_dir / "final_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(eval2_metrics, fh, indent=2)

    params = payload.get("params", {})
    split_summary = params.get("split_summary") or {
        "num_frontiers_train": len(train_samples),
        "num_frontiers_eval": len(eval_samples),
        "num_frontiers_eval2": len(eval2_samples),
    }
    with (model_root / "dataset_split_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(split_summary, fh, indent=2)

    if args.export_onnx:
        trainer.to_onnx(onnx_path, node_input_dim=node_input_dim)

    with (model_root / f"{args.model_name}_info.txt").open("w", encoding="utf-8") as fh:
        for key, value in vars(args).items():
            fh.write(f"{key} = {value}\n")
        fh.write(f"samples_path = {samples_path}\n")
        fh.write(f"n_train_samples = {len(train_samples)}\n")
        fh.write(f"n_eval_samples = {len(eval_samples)}\n")
        fh.write(f"n_eval2_samples = {len(eval2_samples)}\n")
        fh.write(f"model_checkpoint = {model_ckpt}\n")
        fh.write(f"onnx_path = {onnx_path}\n")

    return onnx_path.as_posix()


if __name__ == "__main__":
    main(parse_args())

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.data import (
    build_frontier_samples,
    get_dataloaders,
    load_saved_samples,
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
            "'merged': frontier is one combined graph with shared central goal already embedded; Goal CSV path is ignored. "
            "'separated': frontier candidates stay disconnected and goal graph is loaded separately from Goal CSV path."
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--m-failed-state",
        type=int,
        default=100000,
        help="Distance From Goal value treated as failure state.",
    )
    parser.add_argument(
        "--max-percentage-of-failure-states",
        type=float,
        default=0.1,
        help=(
            "Maximum fraction of failure frontiers included in train split "
            "(failure frontier = contains at least one candidate with Distance From Goal == m_failed_state)."
        ),
    )
    parser.add_argument(
        "--max-percentage-of-failure-states-test",
        type=float,
        default=0.2,
        help=(
            "Maximum fraction of failure frontiers included in eval split "
            "(failure frontier = contains at least one candidate with Distance From Goal == m_failed_state)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-train-epochs", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--conv-type", choices=["gine", "rgcn", "gcn"], default="gine")
    parser.add_argument("--pooling-type", choices=["mean", "sum", "max"], default="mean")
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
    parser.add_argument(
        "--track-rank-correlation",
        type=str2bool,
        default=True,
        help="Track reward rank correlation (Spearman) between logits and rewards during evaluation.",
    )
    parser.add_argument("--reward-formulation", type=str, default="negative_distance")
    parser.add_argument("--reward-loss-weight", type=float, default=1.0)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--ranking-loss-type",
        choices=["none", "pairwise", "listwise"],
        default="none",
    )
    parser.add_argument("--reward-temperature", type=float, default=0.5)

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
        train_samples, eval_samples, params = build_frontier_samples(
            folder_data=args.folder_raw_data,
            list_subset_train=args.subset_train,
            kind_of_data=args.kind_of_data,
            dataset_type=args.dataset_type,
            test_size=args.test_size,
            seed=args.seed,
            m_failed_state=args.m_failed_state,
            max_percentage_of_failure_states=args.max_percentage_of_failure_states,
            max_percentage_of_failure_states_test=args.max_percentage_of_failure_states_test,
        )
        save_samples(samples_path, train_samples, eval_samples, params=params)
    payload = load_saved_samples(samples_path)
    return payload, samples_path


def main(args):
    seed_everything(args.seed)
    payload, samples_path = _build_or_load_samples(args)
    train_samples = payload["train_samples"]
    eval_samples = payload["eval_samples"]

    if not train_samples:
        raise ValueError(f"No training frontiers loaded from {samples_path}.")
    if not eval_samples:
        print("[warning] Eval split is empty; evaluation metrics will be trivial.")

    train_loader, eval_loader = get_dataloaders(
        train_samples=train_samples,
        eval_samples=eval_samples,
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

    model = FrontierPolicyNetwork(
        node_input_dim=node_input_dim,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        conv_type=args.conv_type,
        pooling_type=args.pooling_type,
        use_global_context=args.use_global_context,
        mlp_depth=args.mlp_depth,
        use_goal_separate_input=use_goal_separate_input,
    )
    trainer = RLFrontierTrainer(
        model=model,
        reward_formulation=args.reward_formulation,
        m_failed_state=float(args.m_failed_state),
        kind_of_data=args.kind_of_data,
        track_rank_correlation=args.track_rank_correlation,
        reward_loss_weight=args.reward_loss_weight,
        ranking_loss_weight=args.ranking_loss_weight,
        ranking_loss_type=args.ranking_loss_type,
        reward_temperature=args.reward_temperature,
    )

    _, model_root = _paths(args)
    model_ckpt = model_root / f"{args.model_name}.pt"
    onnx_path = model_root / f"{args.model_name}.onnx"

    if args.train:
        trainer.train(
            train_loader=train_loader,
            eval_loader=eval_loader,
            n_epochs=args.n_train_epochs,
            checkpoint_dir=model_root.as_posix(),
            model_name=args.model_name,
            eval_every=args.eval_every,
        )

    if model_ckpt.exists():
        loaded = RLFrontierTrainer.load_model(model_ckpt, device=trainer.device)
        trainer.model = loaded.to(trainer.device)
    elif not args.train:
        raise FileNotFoundError(f"Missing checkpoint: {model_ckpt}")

    if args.evaluate:
        metrics = trainer.evaluate(eval_loader, verbose=True)
        with (model_root / f"{args.model_name}_eval.json").open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

        with (model_root / f"{args.model_name}_eval_errors.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "n_frontiers": metrics["n_frontiers"],
                    "n_errors": metrics["n_errors"],
                    "error_rate": metrics["error_rate"],
                    "n_action_errors": metrics["n_action_errors"],
                    "action_error_rate": metrics["action_error_rate"],
                    "n_failure_choices": metrics["n_failure_choices"],
                    "failure_choice_rate": metrics["failure_choice_rate"],
                    "n_failure_frontiers": metrics["n_failure_frontiers"],
                    "failure_frontier_rate": metrics["failure_frontier_rate"],
                    "failure_metrics_applicable": metrics["failure_metrics_applicable"],
                    "failure_avoidance_accuracy": metrics["failure_avoidance_accuracy"],
                    "eval_reward_kl": metrics["eval_reward_kl"],
                    "eval_reward_js": metrics["eval_reward_js"],
                    "eval_reward_js_normalized": metrics["eval_reward_js_normalized"],
                    "eval_reward_mae": metrics["eval_reward_mae"],
                    "eval_reward_rmse": metrics["eval_reward_rmse"],
                    "eval_score_std_within_frontier": metrics[
                        "eval_score_std_within_frontier"
                    ],
                    "errors": metrics["errors"],
                },
                fh,
                indent=2,
            )

    params = payload.get("params", {})
    split_summary = params.get("split_summary")
    if not split_summary:
        split_summary = {
            "num_frontiers_train": len(train_samples),
            "num_frontiers_eval": len(eval_samples),
            "m_failed_state": float(args.m_failed_state),
            "requested_max_percentage_of_failure_states_train": float(
                args.max_percentage_of_failure_states
            ),
            "requested_max_percentage_of_failure_states_eval": float(
                args.max_percentage_of_failure_states_test
            ),
        }
    with (model_root / "dataset_split_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(split_summary, fh, indent=2)

    if args.export_onnx:
        trainer.to_onnx(onnx_path, node_input_dim=node_input_dim)

    with (model_root / f"{args.model_name}_info.txt").open("w", encoding="utf-8") as fh:
        for key, value in vars(args).items():
            fh.write(f"{key} = {value}\n")
        fh.write(f"samples_path = {samples_path}\n")
        fh.write(f"model_checkpoint = {model_ckpt}\n")
        fh.write(f"onnx_path = {onnx_path}\n")

    return onnx_path.as_posix()


if __name__ == "__main__":
    main(parse_args())

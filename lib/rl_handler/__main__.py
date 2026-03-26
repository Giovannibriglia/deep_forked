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
    parser.add_argument("--kind-of-data", choices=["merged", "separated"], default="merged")
    parser.add_argument("--test-size", type=float, default=0.2)
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
    parser.add_argument("--reward-formulation", type=str, default="negative_distance")

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
    model = FrontierPolicyNetwork(
        node_input_dim=node_input_dim,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        conv_type=args.conv_type,
        pooling_type=args.pooling_type,
        use_global_context=args.use_global_context,
        mlp_depth=args.mlp_depth,
    )
    trainer = RLFrontierTrainer(
        model=model,
        reward_formulation=args.reward_formulation,
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
        metrics = trainer.evaluate(eval_loader)
        with (model_root / f"{args.model_name}_eval.json").open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print("Evaluation:", metrics)

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

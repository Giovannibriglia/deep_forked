import argparse
import concurrent.futures
import multiprocessing
import os
import subprocess
import time


def find_training_data_folders(batch_root):
    models_root = os.path.join(batch_root, "_models")
    training_data_folders = []
    for root, dirs, _ in os.walk(models_root):
        if "training_data" in dirs:
            training_data_folders.append(os.path.join(root, "training_data"))
    return training_data_folders


def run_training(
    training_data_folder,
    no_goal,
    dataset_type,
    max_regular_distance_for_reward,
    n_max_dataset_queries,
    lr,
    weight_decay,
    max_grad_norm,
    early_stopping_patience_evals,
    failure_reward_value,
    build_data,
    build_eval_data,
):
    if not os.path.isdir(training_data_folder):
        print(f"[ERROR] Missing training folder: {training_data_folder}")
        return

    instance_names = sorted(
        name
        for name in os.listdir(training_data_folder)
        if os.path.isdir(os.path.join(training_data_folder, name))
    )
    if not instance_names:
        print(f"[WARNING] Empty training folder: {training_data_folder}")
        return

    model_dir = os.path.dirname(training_data_folder)
    cmd = [
        "python3",
        "lib/rl_handler/__main__.py",
        "--folder-raw-data",
        training_data_folder,
        "--subset-train",
        *instance_names,
        "--dir-save-model",
        model_dir,
        "--dir-save-data",
        model_dir,
        "--dataset_type",
        dataset_type,
        "--build-data",
        str(bool(build_data)).lower(),
        "--failure-reward-value",
        str(failure_reward_value),
        "--max-regular-distance-for-reward",
        str(max_regular_distance_for_reward),
        "--n-max-dataset-queries",
        str(n_max_dataset_queries),
        "--build-eval-data",
        str(bool(build_eval_data)).lower(),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--max-grad-norm",
        str(max_grad_norm),
        "--early-stopping-patience-evals",
        str(early_stopping_patience_evals),
    ]
    if no_goal:
        cmd.extend(["--kind-of-data", "separated"])
    print(" ".join(cmd))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    prefix = f"[{os.path.basename(model_dir)}]".ljust(20)
    print(f"{prefix} Running RL training...")

    last_print = 0.0
    for line in iter(process.stdout.readline, ""):
        now = time.time()
        if now - last_print >= 15:
            print(f"{prefix} {line.strip()}")
            last_print = now
    process.stdout.close()
    rc = process.wait()
    if rc == 0:
        print(f"{prefix} [SUCCESS]")
    else:
        print(f"{prefix} [ERROR] return code {rc}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RL model training in parallel for all _models/*/training_data folders."
    )
    parser.add_argument("batch_root")
    parser.add_argument("--no_goal", action="store_true")
    parser.add_argument("--dataset_type", choices=["MAPPED", "HASHED", "BITMASK"], default="HASHED")
    parser.add_argument("--max-regular-distance-for-reward", type=float, default=50.0)
    parser.add_argument(
        "--n-max-dataset-queries",
        type=int,
        default=1000,
        help=(
            "Max eval queries per dataset for random and stress evaluation; "
            "accepts legacy alias --max-random-eval-frontiers-for-dataset."
        ),
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--failure-reward-value", type=float, default=-1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience-evals", type=int, default=10)
    parser.add_argument(
        "--build-data",
        choices=["true", "false"],
        default="true",
        help="Whether to rebuild training samples before training.",
    )
    parser.add_argument(
        "--build-eval-data",
        choices=["true", "false"],
        default="true",
        help=(
            "Whether to rebuild eval sample files "
            "(eval_samples_random.pt and eval_samples_stress.pt) before evaluation."
        ),
    )
    args = parser.parse_args()

    folders = find_training_data_folders(args.batch_root)
    if not folders:
        print(f"[ERROR] No training_data folders found in {args.batch_root}/_models/")
        return

    max_workers = min(multiprocessing.cpu_count(), len(folders))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                run_training,
                folder,
                args.no_goal,
                args.dataset_type,
                args.max_regular_distance_for_reward,
                args.n_max_dataset_queries,
                args.lr,
                args.weight_decay,
                args.max_grad_norm,
                args.early_stopping_patience_evals,
                args.failure_reward_value,
                args.build_data.lower() == "true",
                args.build_eval_data.lower() == "true",
            )
            for folder in folders
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()

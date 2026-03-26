import argparse
import os
import subprocess


def find_domains_with_training(batch_dir):
    domains = []
    for root, dirs, _ in os.walk(batch_dir):
        if "Training" in dirs:
            rel_path = os.path.relpath(root, batch_dir)
            domains.append(rel_path)
    return sorted(domains)


def main():
    parser = argparse.ArgumentParser(
        description="Run RL per-domain training-data generation for all domains in a batch folder."
    )
    parser.add_argument("batch_path", help="Path to batch folder (e.g., exp/gnn_exp/batch1/)")
    parser.add_argument("--deep_exe", default="cmake-release-nn/bin/deep")
    parser.add_argument("--no_goal", action="store_true")
    parser.add_argument("--depth", type=int, default=25)
    parser.add_argument("--discard_factor", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_retries", type=int, default=15)
    parser.add_argument("--dataset_type", choices=["MAPPED", "HASHED", "BITMASK"], default="HASHED")
    parser.add_argument(
        "--script_path",
        default="scripts/rl_exp/create_training_data.py",
        help="Path to the per-domain training-data script.",
    )
    args = parser.parse_args()

    batch_path = os.path.abspath(args.batch_path)
    if not os.path.isdir(batch_path):
        print(f"Error: {batch_path} is not a valid directory.")
        return

    domains = find_domains_with_training(batch_path)
    if not domains:
        print("No domains with Training folders found.")
        return

    for domain in domains:
        cmd = [
            "python3",
            args.script_path,
            batch_path,
            domain,
            args.deep_exe,
            "--depth",
            str(args.depth),
            "--discard_factor",
            str(args.discard_factor),
            "--seed",
            str(args.seed),
            "--max_retries",
            str(args.max_retries),
            "--dataset_type",
            str(args.dataset_type),
        ]
        if args.no_goal:
            cmd.append("--no_goal")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] Failed domain {domain} with code {exc.returncode}")

    print("\n=== Batch complete ===")

if __name__ == "__main__":
    main()

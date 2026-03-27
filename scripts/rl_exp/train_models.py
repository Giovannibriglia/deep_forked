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
    m_failed_state,
    max_percentage_of_failure_states,
    max_percentage_of_failure_states_test,
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
        "--m-failed-state",
        str(m_failed_state),
        "--max-percentage-of-failure-states",
        str(max_percentage_of_failure_states),
        "--max-percentage-of-failure-states-test",
        str(max_percentage_of_failure_states_test),
    ]
    if no_goal:
        cmd.extend(["--kind-of-data", "separated"])

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
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
    parser.add_argument("--m-failed-state", type=int, default=100000)
    parser.add_argument("--max-percentage-of-failure-states", type=float, default=0.1)
    parser.add_argument("--max-percentage-of-failure-states-test", type=float, default=0.2)
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
                args.m_failed_state,
                args.max_percentage_of_failure_states,
                args.max_percentage_of_failure_states_test,
            )
            for folder in folders
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()

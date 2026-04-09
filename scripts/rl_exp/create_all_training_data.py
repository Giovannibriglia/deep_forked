import subprocess
import sys


def main():
    cmd = ["python3", "scripts/gnn_exp/create_all_training_data.py", *sys.argv[1:]]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

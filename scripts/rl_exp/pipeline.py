
import subprocess, sys

def run(cmd):
    print(f"[PIPELINE] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    data = sys.argv[1]

    #run(f"python3 scripts/rl_exp/run_all.py {data}")
    run("python3 scripts/rl_exp/aggregate.py")
    run("python3 scripts/rl_exp/analyze_results.py")
    run("python3 scripts/rl_exp/advanced_analysis.py")
    run("python3 scripts/rl_exp/plot_results.py")
    run("python3 scripts/rl_exp/plot_results_best.py")
    run("python3 scripts/rl_exp/plot_results_best_isolated.py")

    print("=== PIPELINE COMPLETE ===")

import pandas as pd
from pathlib import Path

df = pd.read_csv("combined_results/aggregate.csv")

# --- numeric cleanup ---
df["NodesExpanded_mean"] = pd.to_numeric(df["NodesExpanded_mean"], errors="coerce")
df["TotalExecutionTime_mean"] = pd.to_numeric(df["TotalExecutionTime_mean"], errors="coerce")

df["SolvedPct"] = df["Solved"] / df["Total"] * 100

out = Path("combined_results/analysis")
out.mkdir(exist_ok=True)

# --- BEST PER GROUP (min nodes) ---
best = (
    df.sort_values("NodesExpanded_mean")
    .groupby(["Fringe","Strict"])
    .first()
    .reset_index()
)

best.to_csv(out / "best_by_nodes.csv", index=False)

# --- FULL GROUPED STATS ---
grouped = (
    df.groupby(["Heuristic","Fringe","Strict"])
    .mean(numeric_only=True)
    .reset_index()
)

grouped.to_csv(out / "grouped_summary.csv", index=False)

#print("[OK] analysis complete")
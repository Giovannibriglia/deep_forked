
import pandas as pd
from pathlib import Path

df = pd.read_csv("combined_results/aggregate.csv")

df["TotalExecutionTime_mean"] = pd.to_numeric(df["TotalExecutionTime_mean"], errors='coerce')
df["NodesExpanded_mean"] = pd.to_numeric(df["NodesExpanded_mean"], errors='coerce')

def dominates(a,b):
    return (a["TotalExecutionTime_mean"]<=b["TotalExecutionTime_mean"] and
            a["NodesExpanded_mean"]<=b["NodesExpanded_mean"] and
            (a["TotalExecutionTime_mean"]<b["TotalExecutionTime_mean"] or
             a["NodesExpanded_mean"]<b["NodesExpanded_mean"]))

pareto=[]
for i,a in df.iterrows():
    if not any(dominates(b,a) for j,b in df.iterrows() if i!=j):
        pareto.append(a)

out=Path("combined_results/analysis")
out.mkdir(exist_ok=True)

pd.DataFrame(pareto).to_csv(out/"pareto.csv",index=False)

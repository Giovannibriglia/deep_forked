# RL Handler

## 0) Overall Idea
`rl_handler` is an offline RL frontier selector: given one frontier (a set of successor graphs from the same predecessor), it assigns a score to each candidate and picks the best one.

`[FIGURE PLACEHOLDER: pipeline from CSV/DOT -> graph composition -> GNN -> logits -> selected action]`

The model is trained to maximize expected reward over each frontier, where rewards are derived from distance-to-goal values.

## 1) Pre-processing
1. Read CSVs (`File Path`, `Depth`, `Distance From Goal`, `File Path Predecessor`, and `Goal` only for `separated` mode).
2. Group by predecessor to build frontiers, remove singleton frontiers, remove all-failure frontiers, and reject ambiguous goal mappings in `separated`.
3. Load DOT graphs into PyG tensors (`node_features`, `edge_index`, `edge_attr`) via `graph_utils.load_pyg_graph`.
4. Compose each frontier with `combine_graphs`:
   `merged`: merge successors by shared node identity (goal is assumed already embedded in each successor graph).
   `separated`: keep successors disconnected and load goal graph as separate tensors.
5. Build sample tensors:
   candidate rewards (`reward_target`), failures (`is_failure`), oracle (`oracle_index`, `oracle_reward`), and candidate mapping (`action_map`).
6. Collate batches with frontier-aware indexing (`candidate_batch`, `frontier_ptr`) plus optional goal tensors and merged pooling tensors.

## 2) Algorithm
1. Encode nodes with a GNN encoder (`GINE`/`RGCN`/`GCN`) using node and edge categorical embeddings.
2. Pool node embeddings to one embedding per candidate graph.
3. If `use_global_context=true`, concatenate each candidate embedding with the mean embedding of its frontier.
4. If `use_goal_separate_input=true` and goal tensors are present, concatenate pooled goal embedding as an extra branch.
5. Pass through an MLP head to get one scalar per candidate (`logit`).
6. Training loss maximizes frontier expected reward:
   `p_i = softmax(logits_i)` within each frontier, `loss = - mean_frontier(sum_i p_i * reward_i)`.

## 3) What It Returns (Logits and Context Semantics)
- Forward output is a 1D tensor of logits: one raw score per candidate.
- A higher logit means stronger preference before normalization.
- Selection is `argmax(logits)` (or `argmax(softmax(logits))`, equivalent).
- With `use_global_context=true`, each candidate score depends on all candidates in the same frontier (because the context term is a frontier mean).
- With `use_global_context=false`, that explicit frontier-level coupling is removed.
- Context aggregation is order-independent (mean pooling is permutation-invariant), but it is graph-dependent: changing candidate graphs (or removing one candidate) changes context and can change all logits.

## 4) ONNX Usage (Merged vs Separated)
Export:
1. `merged`: `python __main__.py --kind-of-data merged --export-onnx true`
2. `separated`: `python __main__.py --kind-of-data separated --use-goal-separate-input true --export-onnx true`

How to combine a frontier of three successor graphs (`s1`, `s2`, `s3`):

```python
from src.graph_utils import load_pyg_graph, combine_graphs
import numpy as np

g1 = load_pyg_graph("s1.dot", dataset_type="HASHED")
g2 = load_pyg_graph("s2.dot", dataset_type="HASHED")
g3 = load_pyg_graph("s3.dot", dataset_type="HASHED")
n_candidates = 3
mask = np.array([True, True, True], dtype=bool)

# merged regime (goal already embedded in successors)
merged = combine_graphs(
    frontier_graphs=[g1, g2, g3],
    goal_graph=None,
    kind_of_data="merged",
    action_ids=[0, 1, 2],
)

onnx_inputs_merged = {
    "node_features": merged.node_features.numpy().astype("float32"),
    "edge_index": merged.edge_index.numpy().astype("int64"),
    "edge_attr": merged.edge_attr.numpy().astype("int64"),
    "membership": merged.membership.numpy().astype("int64"),
    "mask": mask,  # len == number of valid candidates
}

# separated regime (goal passed as separate branch)
goal = load_pyg_graph("goal.dot", dataset_type="HASHED")
separated = combine_graphs(
    frontier_graphs=[g1, g2, g3],
    goal_graph=goal,
    kind_of_data="separated",
    action_ids=[0, 1, 2],
)

onnx_inputs_separated = {
    "node_features": separated.node_features.numpy().astype("float32"),
    "edge_index": separated.edge_index.numpy().astype("int64"),
    "edge_attr": separated.edge_attr.numpy().astype("int64"),
    "membership": separated.membership.numpy().astype("int64"),
    "goal_node_features": goal.node_features.numpy().astype("float32"),
    "goal_edge_index": goal.edge_index.numpy().astype("int64"),
    "goal_edge_attr": goal.edge_attr.numpy().astype("int64"),
    "goal_batch": np.zeros((goal.node_features.size(0),), dtype="int64"),
    "mask": mask,
}
```

What `combine_graphs` returns:
- Common fields (both regimes): `node_features`, `edge_index`, `edge_attr`, `membership`, `action_map`.
- `merged` may also return `pool_node_index` and `pool_membership` (used in PyTorch batching to preserve candidate pooling when nodes are shared).
- `separated` requires the external goal tensors (`goal_*`) when ONNX model was exported with `use_goal_separate_input=true`.

ONNX expected input structure:
- `merged`: `node_features [N,D] float32`, `edge_index [2,E] int64`, `edge_attr [E,1] int64`, `membership [N] int64`, `mask [F] bool`.
- `separated`: same + `goal_node_features [GN,D] float32`, `goal_edge_index [2,GE] int64`, `goal_edge_attr [GE,1] int64`, `goal_batch [GN] int64`.

Output usage:
1. Run ONNX and read `logits`.
2. Use the valid prefix `logits[:n_candidates]`.
3. Pick `argmax` on that prefix.

Note: in `merged`, if heavy node-sharing exists, ONNX output sizing can depend on traced/static assumptions. The project runtime handles this by probing mask lengths and still selecting from the valid prefix.

## 5) Parser (`__main__.py::parse_args`)
Complete parameter reference (default + effect):

### Paths and Naming
| Parameter | Default | Effect |
| --- | --- | --- |
| `--subset-train` | `[]` | Restrict training to specific dataset subfolders. |
| `--folder-raw-data` | `out/NN/Training` | Root containing raw CSV/DOT training data. |
| `--dir-save-data` | `data` | Output root for materialized `.pt` samples. |
| `--dir-save-model` | `models` | Output root for checkpoints, metrics, ONNX. |
| `--experiment-name` | `""` | Optional subdirectory under save roots (run isolation). |
| `--model-name` | `frontier_policy` | Base filename for model/ONNX/report artifacts. |

### Data Semantics and Dataset Build
| Parameter | Default | Effect |
| --- | --- | --- |
| `--dataset_type` | `HASHED` | Node encoding mode: `HASHED`, `MAPPED`, `BITMASK`. |
| `--kind-of-data` | `merged` | Frontier composition mode: `merged` or `separated`. |
| `--n-max-dataset-queries` | `1000` | Max random/stress eval queries generated per dataset. |
| `--max-size-frontier` | `25` | Max frontier size in stress FIFO/LIFO scheduling. |
| `--max-failure-states-per-dataset` | `0.3` | Caps train frontiers with failures (ratio to no-failure train frontiers). |
| `--reward-formulation` | `negative_distance` | Reward mode label (pipeline currently uses distance-derived rewards). |
| `--max-regular-distance-for-reward` | `50.0` | Distance threshold used for reward scaling and failure cut. |
| `--failure-reward-value` | `-1.0` | Reward assigned to failure states (`distance > max_regular_distance`). |
| `--build-data` | `true` | Rebuild materialized train/eval samples from raw data. |
| `--build-eval-data` | `true` | Rebuild random + stress eval sets (used when `--evaluate true`). |

### Training and Optimization
| Parameter | Default | Effect |
| --- | --- | --- |
| `--batch-size` | `9092` | Dataloader batch size (number of frontiers per step). |
| `--n-train-epochs` | `200` | Number of training epochs. |
| `--eval-every` | `20` | Evaluate every N epochs during training. |
| `--num-workers` | `0` | DataLoader worker processes. |
| `--seed` | `42` | RNG seed for split/build/sampling reproducibility. |
| `--lr` | `1e-3` | AdamW learning rate. |
| `--weight-decay` | `0.0` | AdamW weight decay. |
| `--max-grad-norm` | `0.0` | Gradient clipping threshold (`>0` enables clipping). |
| `--early-stopping-patience-evals` | `0` | Early stop after N eval checkpoints without reward improvement (`0` disables). |

### Model Architecture
| Parameter | Default | Effect |
| --- | --- | --- |
| `--gnn-layers` | `3` | Number of message-passing layers in encoder. |
| `--hidden-dim` | `128` | Hidden size for encoder/head. |
| `--conv-type` | `gine` | GNN layer family: `gine`, `rgcn`, `gcn`. |
| `--pooling-type` | `mean` | Node-to-candidate pooling: `mean`, `sum`, `max`. |
| `--edge-emb-dim` | `32` | Edge-label embedding size. |
| `--num-node-labels` | `4096` | Size of node label embedding table. |
| `--use-global-context` | `true` | Concatenate frontier context (mean candidate embedding) in policy head. |
| `--mlp-depth` | `2` | Depth of policy MLP head. |
| `--use-goal-separate-input` | `None` | Enables separate goal branch; if unset: auto `true` for `separated`, else `false`. |

### Execution, Evaluation, ONNX
| Parameter | Default | Effect |
| --- | --- | --- |
| `--train` | `true` | Run training loop. |
| `--evaluate` | `true` | Run evaluation on available eval splits. |
| `--export-onnx` | `true` | Export ONNX model after train/load. |
| `--if-try-example` | `false` | Run PyTorch-vs-ONNX parity checks and frontier order/removal checks. |
| `--onnx-frontier-check-size` | `5` | Candidate count used by ONNX order/removal diagnostic check. |

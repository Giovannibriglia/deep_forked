from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pydot
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

DATASET_TYPE_HASHED = "HASHED"
DATASET_TYPE_MAPPED = "MAPPED"
DATASET_TYPE_BITMASK = "BITMASK"
VALID_DATASET_TYPES = {
    DATASET_TYPE_HASHED,
    DATASET_TYPE_MAPPED,
    DATASET_TYPE_BITMASK,
}
TWO_48_MINUS_1 = float(2**48 - 1)


@dataclass
class CombinedFrontierGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    membership: torch.Tensor
    action_map: torch.Tensor
    pool_node_index: Optional[torch.Tensor] = None
    pool_membership: Optional[torch.Tensor] = None


def _load_dot(path: Path) -> nx.DiGraph:
    src = path.read_text()
    dot = pydot.graph_from_dot_data(src)[0]
    return nx.nx_pydot.from_pydot(dot)


def _normalize_dataset_type(
    dataset_type: Optional[str] = None,
    bitmask: Optional[bool] = None,
) -> str:
    if dataset_type is not None:
        dt = str(dataset_type).upper()
        if dt not in VALID_DATASET_TYPES:
            raise ValueError(
                f"Unsupported dataset_type '{dataset_type}'. "
                f"Expected one of {sorted(VALID_DATASET_TYPES)}."
            )
        return dt
    if bool(bitmask):
        return DATASET_TYPE_BITMASK
    return DATASET_TYPE_HASHED


def _parse_numeric_node_label(node_obj: object) -> float:
    if isinstance(node_obj, (int, float)):
        return float(node_obj)
    if isinstance(node_obj, str):
        return float(node_obj.strip())
    raise TypeError(f"Unsupported node label type: {type(node_obj)}")


def _nx_to_pyg(G: nx.DiGraph, dataset_type: str) -> Data:
    for _, data in G.nodes(data=True):
        data["shape"] = {"circle": 0, "doublecircle": 1}.get(
            data.get("shape", "circle"), 0
        )
    for _, _, d in G.edges(data=True):
        d["edge_label"] = int(str(d.get("label", "0")).replace('"', ""))

    data = from_networkx(G)
    data.edge_index = data.edge_index.long()
    data.edge_attr = data.edge_label.view(-1, 1).to(torch.int64)

    if dataset_type == DATASET_TYPE_BITMASK:
        nodes = list(G.nodes())

        def _to_bits(node_obj: object, bit_len: int | None) -> list[int]:
            if isinstance(node_obj, str):
                s = node_obj.strip()
                if not set(s) <= {"0", "1"}:
                    raise ValueError(f"Node '{node_obj}' is not a bitstring.")
                if bit_len is not None and len(s) != bit_len:
                    raise ValueError(f"Inconsistent bit length for node '{node_obj}'.")
                return [int(ch) for ch in s]
            if isinstance(node_obj, (list, tuple)):
                bits = [int(x) for x in node_obj]
                if not set(bits) <= {0, 1}:
                    raise ValueError(f"Node '{node_obj}' has non-binary values.")
                if bit_len is not None and len(bits) != bit_len:
                    raise ValueError(f"Inconsistent bit length for node '{node_obj}'.")
                return bits
            if isinstance(node_obj, int):
                if bit_len is None:
                    raise ValueError("bit_len is required for int node labels.")
                return [int(ch) for ch in format(node_obj, f"0{bit_len}b")]
            raise TypeError(f"Unsupported node label type: {type(node_obj)}")

        explicit_len = G.graph.get("bit_len", None)
        first = nodes[0]
        inferred_len = len(first) if isinstance(first, (str, list, tuple)) else None
        bit_len = explicit_len if explicit_len is not None else inferred_len
        if bit_len is None:
            raise ValueError("Cannot infer bit length from graph nodes.")

        rows = [_to_bits(n, bit_len) for n in nodes]
        data.node_bits = torch.tensor(rows, dtype=torch.bool)
        data.node_names = data.node_bits.to(torch.float32)
        data.node_features = data.node_bits.to(torch.float32)
    else:
        raw_ids = [_parse_numeric_node_label(n) for n in G.nodes()]
        data.node_names = torch.tensor(raw_ids, dtype=torch.float32)
        if dataset_type == DATASET_TYPE_HASHED:
            # Keep raw IDs; HASHED normalization is handled in-model.
            node_feats = data.node_names
        elif dataset_type == DATASET_TYPE_MAPPED:
            node_feats = data.node_names
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")
        data.node_features = node_feats.view(-1, 1).to(torch.float32)

    return data


def load_pyg_graph(
    path: str,
    dataset_type: Optional[str] = None,
    bitmask: Optional[bool] = None,
) -> Data:
    ds_type = _normalize_dataset_type(dataset_type=dataset_type, bitmask=bitmask)
    g = _load_dot(Path(path))
    return _nx_to_pyg(g, dataset_type=ds_type)


def _compose_disconnected(
    graphs: List[Data],
    membership_values: List[int],
) -> Dict[str, torch.Tensor]:
    node_parts: List[torch.Tensor] = []
    edge_index_parts: List[torch.Tensor] = []
    edge_attr_parts: List[torch.Tensor] = []
    membership_parts: List[torch.Tensor] = []

    node_offset = 0
    for graph, member_value in zip(graphs, membership_values):
        n_nodes = int(graph.node_features.size(0))
        node_parts.append(graph.node_features)
        membership_parts.append(
            torch.full((n_nodes,), member_value, dtype=torch.long)
        )

        if graph.edge_index.numel() > 0:
            edge_index_parts.append(graph.edge_index + node_offset)
            edge_attr_parts.append(graph.edge_attr.to(torch.int64))

        node_offset += n_nodes

    if not node_parts:
        raise ValueError("Cannot compose an empty list of graphs.")

    node_features = torch.cat(node_parts, dim=0)
    membership = torch.cat(membership_parts, dim=0)
    edge_index = (
        torch.cat(edge_index_parts, dim=1)
        if edge_index_parts
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.cat(edge_attr_parts, dim=0)
        if edge_attr_parts
        else torch.zeros((0, 1), dtype=torch.int64)
    )
    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "membership": membership,
    }


def _node_keys(graph: Data) -> List[Tuple[float, ...]]:
    names = getattr(graph, "node_names", None)
    if names is None:
        names = graph.node_features
    names = names.detach().cpu()
    if names.dim() == 1:
        return [(float(v.item()),) for v in names]
    return [tuple(float(x.item()) for x in row) for row in names]


def _edge_attr_tuple(edge_attr: torch.Tensor, idx: int) -> Tuple[int, ...]:
    row = edge_attr[idx].detach().cpu().view(-1)
    return tuple(int(v.item()) for v in row)


def _compose_merged_shared_goal(graphs: List[Data]) -> Dict[str, torch.Tensor]:
    # Nodes are merged by stable identity (node_names when available). If two
    # graphs share central goal nodes, they map to the same merged node.
    node_features_rows: List[torch.Tensor] = []
    key_to_index: Dict[Tuple[float, ...], int] = {}
    # Membership keeps one owner candidate per merged node to stay compatible
    # with the existing model input contract (single membership tensor).
    owner_membership: List[int] = []
    pool_node_index: List[int] = []
    pool_membership: List[int] = []
    edge_rows: List[List[int]] = []
    edge_attr_rows: List[torch.Tensor] = []
    edge_seen = set()

    for member_value, graph in enumerate(graphs):
        keys = _node_keys(graph)
        local_to_global: List[int] = []

        for local_idx, key in enumerate(keys):
            g_idx = key_to_index.get(key)
            if g_idx is None:
                g_idx = len(node_features_rows)
                key_to_index[key] = g_idx
                node_features_rows.append(graph.node_features[local_idx].detach().clone())
                owner_membership.append(member_value)
            local_to_global.append(g_idx)
            pool_node_index.append(g_idx)
            pool_membership.append(member_value)

        if graph.edge_index.numel() == 0:
            continue
        edge_index = graph.edge_index.detach().cpu()
        for e_idx in range(edge_index.size(1)):
            src_local = int(edge_index[0, e_idx].item())
            dst_local = int(edge_index[1, e_idx].item())
            src = local_to_global[src_local]
            dst = local_to_global[dst_local]
            attr_key = _edge_attr_tuple(graph.edge_attr, e_idx)
            edge_key = (src, dst, attr_key)
            if edge_key in edge_seen:
                continue
            edge_seen.add(edge_key)
            edge_rows.append([src, dst])
            edge_attr_rows.append(graph.edge_attr[e_idx].detach().clone().to(torch.int64))

    if not node_features_rows:
        raise ValueError("Cannot compose an empty list of graphs.")

    node_features = torch.stack(node_features_rows, dim=0)
    membership = torch.tensor(owner_membership, dtype=torch.long)
    if edge_rows:
        edge_index = torch.tensor(edge_rows, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr_rows, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.int64)

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "membership": membership,
        "pool_node_index": torch.tensor(pool_node_index, dtype=torch.long),
        "pool_membership": torch.tensor(pool_membership, dtype=torch.long),
    }


def _compose_separated_disconnected(graphs: List[Data]) -> Dict[str, torch.Tensor]:
    return _compose_disconnected(
        graphs=graphs,
        membership_values=list(range(len(graphs))),
    )


def combine_graphs(
    frontier_graphs: List[Data],
    goal_graph: Optional[Data],
    kind_of_data: str,
    action_ids: Optional[List[int]] = None,
) -> CombinedFrontierGraph:
    if not frontier_graphs:
        raise ValueError("Frontier cannot be empty.")

    n_frontier = len(frontier_graphs)
    if action_ids is None:
        action_tensor = torch.arange(n_frontier, dtype=torch.long)
    else:
        action_tensor = torch.tensor(action_ids, dtype=torch.long)

    if kind_of_data not in {"merged", "separated"}:
        raise ValueError(f"Unsupported kind_of_data: {kind_of_data}")

    if kind_of_data == "merged":
        # Goal is embedded in successors; compose into one graph by shared node/edge
        # identity. If no overlap exists, this naturally falls back to concatenation.
        out = _compose_merged_shared_goal(frontier_graphs)
        return CombinedFrontierGraph(
            node_features=out["node_features"],
            edge_index=out["edge_index"],
            edge_attr=out["edge_attr"],
            membership=out["membership"],
            action_map=action_tensor,
            pool_node_index=out["pool_node_index"],
            pool_membership=out["pool_membership"],
        )

    if goal_graph is None:
        raise ValueError(
            "Goal graph must be loaded from dataset Goal path for kind_of_data='separated'."
        )

    # Separated semantics: keep frontier candidates disconnected and pass goal
    # as a separate graph branch to the model.
    out = _compose_separated_disconnected(frontier_graphs)
    return CombinedFrontierGraph(
        node_features=out["node_features"],
        edge_index=out["edge_index"],
        edge_attr=out["edge_attr"],
        membership=out["membership"],
        action_map=action_tensor,
    )

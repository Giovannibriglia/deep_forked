from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import pydot
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


@dataclass
class CombinedFrontierGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    membership: torch.Tensor
    action_map: torch.Tensor


def _load_dot(path: Path) -> nx.DiGraph:
    src = path.read_text()
    dot = pydot.graph_from_dot_data(src)[0]
    return nx.nx_pydot.from_pydot(dot)


def _nx_to_pyg(G: nx.DiGraph, bitmask: bool = False) -> Data:
    for _, data in G.nodes(data=True):
        data["shape"] = {"circle": 0, "doublecircle": 1}.get(
            data.get("shape", "circle"), 0
        )
    for _, _, d in G.edges(data=True):
        d["edge_label"] = int(str(d.get("label", "0")).replace('"', ""))

    data = from_networkx(G)
    data.edge_index = data.edge_index.long()
    data.edge_attr = data.edge_label.view(-1, 1).float()

    if bitmask:
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
    else:
        raw_ids = [int(n) for n in G.nodes()]
        data.node_names = torch.tensor(raw_ids, dtype=torch.float32)

    return data


def load_pyg_graph(path: str, bitmask: bool = False) -> Data:
    g = _load_dot(Path(path))
    data = _nx_to_pyg(g, bitmask=bitmask)
    if bitmask:
        data.node_features = data.node_bits.to(torch.float32)
    else:
        data.node_features = data.node_names.view(-1, 1).to(torch.float32)
    return data


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
            edge_attr_parts.append(graph.edge_attr.to(torch.float32))

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
        else torch.zeros((0, 1), dtype=torch.float32)
    )
    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "membership": membership,
    }


def combine_graphs(
    frontier_graphs: List[Data],
    goal_graph: Data,
    kind_of_data: str,
    action_ids: Optional[List[int]] = None,
) -> CombinedFrontierGraph:
    if not frontier_graphs:
        raise ValueError("Frontier cannot be empty.")
    if goal_graph is None:
        raise ValueError("Goal graph must be loaded from dataset Goal path.")

    n_frontier = len(frontier_graphs)
    if action_ids is None:
        action_tensor = torch.arange(n_frontier, dtype=torch.long)
    else:
        action_tensor = torch.tensor(action_ids, dtype=torch.long)

    if kind_of_data not in {"merged", "separated"}:
        raise ValueError(f"Unsupported kind_of_data: {kind_of_data}")

    if kind_of_data == "merged":
        out = _compose_disconnected(
            graphs=frontier_graphs,
            membership_values=list(range(n_frontier)),
        )
        return CombinedFrontierGraph(
            node_features=out["node_features"],
            edge_index=out["edge_index"],
            edge_attr=out["edge_attr"],
            membership=out["membership"],
            action_map=action_tensor,
        )

    # separated: include goal graph explicitly and minimally link each frontier graph to goal
    all_graphs = frontier_graphs + [goal_graph]
    membership_values = list(range(n_frontier)) + [-1]
    out = _compose_disconnected(
        graphs=all_graphs,
        membership_values=membership_values,
    )

    n_goal_nodes = int(goal_graph.node_features.size(0))
    goal_start = int(out["node_features"].size(0) - n_goal_nodes)
    goal_anchor = goal_start

    root_indices = []
    cursor = 0
    for g in frontier_graphs:
        root_indices.append(cursor)
        cursor += int(g.node_features.size(0))

    link_edges = []
    for src in root_indices:
        link_edges.append([src, goal_anchor])
        link_edges.append([goal_anchor, src])
    link_edge_index = torch.tensor(link_edges, dtype=torch.long).t().contiguous()
    link_edge_attr = torch.zeros((link_edge_index.size(1), 1), dtype=torch.float32)

    edge_index = (
        torch.cat([out["edge_index"], link_edge_index], dim=1)
        if out["edge_index"].numel() > 0
        else link_edge_index
    )
    edge_attr = torch.cat([out["edge_attr"], link_edge_attr], dim=0)

    return CombinedFrontierGraph(
        node_features=out["node_features"],
        edge_index=edge_index,
        edge_attr=edge_attr,
        membership=out["membership"],
        action_map=action_tensor,
    )

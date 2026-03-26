from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    GCNConv,
    GINEConv,
    RGCNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


def _build_mlp(in_dim: int, hidden_dim: int, depth: int, out_dim: int) -> nn.Sequential:
    layers = []
    d_in = in_dim
    for _ in range(max(1, depth)):
        layers.append(nn.Linear(d_in, hidden_dim))
        layers.append(nn.ReLU())
        d_in = hidden_dim
    layers.append(nn.Linear(d_in, out_dim))
    return nn.Sequential(*layers)


class GNNEncoder(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        conv_type: str = "gine",
    ):
        super().__init__()
        self.conv_type = conv_type.lower()
        self.input_proj = nn.Linear(node_input_dim, hidden_dim)
        self.edge_proj = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            if self.conv_type == "gine":
                self.layers.append(
                    GINEConv(
                        nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                        ),
                        edge_dim=hidden_dim,
                    )
                )
            elif self.conv_type == "rgcn":
                self.layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations=64))
            elif self.conv_type == "gcn":
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_proj(node_features.to(torch.float32))

        for conv in self.layers:
            if isinstance(conv, GINEConv):
                e = self.edge_proj(edge_attr.to(torch.float32))
                x = F.relu(conv(x, edge_index, e))
            elif isinstance(conv, RGCNConv):
                edge_type = edge_attr.view(-1).to(torch.long).clamp(min=0, max=63)
                x = F.relu(conv(x, edge_index, edge_type))
            else:
                x = F.relu(conv(x, edge_index))
        return x


class FrontierPolicyNetwork(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int = 128,
        gnn_layers: int = 3,
        conv_type: str = "gine",
        pooling_type: str = "mean",
        use_global_context: bool = True,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.use_global_context = use_global_context
        self.pooling_type = pooling_type
        self.encoder = GNNEncoder(
            node_input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            conv_type=conv_type,
        )
        head_in = hidden_dim * 2 if use_global_context else hidden_dim
        self.policy_head = _build_mlp(
            in_dim=head_in,
            hidden_dim=hidden_dim,
            depth=mlp_depth,
            out_dim=1,
        )

    def _pool_nodes(
        self,
        node_embeddings: torch.Tensor,
        membership: torch.Tensor,
    ) -> torch.Tensor:
        mask = membership >= 0
        x = node_embeddings[mask]
        m = membership[mask]
        if self.pooling_type == "mean":
            return global_mean_pool(x, m)
        if self.pooling_type == "sum":
            return global_add_pool(x, m)
        if self.pooling_type == "max":
            return global_max_pool(x, m)
        raise ValueError(f"Unsupported pooling_type: {self.pooling_type}")

    def _contextualize(
        self,
        z: torch.Tensor,
        candidate_batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_global_context:
            return z
        if candidate_batch is None:
            ctx = z.mean(dim=0, keepdim=True).expand_as(z)
            return torch.cat([z, ctx], dim=-1)
        ctx_per_frontier = global_mean_pool(z, candidate_batch)
        ctx = ctx_per_frontier[candidate_batch]
        return torch.cat([z, ctx], dim=-1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        membership: torch.Tensor,
        candidate_batch: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        node_emb = self.encoder(node_features, edge_index, edge_attr)
        z = self._pool_nodes(node_emb, membership)
        h = self._contextualize(z, candidate_batch)
        logits = self.policy_head(h).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.to(torch.bool), -1e9)
        return logits

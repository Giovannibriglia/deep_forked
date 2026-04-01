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
        use_goal_separate_input: bool = False,
    ):
        super().__init__()
        self.use_global_context = use_global_context
        self.pooling_type = pooling_type
        self.use_goal_separate_input = use_goal_separate_input
        self.encoder = GNNEncoder(
            node_input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            conv_type=conv_type,
        )
        head_in = hidden_dim * 2 if use_global_context else hidden_dim
        if self.use_goal_separate_input:
            head_in += hidden_dim
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
        pool_node_index: Optional[torch.Tensor] = None,
        pool_membership: Optional[torch.Tensor] = None,
        expected_size: Optional[int] = None,
    ) -> torch.Tensor:
        if membership.dim() != 1:
            raise ValueError(f"membership must be 1D, got shape {tuple(membership.shape)}")
        if expected_size is not None and int(expected_size) < 0:
            raise ValueError(f"expected_size must be non-negative, got {expected_size}")

        if (pool_node_index is None) != (pool_membership is None):
            raise ValueError(
                "pool_node_index and pool_membership must be provided together."
            )
        if pool_node_index is not None and pool_membership is not None:
            if pool_node_index.dim() != 1 or pool_membership.dim() != 1:
                raise ValueError(
                    "pool_node_index and pool_membership must be 1D tensors."
                )
            if pool_node_index.numel() != pool_membership.numel():
                raise ValueError(
                    "pool_node_index and pool_membership must have identical lengths."
                )
            if pool_node_index.numel() > 0:
                if int(pool_node_index.min().item()) < 0:
                    raise ValueError("pool_node_index contains negative indices.")
                if int(pool_node_index.max().item()) >= int(node_embeddings.size(0)):
                    raise ValueError(
                        f"pool_node_index out of bounds: max={int(pool_node_index.max().item())} "
                        f"n_nodes={int(node_embeddings.size(0))}"
                    )
            if pool_membership.numel() > 0 and int(pool_membership.min().item()) < 0:
                raise ValueError("pool_membership contains negative indices.")
            x = node_embeddings[pool_node_index]
            m = pool_membership
        else:
            mask = membership >= 0
            x = node_embeddings[mask]
            m = membership[mask]
        if m.dim() != 1:
            raise ValueError(f"pool membership vector must be 1D, got shape {tuple(m.shape)}")
        if x.size(0) != m.numel():
            raise ValueError(
                f"Pooled tensor and membership mismatch: x.size(0)={x.size(0)} m.numel()={m.numel()}"
            )

        size = int(expected_size) if expected_size is not None else None
        if self.pooling_type == "mean":
            pooled = global_mean_pool(x, m, size=size)
        elif self.pooling_type == "sum":
            pooled = global_add_pool(x, m, size=size)
        elif self.pooling_type == "max":
            pooled = global_max_pool(x, m, size=size)
        else:
            raise ValueError(f"Unsupported pooling_type: {self.pooling_type}")
        if expected_size is not None and pooled.size(0) != int(expected_size):
            raise AssertionError(
                f"Pooled candidate size mismatch: pooled.size(0)={pooled.size(0)} "
                f"expected_size={int(expected_size)}"
            )
        return pooled

    def _pool_goal(
        self,
        goal_node_embeddings: torch.Tensor,
        goal_batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling_type == "mean":
            return global_mean_pool(goal_node_embeddings, goal_batch)
        if self.pooling_type == "sum":
            return global_add_pool(goal_node_embeddings, goal_batch)
        if self.pooling_type == "max":
            return global_max_pool(goal_node_embeddings, goal_batch)
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
        if candidate_batch.dim() != 1:
            raise ValueError(
                f"candidate_batch must be 1D, got shape {tuple(candidate_batch.shape)}"
            )
        if candidate_batch.numel() > 0 and int(candidate_batch.min().item()) < 0:
            raise ValueError("candidate_batch contains negative frontier indices.")

        assert z.size(0) == candidate_batch.numel(), (
            f"Mismatch before global_mean_pool: "
            f"z.size(0)={z.size(0)} candidate_batch.numel()={candidate_batch.numel()} "
            f"candidate_batch.min()={int(candidate_batch.min()) if candidate_batch.numel() > 0 else 'NA'} "
            f"candidate_batch.max()={int(candidate_batch.max()) if candidate_batch.numel() > 0 else 'NA'}"
        )

        if candidate_batch.numel() == 0:
            ctx = z.new_zeros(z.shape)
            return torch.cat([z, ctx], dim=-1)
        n_frontiers = int(candidate_batch.max().item()) + 1
        ctx_per_frontier = global_mean_pool(z, candidate_batch, size=n_frontiers)
        if int(candidate_batch.max().item()) >= ctx_per_frontier.size(0):
            raise ValueError(
                f"candidate_batch index out of range: max={int(candidate_batch.max().item())} "
                f"n_frontiers={ctx_per_frontier.size(0)}"
            )
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
        goal_node_features: Optional[torch.Tensor] = None,
        goal_edge_index: Optional[torch.Tensor] = None,
        goal_edge_attr: Optional[torch.Tensor] = None,
        goal_batch: Optional[torch.Tensor] = None,
        pool_node_index: Optional[torch.Tensor] = None,
        pool_membership: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        node_emb = self.encoder(node_features, edge_index, edge_attr)
        expected_num_candidates = (
            int(candidate_batch.numel()) if candidate_batch is not None else None
        )
        z = self._pool_nodes(
            node_emb,
            membership,
            pool_node_index=pool_node_index,
            pool_membership=pool_membership,
            expected_size=expected_num_candidates,
        )
        h = self._contextualize(z, candidate_batch)
        if self.use_goal_separate_input:
            if (
                goal_node_features is not None
                and goal_edge_index is not None
                and goal_edge_attr is not None
                and goal_batch is not None
            ):
                goal_node_emb = self.encoder(
                    goal_node_features,
                    goal_edge_index,
                    goal_edge_attr,
                )
                goal_emb = self._pool_goal(goal_node_emb, goal_batch)
                if candidate_batch is None:
                    goal_per_candidate = goal_emb.mean(dim=0, keepdim=True).expand(h.size(0), -1)
                else:
                    goal_per_candidate = goal_emb[candidate_batch]
            else:
                goal_per_candidate = torch.zeros(
                    (h.size(0), self.encoder.input_proj.out_features),
                    device=h.device,
                    dtype=h.dtype,
                )
            h = torch.cat([h, goal_per_candidate], dim=-1)
        logits = self.policy_head(h).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.to(torch.bool), -1e9)
        return logits

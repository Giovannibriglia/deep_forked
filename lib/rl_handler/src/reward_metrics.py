from __future__ import annotations

import math

import torch

EPS = 1e-12


def masked_softmax(x: torch.Tensor, mask: torch.Tensor, tau: float) -> torch.Tensor:
    x = x / tau
    x = x.masked_fill(~mask, -1e9)
    x_max = x.max(dim=1, keepdim=True).values
    exp_x = torch.exp(x - x_max) * mask
    denom = exp_x.sum(dim=1, keepdim=True).clamp_min(EPS)
    return exp_x / denom


def kl_divergence(p: torch.Tensor, q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(EPS)
    q = q.clamp_min(EPS)
    return ((p * (torch.log(p) - torch.log(q))) * mask).sum(dim=1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, mask) + 0.5 * kl_divergence(q, m, mask)


def js_normalized(p: torch.Tensor, q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return js_divergence(p, q, mask) / math.log(2.0)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs() * mask
    return diff.sum() / mask.sum().clamp_min(1)


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = ((pred - target) ** 2) * mask
    return torch.sqrt(diff.sum() / mask.sum().clamp_min(1))


def score_std_within_frontier(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vals = []
    for b in range(scores.shape[0]):
        s = scores[b][mask[b]]
        if s.numel() > 1:
            vals.append(s.std(unbiased=False))
    if not vals:
        return torch.tensor(0.0, device=scores.device)
    return torch.stack(vals).mean()


def pairwise_ranking_loss(
    scores: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    loss = scores.new_tensor(0.0)
    count = 0
    batch_size = scores.shape[0]

    for b in range(batch_size):
        idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
        for i in idx:
            for j in idx:
                if rewards[b, i] > rewards[b, j]:
                    loss = loss + torch.relu(1.0 - (scores[b, i] - scores[b, j]))
                    count += 1

    if count == 0:
        return scores.new_tensor(0.0)
    return loss / count


def listwise_loss(
    scores: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor, tau: float
) -> torch.Tensor:
    p_true = masked_softmax(rewards, mask, tau)
    p_pred = masked_softmax(scores, mask, tau)
    p_pred = p_pred.clamp_min(1e-12)
    ce = -((p_true * torch.log(p_pred)) * mask).sum(dim=1)
    return ce.mean()

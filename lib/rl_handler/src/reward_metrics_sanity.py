from __future__ import annotations

import torch

try:
    from src.reward_metrics import (
        js_normalized,
        listwise_loss,
        masked_softmax,
        pairwise_ranking_loss,
        score_std_within_frontier,
    )
except ModuleNotFoundError:
    from reward_metrics import (  # type: ignore[no-redef]
        js_normalized,
        listwise_loss,
        masked_softmax,
        pairwise_ranking_loss,
        score_std_within_frontier,
    )


def main() -> None:
    mask = torch.tensor([[True, True]], dtype=torch.bool)
    true_scores = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    pred_scores = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    p_true = masked_softmax(true_scores, mask, tau=0.5)
    p_pred = masked_softmax(pred_scores, mask, tau=0.5)
    js_same = float(js_normalized(p_true, p_pred, mask).item())
    assert abs(js_same) < 1e-6, f"Expected JS normalized ~0, got {js_same}"

    true_scores = torch.tensor([[20.0, -20.0]], dtype=torch.float32)
    pred_scores = torch.tensor([[-20.0, 20.0]], dtype=torch.float32)
    p_true = masked_softmax(true_scores, mask, tau=1.0)
    p_pred = masked_softmax(pred_scores, mask, tau=1.0)
    js_opposite = float(js_normalized(p_true, p_pred, mask).item())
    assert 0.999 <= js_opposite <= 1.001, (
        f"Expected JS normalized ~1 for opposite binary distributions, got {js_opposite}"
    )

    scores = torch.tensor([[3.0, 3.0, 3.0]], dtype=torch.float32)
    mask3 = torch.tensor([[True, True, True]], dtype=torch.bool)
    std_val = float(score_std_within_frontier(scores, mask3).item())
    assert abs(std_val) < 1e-6, f"Expected score std ~0 for constant scores, got {std_val}"

    scores = torch.tensor(
        [[1.1, 0.2, -0.2], [0.3, -0.1, 0.0]],
        dtype=torch.float32,
    )
    rewards = torch.tensor(
        [[1.0, 0.3, -0.5], [0.4, -0.2, 0.0]],
        dtype=torch.float32,
    )
    mask_var = torch.tensor(
        [[True, True, True], [True, True, False]],
        dtype=torch.bool,
    )
    pairwise = pairwise_ranking_loss(scores, rewards, mask_var)
    listwise = listwise_loss(scores, rewards, mask_var, tau=0.5)
    assert pairwise.dim() == 0, "Pairwise ranking loss should be scalar."
    assert listwise.dim() == 0, "Listwise ranking loss should be scalar."
    assert torch.isfinite(pairwise), "Pairwise ranking loss should be finite."
    assert torch.isfinite(listwise), "Listwise ranking loss should be finite."

    print("reward_metrics_sanity: all checks passed.")


if __name__ == "__main__":
    main()

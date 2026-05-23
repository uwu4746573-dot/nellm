# [Resolved] Issue 1: Implement LatentRouterLayer

**Component:** `LatentRouterLayer`
**Objective:** Create the PyTorch module responsible for routing current facts to relevant transformations from a base.

## Specification
* **Input:** `F_current` tensor of shape `[B, D_v]`, where `D_v = 2048`.
* **Operations:**
  1. Project fact to a query vector: `Q = Linear(D_v -> D_k)(F_current)`, where `D_k = 128`.
  2. Compute cosine similarity logits against transformation keys `K_T` (shape `[N_T, D_k]`): `Logits = (Q @ K_T.T) / sqrt(D_k)`
  3. **Training (Soft mode):** Apply Gumbel-Softmax to logits to get `weights`. Extract `V_T_soft = weights @ V_T` (shape `[B, D_v]`).
  4. **Inference (Hard mode):** Select `W` (Beam width = 5) indices with top logits.
* **Output:** `V_T_selected` of shape `[B, W, D_v]` (Inference) or `[B, D_v]` (Training).

## Tasks
- [x] Define the `LatentRouterLayer` `nn.Module`.
- [x] Implement query projection linear layer.
- [x] Implement cosine similarity scaled dot-product.
- [x] Add Gumbel-Softmax routing for training.
- [x] Add Top-W hard selection for inference.
- [x] Add unit tests verifying output shapes for both modes.


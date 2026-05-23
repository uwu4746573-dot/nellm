# [Resolved] Issue 006: LatentDecoderLayer

**Description:**
Implement the LatentDecoderLayer based on PLaT/LaDiR concepts. This layer bridges our latent reasoning space back to the LLM. It takes the final reasoning vector $F_n$ and projects it into a "soft prompt" or "continuous prefix" that can be prepended to the LLM's input embeddings for answer generation.

**Requirements:**
- **File:** `src/models/decoder.py`
- **Class:** `LatentDecoderLayer(nn.Module)`
- **Input:** $F_n$ tensor of shape `[B, D_v]` (default `D_v=2048`).
- **Mechanism:**
  - We want to unroll the dense $F_n$ into a sequence of `prefix_len` tokens (e.g., `prefix_len=16`).
  - Projection: `Linear(D_v, prefix_len * D_model)` (where `D_model=4096`).
  - Reshape: Reshape the output to `[B, prefix_len, D_model]`.
  - Normalization: Apply `LayerNorm(D_model)` to stabilize the soft prompt.
- **Output:** Soft prompt tensor of shape `[B, prefix_len, D_model]`.
- **Tests:** Write unit tests in `tests/test_decoder.py`.

**Definition of Done:**
- [x] Code implemented
- [x] Tests passing
- [x] Shape transformations correct
- [x] Batch dimension handled correctly

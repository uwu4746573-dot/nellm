# [Resolved] Issue 005: LatentEncoderLayer

**Description:**
Implement the LatentEncoderLayer based on the NV-Embed-v2 Latent Attention Pooling approach. We assume the base LLM (e.g. Mistral) is external. This layer takes the LLM's raw hidden states and compresses them into our latent reasoning fact vector $F_1$.

**Requirements:**
- **File:** `src/models/encoder.py`
- **Class:** `LatentEncoderLayer(nn.Module)`
- **Input:** `hidden_states` tensor of shape `[B, SeqLen, D_model]` (default `D_model=4096`).
- **Mechanism (Latent Attention Pooling):**
  - Learnable latent queries: `self.latent_queries = nn.Parameter(torch.randn(num_latents, D_model))` (e.g., `num_latents=512`).
  - Cross-attention: The latent queries attend to the `hidden_states`. You can use `nn.MultiheadAttention` where queries are `latent_queries` (expanded to batch size), and keys/values are `hidden_states`.
  - Mean pooling: Average the output of the cross-attention over the `num_latents` dimension to get a single vector per batch item `[B, D_model]`.
- **Projection:** A linear layer `Linear(D_model, D_v)` (where `D_v=2048`) to project the pooled representation into our reasoning latent space.
- **Output:** $F_1$ tensor of shape `[B, D_v]`.
- **Tests:** Write unit tests in `tests/test_encoder.py`.

**Definition of Done:**
- [x] Code implemented
- [x] Tests passing
- [x] No NaNs/Infs
- [x] Batch dimension handled correctly

# Research Report: Encoding Reasoning & Abstract Thought into Dense Vectors (2024–2026)
*Compiled May 23, 2026*

## 🥇 #1 — PLaT: Planning with Latent Thoughts (Jan 2026)
- **Concept:** Reformulates latent reasoning as planning in a continuous manifold.
- **Mechanism:** Autoregressively evolves a deterministic trajectory of latent planning states.
- **Relevance to NeLL-M:** PLaT's "planner states" map exactly to our Fact vectors (F). The reasoning terminates dynamically and is only grounded into natural language at output time.

## 🥈 #2 — LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning (Oct 2025)
- **Concept:** Replaces autoregressive scratchpad with latent diffusion over compressed semantic thought tokens.
- **Mechanism:** A VAE encodes reasoning steps into compact latent thought tokens. A diffusion model iteratively refines them.
- **Relevance to NeLL-M:** The VAE encoder acts as a "what to think about" encoder. Iterative refinement maps to our T(F) transformations.

## 🥉 #3 — V-JEPA 2 / JEPA Family (Meta AI, 2025)
- **Concept:** Predicts outcomes in latent space rather than pixel/token space.
- **Mechanism:** Explicitly discards irrelevant surface details and retains semantic and causal structure.
- **Relevance to NeLL-M:** JEPA encoders are invariant to surface variation and sensitive to structural/causal patterns, perfect for generating F_1.

## Strategic Recommendation for NeLL-M
The optimal architecture for F_1 and the overall reasoning pipeline is a hybrid:
1. **Encoder backbone:** JEPA-style (predict abstract structure, invariant to phrasing).
2. **Refinement mechanism:** Iterative latent state refinement (similar to our MoE Synthesizer applying T to F).
3. **Training signal:** Preference optimization (RLHF) over the latent plan, rewarding F_n vectors that lead to correct final answers.

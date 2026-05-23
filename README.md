# NeLL-M — Neural Latent Language Model

A research implementation of **Latent Reasoning with a Dynamic K-V Logic Cache** — an architecture where reasoning happens in continuous vector space rather than via token prediction.

## Core Idea

Instead of autoregressive token generation, the model:
1. Encodes the query into a start fact vector **F₁**
2. Iteratively applies transformation rules **T** to facts **F** via beam search (W=5 branches)
3. Verifies each step with a Critic
4. On success, memorizes the shortcut **F₁ ⇒ Fₙ** as a new rule **T** in the K-V base

## Architecture

```
Query (text)
     │
     ▼
LatentEncoderLayer          → F₁ [B, 2048]
     │
     ▼  (beam search loop, W=5 branches)
LatentRouterLayer           → retrieves V_T [B, W, 2048] from T-Base
     │
     ▼
SynthesizerLayer (AdaLN+MoE) → F_new [B, 2048]   (T applied to F)
     │
     ▼
CriticLayer                 → halt_prob [B, 1]
     │
     ├── if done ──► MemorizerLayer → new K_T [B,128], V_T [B,2048]
     │                               written to T-Base
     └── else ──────► next beam step
     │
     ▼
LatentDecoderLayer          → answer tokens (from full reasoning chain F₁..Fₙ)
```

## Core Layer Specs (LLD)

| Layer | Input | Output | Key detail |
|-------|-------|--------|------------|
| `LatentRouterLayer` | F [B,2048] | V_T [B,W,2048] | Gumbel-Softmax (train) / Top-K (infer) |
| `SynthesizerLayer` | F [B,2048], V_T [B,2048] | F_new [B,2048] | AdaLN (FiLM) + MoE (8 experts, Top-2) |
| `CriticLayer` | F_new [B,2048] | halt_prob [B,1] | MLP + BCEWithLogitsLoss |
| `MemorizerLayer` | F_1, F_n [B,2048] | K_T [B,128], V_T [B,2048] | Shortcut rule synthesis |

## Project Structure

```
nellm/
├── src/
│   ├── models/
│   │   ├── router.py        # LatentRouterLayer
│   │   ├── synthesizer.py   # SynthesizerLayer (AdaLN + MoE)
│   │   ├── critic.py        # CriticLayer
│   │   └── memorizer.py     # MemorizerLayer
├── tests/
│   ├── test_router.py
│   ├── test_synthesizer.py  # 30 tests
│   ├── test_critic.py       # 15 tests
│   └── test_memorizer.py    # 15 tests
├── issues/                  # Dev task tracker
├── scripts/                 # Kaggle deploy scripts
└── kanban_board.md
```

## Status

| Layer | Tests | Review |
|-------|-------|--------|
| RouterLayer | ✅ pass | ✅ approved |
| SynthesizerLayer | ✅ 30/30 | ✅ approved |
| CriticLayer | ✅ 15/15 | ✅ approved |
| MemorizerLayer | ✅ 15/15 | ✅ approved |

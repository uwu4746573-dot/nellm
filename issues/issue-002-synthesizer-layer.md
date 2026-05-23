# Issue 2: Implement SynthesizerLayer

**Component:** `SynthesizerLayer`
**Objective:** Implement the layer that modulates facts with transformations (AdaLN) and generates new facts using a Mixture of Experts (MoE).

## Specification
* **Input:** `F_current` `[B, D_v]`, `V_T` `[B, D_v]` (`D_v = 2048`).
* **AdaLN (FiLM) Operation:**
  1. Predict scale & shift from `V_T`: `gamma, beta = Linear(D_v -> 2 * D_v)(V_T)`
  2. Modulate fact: `F_mod = gamma * LayerNorm(F_current) + beta`
* **MoE Operation:**
  1. Top-2 Gating: `gates = Top2Gating(F_mod)` (E = 8 experts).
  2. Experts: Each is `Linear(2048 -> 8192) -> GELU -> Linear(8192 -> 2048)`.
  3. Combine: `F_new = Sum(gate_i * Expert_i(F_mod))`
* **Output:** `F_new` of shape `[B, D_v]`.

## Tasks
- [ ] Implement `AdaLN` module with projection from `V_T` to `gamma` and `beta`.
- [ ] Implement `MoE` module with 8 experts (MLPs with GELU activation).
- [ ] Implement `Top2Gating` for expert routing.
- [ ] Assemble `SynthesizerLayer` combining AdaLN and MoE.
- [ ] Add unit tests for shape validation and correct gradients.

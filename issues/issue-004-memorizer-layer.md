# [Resolved] Issue 4: Implement MemorizerLayer

**Component:** `MemorizerLayer`
**Objective:** Implement the layer to generate shortcut transformation keys and values from successful reasoning paths.

## Specification
* **Input:** Starting fact `F_1` `[B, D_v]`, final fact `F_n` `[B, D_v]`.
* **Operations:**
  1. Generate new transformation value (Abstraction): `V_T_new = MLP_reasoning([F_1, F_n])` (concat `F_1` and `F_n`).
  2. Generate new transformation key (Condition): `K_T_new = Linear(D_v -> D_k)(F_1)`, where `D_k = 128`.
* **Output:** `V_T_new` `[B, D_v]`, `K_T_new` `[B, D_k]`.

## Tasks
- [x] Create `MemorizerLayer` `nn.Module`.
- [x] Implement the `MLP_reasoning` abstraction network (e.g. `Linear(2 * D_v, Hidden) -> ... -> Linear(Hidden, D_v)`).
- [x] Implement the key projection layer.
- [x] Add unit tests verifying concat dimensions and output shapes.

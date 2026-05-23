# Issue 3: Implement CriticLayer

**Component:** `CriticLayer`
**Objective:** Implement the MLP-based verifier to evaluate if a current fact `F_new` constitutes a final answer.

## Specification
* **Input:** `F_new` `[B, D_v]` (optionally concat global context `F_1`, where `D_v = 2048`).
* **Operation:** 
  - Small MLP classifier: `Score = Sigmoid(MLP(D_v -> 256 -> 1)(F_new))`
* **Output:** `halt_prob` tensor of shape `[B, 1]` with values in `[0, 1]`.

## Tasks
- [ ] Create `CriticLayer` `nn.Module`.
- [ ] Implement the MLP architecture (`Linear(D_v, 256) -> Activation -> Linear(256, 1)`).
- [ ] Apply Sigmoid activation to produce probability.
- [ ] (Optional) Add capability to condition on initial fact `F_1`.
- [ ] Add unit tests verifying bounds `[0, 1]` and shapes.

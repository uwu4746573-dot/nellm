import torch
import torch.nn as nn


class MemorizerLayer(nn.Module):
    """Memorizer Layer: generates shortcut transformation keys and values
    from a successful reasoning path (start fact -> final fact).

    Given the start fact F_1 and the final fact F_n from a completed reasoning
    trajectory, this layer produces:
      - V_T_new: a new transformation *value* that encodes the abstracted
                 shortcut, produced by an MLP that reasons over the
                 concatenation [F_1 || F_n].
      - K_T_new: a new transformation *key* (condition) projected from F_1,
                 used to retrieve this shortcut in future inference passes.

    Shapes (with LLD-specified defaults):
        F_1     : [B, D_v=2048]
        F_n     : [B, D_v=2048]
        concat  : [B, 2*D_v=4096]
        V_T_new : [B, D_v=2048]
        K_T_new : [B, D_k=128]
    """

    def __init__(self, d_v: int = 2048, d_k: int = 128) -> None:
        """
        Args:
            d_v: Dimension of value / fact vectors (default 2048).
            d_k: Dimension of key vectors (default 128).
        """
        super().__init__()
        self.d_v = d_v
        self.d_k = d_k

        # MLP reasoning subnetwork: [F_1 || F_n] -> V_T_new
        # Linear(2*D_v -> 2*D_v) -> GELU -> Linear(2*D_v -> D_v)
        self.mlp_reasoning = nn.Sequential(
            nn.Linear(2 * d_v, 2 * d_v),
            nn.GELU(),
            nn.Linear(2 * d_v, d_v),
        )

        # Key projection: F_1 -> K_T_new  [B, D_k]
        self.key_proj = nn.Linear(d_v, d_k)

    def forward(
        self,
        f_1: torch.Tensor,
        f_n: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute shortcut transformation value and key.

        Args:
            f_1: Starting fact tensor of shape [B, D_v].
            f_n: Final (conclusion) fact tensor of shape [B, D_v].

        Returns:
            v_t_new: New transformation value, shape [B, D_v].
            k_t_new: New transformation key,   shape [B, D_k].
        """
        B, D_v = f_1.shape
        assert D_v == self.d_v, (
            f"Expected f_1 last dim to be {self.d_v}, got {D_v}"
        )
        assert f_n.shape == f_1.shape, (
            f"f_1 and f_n must have the same shape; got {f_1.shape} vs {f_n.shape}"
        )

        # 1. Concatenate start and final facts -> [B, 2*D_v]
        concat = torch.cat([f_1, f_n], dim=-1)   # [B, 4096]

        # 2. MLP reasoning to produce new value -> [B, D_v]
        v_t_new = self.mlp_reasoning(concat)      # [B, 2048]

        # 3. Key generation from start fact -> [B, D_k]
        k_t_new = self.key_proj(f_1)              # [B, 128]

        return v_t_new, k_t_new

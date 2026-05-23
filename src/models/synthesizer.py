import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLN(nn.Module):
    """Adaptive Layer Normalization (FiLM-style).

    Given a conditioning vector ``V_T`` of shape ``[B, D_v]``, predicts
    per-sample scale (gamma) and shift (beta) and applies them to a
    layer-normalised version of the input ``F_current``.

    Args:
        d_v: Feature dimensionality (default 2048).
    """

    def __init__(self, d_v: int = 2048) -> None:
        super().__init__()
        self.d_v = d_v
        # Projects V_T -> gamma || beta  (both of size D_v)
        self.film_proj = nn.Linear(d_v, 2 * d_v)
        # Layer norm without affine so AdaLN controls scale/shift fully
        self.norm = nn.LayerNorm(d_v, elementwise_affine=False)

    def forward(self, f_current: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_current: [B, D_v]  – fact representation to modulate.
            v_t:       [B, D_v]  – conditioning transformation vector.
        Returns:
            f_mod:     [B, D_v]  – modulated fact.
        """
        # Predict gamma and beta from conditioning signal
        gamma_beta = self.film_proj(v_t)          # [B, 2*D_v]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each [B, D_v]

        # Apply FiLM: gamma * LayerNorm(F_current) + beta
        f_normed = self.norm(f_current)            # [B, D_v]
        f_mod = gamma * f_normed + beta            # [B, D_v]
        return f_mod


class ExpertMLP(nn.Module):
    """Single expert MLP: Linear(D_v -> D_ff) -> GELU -> Linear(D_ff -> D_v).

    Args:
        d_v:  Input/output dimensionality (default 2048).
        d_ff: Hidden dimensionality (default 8192).
    """

    def __init__(self, d_v: int = 2048, d_ff: int = 8192) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_v, d_ff)
        self.fc2 = nn.Linear(d_ff, d_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D_v]
        Returns:
            [B, D_v]
        """
        return self.fc2(F.gelu(self.fc1(x)))


class Top2Gating(nn.Module):
    """Top-2 sparse gating for Mixture of Experts.

    Computes a softmax over ``num_experts`` logits and returns the two
    highest-weight experts along with their (re-normalised) gate values.

    Args:
        d_v:         Input dimensionality.
        num_experts: Total number of experts (default 8).
    """

    def __init__(self, d_v: int = 2048, num_experts: int = 8) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.gate_proj = nn.Linear(d_v, num_experts)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, D_v]
        Returns:
            top2_indices: [B, 2]   – indices of the two selected experts.
            top2_weights: [B, 2]   – re-normalised gate weights (sum to 1).
        """
        logits = self.gate_proj(x)                          # [B, num_experts]
        probs = F.softmax(logits, dim=-1)                   # [B, num_experts]

        top2_weights, top2_indices = torch.topk(probs, k=2, dim=-1)  # [B, 2]
        # Re-normalise so the two selected weights sum to 1
        top2_weights = top2_weights / top2_weights.sum(dim=-1, keepdim=True)

        return top2_indices, top2_weights


class MoE(nn.Module):
    """Mixture of Experts with Top-2 gating.

    Args:
        d_v:         Feature dimensionality (default 2048).
        d_ff:        Expert hidden dimensionality (default 8192).
        num_experts: Total number of experts (default 8).
    """

    def __init__(self, d_v: int = 2048, d_ff: int = 8192, num_experts: int = 8) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.gating = Top2Gating(d_v=d_v, num_experts=num_experts)
        self.experts = nn.ModuleList(
            [ExpertMLP(d_v=d_v, d_ff=d_ff) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D_v]
        Returns:
            [B, D_v]  – weighted combination of top-2 expert outputs.
        """
        top2_indices, top2_weights = self.gating(x)  # [B,2], [B,2]
        B, D_v = x.shape

        # Accumulate weighted expert outputs
        output = torch.zeros(B, D_v, device=x.device, dtype=x.dtype)
        for k in range(2):
            indices_k = top2_indices[:, k]    # [B]
            weights_k = top2_weights[:, k]    # [B]

            # Group samples by selected expert to process them in bulk
            for e_idx in range(self.num_experts):
                mask = (indices_k == e_idx)   # [B] boolean
                if not mask.any():
                    continue
                x_e = x[mask]                          # [n_e, D_v]
                expert_out = self.experts[e_idx](x_e)  # [n_e, D_v]
                output[mask] += weights_k[mask].unsqueeze(-1) * expert_out

        return output


class SynthesizerLayer(nn.Module):
    """Synthesizer layer combining AdaLN modulation with Top-2 MoE.

    Processing pipeline::

        F_mod  = AdaLN(F_current, V_T)   # FiLM modulation
        F_new  = MoE(F_mod)              # Mixture of Experts

    Args:
        d_v:         Latent feature dimensionality (default 2048).
        d_ff:        Expert hidden dimensionality (default 8192).
        num_experts: Number of MoE experts (default 8).
    """

    def __init__(
        self,
        d_v: int = 2048,
        d_ff: int = 8192,
        num_experts: int = 8,
    ) -> None:
        super().__init__()
        self.d_v = d_v
        self.adaLN = AdaLN(d_v=d_v)
        self.moe = MoE(d_v=d_v, d_ff=d_ff, num_experts=num_experts)

    def forward(self, f_current: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_current: [B, D_v=2048] – current fact representation.
            v_t:       [B, D_v=2048] – transformation vector from LatentRouterLayer.
        Returns:
            f_new:     [B, D_v=2048] – updated fact representation.
        """
        assert f_current.shape == v_t.shape, (
            f"f_current and v_t must have the same shape, "
            f"got {f_current.shape} vs {v_t.shape}"
        )
        B, D_v = f_current.shape
        assert D_v == self.d_v, f"Expected D_v={self.d_v}, got {D_v}"

        f_mod = self.adaLN(f_current, v_t)  # [B, D_v]
        f_new = self.moe(f_mod)              # [B, D_v]
        return f_new

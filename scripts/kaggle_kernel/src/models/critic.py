import torch
import torch.nn as nn


class CriticLayer(nn.Module):
    """MLP-based verifier that estimates the halting logit.

    Given a current fact representation ``F_new`` (and optionally an initial
    global-context fact ``F_1``), the layer outputs a scalar **raw logit**
    per batch item.  The Sigmoid activation is intentionally omitted so that
    training can use ``nn.BCEWithLogitsLoss``, which fuses the log and sigmoid
    in a single numerically-stable operation.  To obtain a probability at
    inference time, apply ``torch.sigmoid()`` to the returned logit.

    Architecture::

        [concat F_new (+ F_1)] -> Linear(D_in, 256) -> GELU -> Linear(256, 1)  # raw logit

    Args:
        d_v (int): Fact vector dimensionality (``D_v``). Default: 2048.
        use_global_context (bool): When ``True`` the layer expects a second
            argument ``F_1`` and concatenates it with ``F_new`` before the MLP,
            setting ``D_in = 2 * d_v``.  When ``False``, ``D_in = d_v``.
    """

    def __init__(self, d_v: int = 2048, use_global_context: bool = False) -> None:
        super().__init__()
        self.d_v = d_v
        self.use_global_context = use_global_context

        d_in = 2 * d_v if use_global_context else d_v

        self.mlp = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            # No Sigmoid here — use nn.BCEWithLogitsLoss during training for
            # numerical stability (fused log-sigmoid avoids float overflow).
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        f_new: torch.Tensor,
        f_1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute halt logit (raw, pre-sigmoid).

        Args:
            f_new: Current fact tensor of shape ``[B, D_v]``.
            f_1:   Optional global-context tensor of shape ``[B, D_v]``.
                   Required when ``use_global_context=True``.

        Returns:
            halt_logit: Raw logit tensor of shape ``[B, 1]`` (unbounded).
                        Pass to ``nn.BCEWithLogitsLoss`` during training, or
                        apply ``torch.sigmoid()`` at inference to get a
                        probability in ``[0, 1]``.
        """
        B, D_v = f_new.shape
        assert D_v == self.d_v, (
            f"Expected D_v={self.d_v}, got {D_v}"
        )

        if self.use_global_context:
            assert f_1 is not None, (
                "f_1 must be provided when use_global_context=True"
            )
            assert f_1.shape == f_new.shape, (
                f"f_1 shape {f_1.shape} must match f_new shape {f_new.shape}"
            )
            x = torch.cat([f_new, f_1], dim=-1)  # [B, 2*D_v]
        else:
            x = f_new  # [B, D_v]

        halt_logit = self.mlp(x)  # [B, 1] — raw logit
        return halt_logit

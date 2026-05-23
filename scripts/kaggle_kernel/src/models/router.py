import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LatentRouterLayer(nn.Module):
    def __init__(self, d_v: int = 2048, d_k: int = 128, n_t: int = 64, beam_width: int = 5, temperature: float = 1.0):
        super().__init__()
        self.d_v = d_v
        self.d_k = d_k
        self.n_t = n_t
        self.beam_width = beam_width
        self.temperature = temperature
        
        self.query_proj = nn.Linear(d_v, d_k)
        self.K_T = nn.Parameter(torch.randn(n_t, d_k) / math.sqrt(d_k))
        self.V_T = nn.Parameter(torch.randn(n_t, d_v) / math.sqrt(d_v))
        
    def forward(self, f_current: torch.Tensor) -> torch.Tensor:
        """
        Input: F_current tensor of shape [B, D_v]
        Returns:
            V_T_selected: [B, W, D_v] in inference mode (eval)
            V_T_soft: [B, D_v] in training mode
        """
        B, D_v = f_current.shape
        assert D_v == self.d_v, f"Expected D_v to be {self.d_v}, got {D_v}"
        
        # 1. Project fact to query
        Q = self.query_proj(f_current) # [B, D_k]
        
        # 2. Compute logits against transformation keys
        logits = torch.matmul(Q, self.K_T.t()) / math.sqrt(self.d_k) # [B, N_T]
        
        if self.training:
            # 3. Training (Soft mode)
            weights = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1) # [B, N_T]
            v_t_soft = torch.matmul(weights, self.V_T) # [B, D_v]
            return v_t_soft
        else:
            # 4. Inference (Hard mode)
            w = min(self.beam_width, self.n_t)
            _, topk_indices = torch.topk(logits, k=w, dim=-1) # [B, W]
            
            # Extract V_T_selected
            v_t_selected = self.V_T[topk_indices] # [B, W, D_v]
            return v_t_selected

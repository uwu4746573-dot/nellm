import torch
import torch.nn as nn

class LatentEncoderLayer(nn.Module):
    """
    Latent Encoder Layer using Latent Attention Pooling.
    Compresses raw LLM hidden states into a latent reasoning fact vector.
    """
    def __init__(self, d_model: int = 4096, num_latents: int = 512, d_v: int = 2048, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_latents = num_latents
        self.d_v = d_v
        
        # Learnable latent queries
        self.latent_queries = nn.Parameter(torch.randn(num_latents, d_model))
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        
        # Linear projection to latent reasoning space
        self.projection = nn.Linear(d_model, d_v)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Tensor of shape [B, SeqLen, D_model]
        
        Returns:
            F_1 tensor of shape [B, D_v]
        """
        B, seq_len, d_m = hidden_states.shape
        assert d_m == self.d_model, f"Expected hidden_states with last dim {self.d_model}, got {d_m}"
        
        # Expand latent queries for the batch
        # [B, num_latents, D_model]
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Cross-attention: queries attend to hidden_states (keys/values)
        # attn_output shape: [B, num_latents, D_model]
        attn_output, _ = self.cross_attention(query=queries, key=hidden_states, value=hidden_states)
        
        # Mean pooling over the num_latents dimension
        # pooled_output shape: [B, D_model]
        pooled_output = attn_output.mean(dim=1)
        
        # Project to D_v
        # out shape: [B, D_v]
        out = self.projection(pooled_output)
        
        return out

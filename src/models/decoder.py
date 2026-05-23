import torch
import torch.nn as nn

class LatentDecoderLayer(nn.Module):
    """
    LatentDecoderLayer projects the final latent reasoning vector back to 
    the LLM's input embedding space as a soft prompt/continuous prefix.
    """
    def __init__(self, d_v: int = 2048, d_model: int = 4096, prefix_len: int = 16):
        """
        Args:
            d_v (int): Dimension of the input latent reasoning vector. Default: 2048.
            d_model (int): Dimension of the LLM's embedding space. Default: 4096.
            prefix_len (int): Number of tokens in the unrolled soft prompt sequence. Default: 16.
        """
        super().__init__()
        self.d_v = d_v
        self.d_model = d_model
        self.prefix_len = prefix_len
        
        # Project the dense reasoning vector into a sequence of prefix_len tokens
        self.projection = nn.Linear(d_v, prefix_len * d_model)
        
        # Apply layer normalization to stabilize the soft prompt
        self.norm = nn.LayerNorm(d_model)

    def forward(self, f_n: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_n (torch.Tensor): Final reasoning vector of shape [B, d_v].
            
        Returns:
            torch.Tensor: Soft prompt tensor of shape [B, prefix_len, d_model].
        """
        # Get batch size
        batch_size = f_n.shape[0]
        
        # Project: [B, d_v] -> [B, prefix_len * d_model]
        projected = self.projection(f_n)
        
        # Reshape: [B, prefix_len * d_model] -> [B, prefix_len, d_model]
        reshaped = projected.view(batch_size, self.prefix_len, self.d_model)
        
        # Normalize: [B, prefix_len, d_model] -> [B, prefix_len, d_model]
        soft_prompt = self.norm(reshaped)
        
        return soft_prompt

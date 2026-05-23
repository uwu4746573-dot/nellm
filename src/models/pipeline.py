import torch
import torch.nn as nn

from .encoder import LatentEncoderLayer
from .router import LatentRouterLayer
from .synthesizer import SynthesizerLayer
from .critic import CriticLayer
from .memorizer import MemorizerLayer
from .decoder import LatentDecoderLayer

class NeLLMReasoningPipeline(nn.Module):
    """
    The complete NeLL-M Latent Reasoning Pipeline.
    This module orchestrates the 6 core layers into a unified reasoning process.
    """
    def __init__(
        self, 
        d_model: int = 4096, 
        d_v: int = 2048, 
        d_k: int = 128, 
        n_t: int = 64, 
        beam_width: int = 5, 
        prefix_len: int = 16
    ):
        super().__init__()
        
        # 1. Input Layer
        self.encoder = LatentEncoderLayer(d_model=d_model, d_v=d_v)
        
        # 2. Reasoning Loop Layers
        self.router = LatentRouterLayer(d_v=d_v, d_k=d_k, n_t=n_t, beam_width=beam_width)
        self.synthesizer = SynthesizerLayer(d_v=d_v)
        self.critic = CriticLayer(d_v=d_v, use_global_context=True)
        
        # 3. K-V Logic Cache Layer
        self.memorizer = MemorizerLayer(d_v=d_v, d_k=d_k)
        
        # 4. Output Layer
        self.decoder = LatentDecoderLayer(d_v=d_v, d_model=d_model, prefix_len=prefix_len)

    def generate_soft_prompt(self, hidden_states: torch.Tensor, max_steps: int = 3) -> torch.Tensor:
        """
        Executes the latent reasoning loop.
        Args:
            hidden_states: [B, SeqLen, D_model] from the base LLM.
            max_steps: Maximum reasoning depth before forced exit.
        Returns:
            soft_prompt: [B, prefix_len, D_model] to be passed back to the LLM.
        """
        B = hidden_states.size(0)
        
        # 1. Encode into reasoning space
        f_1 = self.encoder(hidden_states) # [B, D_v]
        f_current = f_1
        
        # In a real training scenario, we would keep track of the full W=5 beam tree.
        # For simplicity in this forward pass, we demonstrate the greedy/single-path execution.
        
        for step in range(max_steps):
            # 2. Find relevant transformation rule
            # During training, router returns soft Gumbel-Softmax weights [B, D_v]
            v_t = self.router(f_current) 
            
            # If in inference mode, we would branch out here for beam search
            if not self.training:
                # v_t has shape [B, W, D_v]. We take the top branch for this demo.
                v_t = v_t[:, 0, :] 
                
            # 3. Apply transformation
            f_new = self.synthesizer(f_current, v_t) # [B, D_v]
            
            # 4. Evaluate step
            # Critic evaluates if reasoning is complete (halt probability)
            halt_prob = self.critic(f_new, f_1) # [B, 1]
            
            f_current = f_new
            
            # (Inference) If halt probability > threshold, stop early
            if not self.training and (halt_prob > 0.5).all():
                break

        # 5. Memorize shortcut (optional, triggered on successful reasoning)
        # self.memorizer(f_1, f_current)
        
        # 6. Decode back to LLM space
        soft_prompt = self.decoder(f_current) # [B, prefix_len, D_model]
        
        return soft_prompt

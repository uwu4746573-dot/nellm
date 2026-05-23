"""
NeLL-M: Neural Latent Language Model
Core model layers for latent reasoning with dynamic K-V cache.

Architecture:
  F (Fact vector):  [B, D_v=2048] — current reasoning state
  T (Transform vector): [B, D_v=2048] — logical rule/operator

Pipeline per reasoning step:
  LatentRouterLayer -> SynthesizerLayer -> CriticLayer
  On success: MemorizerLayer writes shortcut T to K-V base
"""

from .router import LatentRouterLayer
from .synthesizer import SynthesizerLayer, AdaLN, MoE, ExpertMLP, Top2Gating
from .critic import CriticLayer
from .memorizer import MemorizerLayer

__all__ = [
    "LatentRouterLayer",
    "SynthesizerLayer",
    "AdaLN",
    "MoE",
    "ExpertMLP",
    "Top2Gating",
    "CriticLayer",
    "MemorizerLayer",
]

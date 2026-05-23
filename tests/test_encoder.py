import torch
import pytest
from src.models.encoder import LatentEncoderLayer

def test_latent_encoder_layer_shapes():
    B = 2
    SeqLen = 128
    d_model = 64
    num_latents = 16
    d_v = 32
    num_heads = 4
    
    layer = LatentEncoderLayer(d_model=d_model, num_latents=num_latents, d_v=d_v, num_heads=num_heads)
    
    # Create dummy input
    hidden_states = torch.randn(B, SeqLen, d_model)
    
    # Forward pass
    out = layer(hidden_states)
    
    # Check output shape
    assert out.shape == (B, d_v), f"Expected shape ({B}, {d_v}), got {out.shape}"

def test_latent_encoder_layer_no_nans():
    B = 4
    SeqLen = 64
    d_model = 128
    num_latents = 32
    d_v = 64
    num_heads = 8
    
    layer = LatentEncoderLayer(d_model=d_model, num_latents=num_latents, d_v=d_v, num_heads=num_heads)
    
    # Create valid dummy input
    hidden_states = torch.randn(B, SeqLen, d_model)
    
    # Forward pass
    out = layer(hidden_states)
    
    # Check for NaNs and Infs in forward pass
    assert not torch.isnan(out).any(), "Output contains NaNs"
    assert not torch.isinf(out).any(), "Output contains Infs"

    # Backward pass
    loss = out.sum()
    loss.backward()

    # Check for NaNs and Infs in gradients
    for name, param in layer.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"Gradient of {name} contains NaNs"
            assert not torch.isinf(param.grad).any(), f"Gradient of {name} contains Infs"

def test_latent_encoder_layer_batch_independence():
    B = 3
    SeqLen = 32
    d_model = 64
    num_latents = 16
    d_v = 32
    num_heads = 4
    
    layer = LatentEncoderLayer(d_model=d_model, num_latents=num_latents, d_v=d_v, num_heads=num_heads)
    layer.eval()  # Set to eval mode for deterministic behavior if there were dropouts
    
    hidden_states = torch.randn(B, SeqLen, d_model)
    
    # Forward pass on the whole batch
    out_batch = layer(hidden_states)
    
    # Forward pass on the first element of the batch
    out_single = layer(hidden_states[0:1])
    
    # Check if the output for the first element is the same in both cases
    assert torch.allclose(out_batch[0:1], out_single, atol=1e-5), "Batch processing is not independent"

def test_latent_encoder_layer_default_args():
    # Test if we can initialize with default args (might be too large to run effectively, but we can just init)
    layer = LatentEncoderLayer()
    assert layer.d_model == 4096
    assert layer.num_latents == 512
    assert layer.d_v == 2048

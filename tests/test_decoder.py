import torch
import pytest
from src.models.decoder import LatentDecoderLayer

def test_latent_decoder_layer_shapes():
    batch_size = 4
    d_v = 2048
    d_model = 4096
    prefix_len = 16
    
    layer = LatentDecoderLayer(d_v=d_v, d_model=d_model, prefix_len=prefix_len)
    
    # Input tensor shape: [B, D_v]
    f_n = torch.randn(batch_size, d_v)
    
    # Forward pass
    output = layer(f_n)
    
    # Expected output shape: [B, prefix_len, d_model]
    expected_shape = (batch_size, prefix_len, d_model)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"

def test_latent_decoder_layer_variable_dimensions():
    batch_size = 2
    d_v = 1024
    d_model = 512
    prefix_len = 8
    
    layer = LatentDecoderLayer(d_v=d_v, d_model=d_model, prefix_len=prefix_len)
    
    f_n = torch.randn(batch_size, d_v)
    output = layer(f_n)
    
    expected_shape = (batch_size, prefix_len, d_model)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"

def test_latent_decoder_layer_forward_pass_no_errors():
    layer = LatentDecoderLayer()
    f_n = torch.randn(1, 2048)  # Batch size 1, default d_v
    
    try:
        output = layer(f_n)
        assert not torch.isnan(output).any(), "Output contains NaNs"
    except Exception as e:
        pytest.fail(f"Forward pass raised an exception: {e}")

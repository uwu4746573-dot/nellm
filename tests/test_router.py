import torch
import unittest
import sys
import os

# Add src to python path for testing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.router import LatentRouterLayer

class TestLatentRouterLayer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.d_v = 2048
        self.d_k = 128
        self.n_t = 64
        self.beam_width = 5
        self.layer = LatentRouterLayer(
            d_v=self.d_v,
            d_k=self.d_k,
            n_t=self.n_t,
            beam_width=self.beam_width
        )
        self.f_current = torch.randn(self.batch_size, self.d_v)

    def test_training_mode(self):
        """Test output shape in training mode (Soft mode)"""
        self.layer.train()
        output = self.layer(self.f_current)
        self.assertEqual(output.shape, (self.batch_size, self.d_v))

    def test_inference_mode(self):
        """Test output shape in inference mode (Hard mode)"""
        self.layer.eval()
        output = self.layer(self.f_current)
        self.assertEqual(output.shape, (self.batch_size, self.beam_width, self.d_v))

if __name__ == '__main__':
    unittest.main()

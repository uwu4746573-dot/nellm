import torch
import unittest
import sys
import os

# Add project root to Python path for testing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.memorizer import MemorizerLayer


class TestMemorizerLayer(unittest.TestCase):
    """Unit tests for MemorizerLayer (Issue 4)."""

    def setUp(self):
        self.batch_size = 4
        self.d_v = 2048
        self.d_k = 128
        self.layer = MemorizerLayer(d_v=self.d_v, d_k=self.d_k)
        self.f_1 = torch.randn(self.batch_size, self.d_v)
        self.f_n = torch.randn(self.batch_size, self.d_v)

    # ------------------------------------------------------------------
    # Dimension / shape tests
    # ------------------------------------------------------------------

    def test_concatenation_dimension(self):
        """Intermediate concat [F_1 || F_n] must be [B, 2*D_v]."""
        concat = torch.cat([self.f_1, self.f_n], dim=-1)
        self.assertEqual(concat.shape, (self.batch_size, 2 * self.d_v))

    def test_v_t_new_shape(self):
        """V_T_new output must have shape [B, D_v]."""
        v_t_new, _ = self.layer(self.f_1, self.f_n)
        self.assertEqual(v_t_new.shape, (self.batch_size, self.d_v))

    def test_k_t_new_shape(self):
        """K_T_new output must have shape [B, D_k]."""
        _, k_t_new = self.layer(self.f_1, self.f_n)
        self.assertEqual(k_t_new.shape, (self.batch_size, self.d_k))

    def test_output_is_tuple_of_two_tensors(self):
        """forward() must return exactly two tensors."""
        result = self.layer(self.f_1, self.f_n)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    # ------------------------------------------------------------------
    # Default dimension values
    # ------------------------------------------------------------------

    def test_default_dimensions(self):
        """Default d_v=2048, d_k=128 as specified in the LLD."""
        default_layer = MemorizerLayer()
        self.assertEqual(default_layer.d_v, 2048)
        self.assertEqual(default_layer.d_k, 128)

    # ------------------------------------------------------------------
    # Batch-size independence
    # ------------------------------------------------------------------

    def test_single_sample_batch(self):
        """Works correctly with batch size of 1."""
        f_1 = torch.randn(1, self.d_v)
        f_n = torch.randn(1, self.d_v)
        v_t_new, k_t_new = self.layer(f_1, f_n)
        self.assertEqual(v_t_new.shape, (1, self.d_v))
        self.assertEqual(k_t_new.shape, (1, self.d_k))

    def test_large_batch(self):
        """Works correctly with a larger batch size."""
        f_1 = torch.randn(32, self.d_v)
        f_n = torch.randn(32, self.d_v)
        v_t_new, k_t_new = self.layer(f_1, f_n)
        self.assertEqual(v_t_new.shape, (32, self.d_v))
        self.assertEqual(k_t_new.shape, (32, self.d_k))

    # ------------------------------------------------------------------
    # Key isolation: K_T_new depends only on F_1
    # ------------------------------------------------------------------

    def test_key_depends_only_on_f1(self):
        """K_T_new must be identical for different F_n (key comes from F_1 only)."""
        f_n_alt = torch.randn(self.batch_size, self.d_v)
        _, k_t_new_a = self.layer(self.f_1, self.f_n)
        _, k_t_new_b = self.layer(self.f_1, f_n_alt)
        self.assertTrue(
            torch.allclose(k_t_new_a, k_t_new_b),
            "K_T_new should be identical when F_1 is the same, regardless of F_n.",
        )

    # ------------------------------------------------------------------
    # MLP subnetwork architecture sanity
    # ------------------------------------------------------------------

    def test_mlp_reasoning_layer_count(self):
        """MLP reasoning subnetwork must have exactly two Linear layers."""
        linear_layers = [
            m for m in self.layer.mlp_reasoning.modules()
            if isinstance(m, torch.nn.Linear)
        ]
        self.assertEqual(len(linear_layers), 2)

    def test_mlp_reasoning_hidden_dimensions(self):
        """MLP hidden Linear(4096->4096) and output Linear(4096->2048)."""
        linear_layers = [
            m for m in self.layer.mlp_reasoning.modules()
            if isinstance(m, torch.nn.Linear)
        ]
        # First layer: 2*D_v -> 2*D_v
        self.assertEqual(linear_layers[0].in_features, 2 * self.d_v)
        self.assertEqual(linear_layers[0].out_features, 2 * self.d_v)
        # Second layer: 2*D_v -> D_v
        self.assertEqual(linear_layers[1].in_features, 2 * self.d_v)
        self.assertEqual(linear_layers[1].out_features, self.d_v)

    def test_key_proj_dimensions(self):
        """Key projection must be Linear(D_v -> D_k)."""
        self.assertEqual(self.layer.key_proj.in_features, self.d_v)
        self.assertEqual(self.layer.key_proj.out_features, self.d_k)

    # ------------------------------------------------------------------
    # Gradient flow
    # ------------------------------------------------------------------

    def test_gradients_flow_through_v_t_new(self):
        """V_T_new must support gradient back-propagation."""
        f_1 = torch.randn(self.batch_size, self.d_v, requires_grad=True)
        f_n = torch.randn(self.batch_size, self.d_v, requires_grad=True)
        v_t_new, _ = self.layer(f_1, f_n)
        loss = v_t_new.sum()
        loss.backward()
        self.assertIsNotNone(f_1.grad)
        self.assertIsNotNone(f_n.grad)

    def test_gradients_flow_through_k_t_new(self):
        """K_T_new must support gradient back-propagation."""
        f_1 = torch.randn(self.batch_size, self.d_v, requires_grad=True)
        f_n = torch.randn(self.batch_size, self.d_v, requires_grad=True)
        _, k_t_new = self.layer(f_1, f_n)
        loss = k_t_new.sum()
        loss.backward()
        self.assertIsNotNone(f_1.grad)

    # ------------------------------------------------------------------
    # Assertion guards
    # ------------------------------------------------------------------

    def test_mismatched_shapes_raise(self):
        """Mismatched F_1 / F_n shapes must raise AssertionError."""
        f_n_wrong = torch.randn(self.batch_size, self.d_v + 1)
        with self.assertRaises((AssertionError, RuntimeError)):
            self.layer(self.f_1, f_n_wrong)

    def test_wrong_d_v_raises(self):
        """Wrong feature dimension must raise AssertionError."""
        f_bad = torch.randn(self.batch_size, self.d_v + 64)
        with self.assertRaises((AssertionError, RuntimeError)):
            self.layer(f_bad, f_bad)


if __name__ == '__main__':
    unittest.main()

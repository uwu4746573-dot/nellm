"""Unit tests for CriticLayer."""
import unittest
import sys
import os

import torch
import torch.nn as nn

# Add project root to path so `src` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.critic import CriticLayer


class TestCriticLayerBasic(unittest.TestCase):
    """Tests for CriticLayer without global context."""

    def setUp(self):
        self.batch_size = 4
        self.d_v = 2048
        self.layer = CriticLayer(d_v=self.d_v, use_global_context=False)
        self.f_new = torch.randn(self.batch_size, self.d_v)

    # ------------------------------------------------------------------
    # Shape tests
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """halt_prob must be [B, 1]."""
        output = self.layer(self.f_new)
        self.assertEqual(output.shape, (self.batch_size, 1))

    # ------------------------------------------------------------------
    # Value-range tests
    # ------------------------------------------------------------------

    def test_output_is_unbounded_logit(self):
        """Raw logit output should NOT be constrained to [0, 1]."""
        # With random weights and large-magnitude inputs at least some logits
        # will land outside [0, 1], confirming Sigmoid is absent.
        torch.manual_seed(0)
        layer = CriticLayer(d_v=self.d_v, use_global_context=False)
        f_extreme = torch.full((self.batch_size, self.d_v), 1e2)
        out = layer(f_extreme)
        # At least one logit should be outside (0, 1) after an untrained pass.
        self.assertFalse(
            ((out >= 0.0) & (out <= 1.0)).all(),
            "All logits happen to be in [0,1] — Sigmoid may still be present",
        )

    def test_sigmoid_at_inference_gives_probability(self):
        """Applying torch.sigmoid() to the logit must yield values in [0, 1]."""
        output = self.layer(self.f_new)
        prob = torch.sigmoid(output)
        self.assertTrue((prob >= 0.0).all())
        self.assertTrue((prob <= 1.0).all())

    # ------------------------------------------------------------------
    # Architecture tests
    # ------------------------------------------------------------------

    def test_mlp_architecture(self):
        """MLP should be: Linear(d_v->256) -> GELU -> Linear(256->1) (no Sigmoid)."""
        layers = list(self.layer.mlp.children())
        self.assertEqual(len(layers), 3, "Expected exactly 3 layers (no Sigmoid)")
        self.assertIsInstance(layers[0], nn.Linear)
        self.assertEqual(layers[0].in_features, self.d_v)
        self.assertEqual(layers[0].out_features, 256)
        self.assertIsInstance(layers[1], nn.GELU)
        self.assertIsInstance(layers[2], nn.Linear)
        self.assertEqual(layers[2].in_features, 256)
        self.assertEqual(layers[2].out_features, 1)
        # Confirm Sigmoid is absent
        for layer in layers:
            self.assertNotIsInstance(layer, nn.Sigmoid, "Sigmoid must not be in MLP")

    def test_batch_size_one(self):
        """Should work for a single-item batch."""
        f = torch.randn(1, self.d_v)
        out = self.layer(f)
        self.assertEqual(out.shape, (1, 1))

    def test_gradient_flows(self):
        """Gradients must flow back through halt_prob to f_new."""
        f = torch.randn(self.batch_size, self.d_v, requires_grad=True)
        out = self.layer(f)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(f.grad)
        self.assertFalse(torch.all(f.grad == 0))

    def test_no_f1_raises_without_context_flag(self):
        """Passing f_1 while use_global_context=False should be silently ignored."""
        f_1 = torch.randn(self.batch_size, self.d_v)
        # f_1 is simply not consumed; the call should succeed
        out = self.layer(self.f_new, f_1=f_1)
        self.assertEqual(out.shape, (self.batch_size, 1))

    # ------------------------------------------------------------------
    # Loss test
    # ------------------------------------------------------------------

    def test_bce_with_logits_loss_compatible(self):
        """Raw logit must be compatible with nn.BCEWithLogitsLoss (no NaN/Inf)."""
        criterion = nn.BCEWithLogitsLoss()
        targets = torch.randint(0, 2, (self.batch_size, 1)).float()
        output = self.layer(self.f_new)
        loss = criterion(output, targets)
        self.assertFalse(torch.isnan(loss), "BCEWithLogits loss is NaN")
        self.assertFalse(torch.isinf(loss), "BCEWithLogits loss is Inf")
        self.assertTrue(loss.item() >= 0.0, "BCEWithLogits loss must be non-negative")

    def test_numerically_stable_extreme_inputs(self):
        """BCEWithLogitsLoss must remain finite even for very large logits."""
        criterion = nn.BCEWithLogitsLoss()
        f_extreme = torch.full((self.batch_size, self.d_v), 1e4)
        targets = torch.ones(self.batch_size, 1)
        output = self.layer(f_extreme)
        loss = criterion(output, targets)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_wrong_d_v_raises(self):
        """Passing a tensor with wrong D_v should raise AssertionError."""
        bad_input = torch.randn(self.batch_size, 512)
        with self.assertRaises(AssertionError):
            self.layer(bad_input)


class TestCriticLayerGlobalContext(unittest.TestCase):
    """Tests for CriticLayer with use_global_context=True."""

    def setUp(self):
        self.batch_size = 4
        self.d_v = 2048
        self.layer = CriticLayer(d_v=self.d_v, use_global_context=True)
        self.f_new = torch.randn(self.batch_size, self.d_v)
        self.f_1 = torch.randn(self.batch_size, self.d_v)

    def test_output_shape_with_context(self):
        """halt_prob must be [B, 1] even when global context is used."""
        output = self.layer(self.f_new, f_1=self.f_1)
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_output_is_unbounded_with_context(self):
        """Output must be a raw logit (not bounded to [0,1]) when context is used."""
        output = self.layer(self.f_new, f_1=self.f_1)
        prob = torch.sigmoid(output)
        # Sigmoid of any logit is always in (0,1)
        self.assertTrue((prob >= 0.0).all())
        self.assertTrue((prob <= 1.0).all())

    def test_mlp_input_dim_with_context(self):
        """First Linear layer must accept 2*d_v inputs."""
        first_linear = list(self.layer.mlp.children())[0]
        self.assertEqual(first_linear.in_features, 2 * self.d_v)

    def test_missing_f1_raises(self):
        """Omitting f_1 when use_global_context=True must raise AssertionError."""
        with self.assertRaises(AssertionError):
            self.layer(self.f_new)

    def test_mismatched_f1_shape_raises(self):
        """f_1 with a different shape than f_new must raise AssertionError."""
        bad_f1 = torch.randn(self.batch_size, 512)
        with self.assertRaises(AssertionError):
            self.layer(self.f_new, f_1=bad_f1)

    def test_bce_with_logits_loss_compatible_with_context(self):
        """BCEWithLogitsLoss must work correctly with the context-conditioned critic."""
        criterion = nn.BCEWithLogitsLoss()
        targets = torch.randint(0, 2, (self.batch_size, 1)).float()
        output = self.layer(self.f_new, f_1=self.f_1)
        loss = criterion(output, targets)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))


if __name__ == "__main__":
    unittest.main()

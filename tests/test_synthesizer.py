import torch
import unittest
import sys
import os

# Add repo root to Python path so `src` is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.synthesizer import AdaLN, ExpertMLP, Top2Gating, MoE, SynthesizerLayer


class TestAdaLN(unittest.TestCase):
    """Tests for the Adaptive Layer Normalization (FiLM) module."""

    def setUp(self):
        self.B = 4
        self.d_v = 2048
        self.layer = AdaLN(d_v=self.d_v)
        self.f_current = torch.randn(self.B, self.d_v)
        self.v_t = torch.randn(self.B, self.d_v)

    def test_output_shape(self):
        """Output must be [B, D_v]."""
        out = self.layer(self.f_current, self.v_t)
        self.assertEqual(out.shape, (self.B, self.d_v))

    def test_output_dtype(self):
        """Output dtype must match input dtype."""
        out = self.layer(self.f_current, self.v_t)
        self.assertEqual(out.dtype, self.f_current.dtype)

    def test_film_proj_output_dim(self):
        """Film projection must produce 2*D_v values."""
        out = self.layer.film_proj(self.v_t)
        self.assertEqual(out.shape, (self.B, 2 * self.d_v))

    def test_gradients_flow(self):
        """Gradients should flow through both f_current and v_t."""
        f = self.f_current.requires_grad_(True)
        v = self.v_t.requires_grad_(True)
        out = self.layer(f, v)
        out.sum().backward()
        self.assertIsNotNone(f.grad)
        self.assertIsNotNone(v.grad)

    def test_different_v_t_changes_output(self):
        """Different conditioning vectors must produce different outputs."""
        out1 = self.layer(self.f_current, self.v_t)
        out2 = self.layer(self.f_current, torch.zeros_like(self.v_t))
        self.assertFalse(torch.allclose(out1, out2))


class TestExpertMLP(unittest.TestCase):
    """Tests for a single Expert MLP."""

    def setUp(self):
        self.B = 4
        self.d_v = 2048
        self.d_ff = 8192
        self.expert = ExpertMLP(d_v=self.d_v, d_ff=self.d_ff)
        self.x = torch.randn(self.B, self.d_v)

    def test_output_shape(self):
        """Output must be [B, D_v]."""
        out = self.expert(self.x)
        self.assertEqual(out.shape, (self.B, self.d_v))

    def test_gradients_flow(self):
        """Gradients should propagate through the expert."""
        x = self.x.requires_grad_(True)
        out = self.expert(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)


class TestTop2Gating(unittest.TestCase):
    """Tests for the Top-2 gating mechanism."""

    def setUp(self):
        self.B = 4
        self.d_v = 2048
        self.num_experts = 8
        self.gating = Top2Gating(d_v=self.d_v, num_experts=self.num_experts)
        self.x = torch.randn(self.B, self.d_v)

    def test_index_shape(self):
        """Returned indices must be [B, 2]."""
        indices, _ = self.gating(self.x)
        self.assertEqual(indices.shape, (self.B, 2))

    def test_weight_shape(self):
        """Returned weights must be [B, 2]."""
        _, weights = self.gating(self.x)
        self.assertEqual(weights.shape, (self.B, 2))

    def test_weights_sum_to_one(self):
        """Re-normalised top-2 weights must sum to 1 per sample."""
        _, weights = self.gating(self.x)
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(self.B), atol=1e-5))

    def test_weights_non_negative(self):
        """All gate weights must be non-negative."""
        _, weights = self.gating(self.x)
        self.assertTrue((weights >= 0).all())

    def test_indices_valid_range(self):
        """All expert indices must be in [0, num_experts)."""
        indices, _ = self.gating(self.x)
        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < self.num_experts).all())

    def test_top2_indices_distinct(self):
        """The two selected experts per sample should be distinct."""
        indices, _ = self.gating(self.x)
        for b in range(self.B):
            self.assertNotEqual(indices[b, 0].item(), indices[b, 1].item())


class TestMoE(unittest.TestCase):
    """Tests for the Mixture of Experts module."""

    def setUp(self):
        self.B = 4
        self.d_v = 2048
        self.d_ff = 8192
        self.num_experts = 8
        self.moe = MoE(d_v=self.d_v, d_ff=self.d_ff, num_experts=self.num_experts)
        self.x = torch.randn(self.B, self.d_v)

    def test_output_shape(self):
        """MoE output must be [B, D_v]."""
        out = self.moe(self.x)
        self.assertEqual(out.shape, (self.B, self.d_v))

    def test_num_experts(self):
        """MoE must contain exactly num_experts expert modules."""
        self.assertEqual(len(self.moe.experts), self.num_experts)

    def test_gradients_flow(self):
        """Gradients must propagate through MoE output."""
        x = self.x.requires_grad_(True)
        out = self.moe(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_deterministic_eval(self):
        """MoE must produce identical results on identical inputs (eval mode)."""
        self.moe.eval()
        with torch.no_grad():
            out1 = self.moe(self.x)
            out2 = self.moe(self.x)
        self.assertTrue(torch.allclose(out1, out2))


class TestSynthesizerLayer(unittest.TestCase):
    """End-to-end tests for SynthesizerLayer."""

    def setUp(self):
        self.B = 4
        self.d_v = 2048
        self.d_ff = 8192
        self.num_experts = 8
        self.layer = SynthesizerLayer(
            d_v=self.d_v,
            d_ff=self.d_ff,
            num_experts=self.num_experts,
        )
        self.f_current = torch.randn(self.B, self.d_v)
        self.v_t = torch.randn(self.B, self.d_v)

    # ------------------------------------------------------------------
    # Shape / dtype tests
    # ------------------------------------------------------------------

    def test_output_shape(self):
        """Primary spec: output shape must be [B, D_v]."""
        out = self.layer(self.f_current, self.v_t)
        self.assertEqual(out.shape, (self.B, self.d_v),
                         msg=f"Expected ({self.B}, {self.d_v}), got {out.shape}")

    def test_output_dtype(self):
        """Output dtype must match input dtype (float32)."""
        out = self.layer(self.f_current, self.v_t)
        self.assertEqual(out.dtype, torch.float32)

    def test_batch_size_one(self):
        """Layer must work with batch size of 1."""
        f = torch.randn(1, self.d_v)
        v = torch.randn(1, self.d_v)
        out = self.layer(f, v)
        self.assertEqual(out.shape, (1, self.d_v))

    def test_larger_batch(self):
        """Layer must work with a larger batch."""
        B = 16
        f = torch.randn(B, self.d_v)
        v = torch.randn(B, self.d_v)
        out = self.layer(f, v)
        self.assertEqual(out.shape, (B, self.d_v))

    # ------------------------------------------------------------------
    # Gradient tests
    # ------------------------------------------------------------------

    def test_gradients_flow_through_f_current(self):
        """Gradient must flow back to f_current."""
        f = self.f_current.requires_grad_(True)
        out = self.layer(f, self.v_t)
        out.sum().backward()
        self.assertIsNotNone(f.grad, "No gradient for f_current")
        self.assertFalse(torch.isnan(f.grad).any(), "NaN gradient for f_current")

    def test_gradients_flow_through_v_t(self):
        """Gradient must flow back to v_t."""
        v = self.v_t.requires_grad_(True)
        out = self.layer(self.f_current, v)
        out.sum().backward()
        self.assertIsNotNone(v.grad, "No gradient for v_t")
        self.assertFalse(torch.isnan(v.grad).any(), "NaN gradient for v_t")

    def test_no_nan_in_output(self):
        """Output must not contain NaN values."""
        out = self.layer(self.f_current, self.v_t)
        self.assertFalse(torch.isnan(out).any(), "Output contains NaN")

    def test_no_inf_in_output(self):
        """Output must not contain Inf values."""
        out = self.layer(self.f_current, self.v_t)
        self.assertFalse(torch.isinf(out).any(), "Output contains Inf")

    # ------------------------------------------------------------------
    # Behavioural tests
    # ------------------------------------------------------------------

    def test_different_v_t_gives_different_output(self):
        """Different conditioning vectors must produce different outputs."""
        out1 = self.layer(self.f_current, self.v_t)
        out2 = self.layer(self.f_current, torch.zeros_like(self.v_t))
        self.assertFalse(torch.allclose(out1, out2),
                         "Output unchanged when V_T changes – modulation not working")

    def test_wrong_d_v_raises(self):
        """Mismatched D_v must raise an AssertionError."""
        bad_f = torch.randn(self.B, 512)
        bad_v = torch.randn(self.B, 512)
        with self.assertRaises(AssertionError):
            self.layer(bad_f, bad_v)

    def test_shape_mismatch_raises(self):
        """f_current and v_t with different shapes must raise AssertionError."""
        bad_v = torch.randn(self.B, self.d_v + 1)
        with self.assertRaises(AssertionError):
            self.layer(self.f_current, bad_v)

    def test_eval_mode_deterministic(self):
        """In eval mode identical inputs must produce identical outputs."""
        self.layer.eval()
        with torch.no_grad():
            out1 = self.layer(self.f_current, self.v_t)
            out2 = self.layer(self.f_current, self.v_t)
        self.assertTrue(torch.allclose(out1, out2))

    def test_parameter_count_sanity(self):
        """Check that the model has the expected rough parameter budget."""
        num_params = sum(p.numel() for p in self.layer.parameters())
        # AdaLN film_proj: 2048*(2*2048) + 2*2048 ≈ 8.4M
        # 8 experts each: 2048*8192 + 8192 + 8192*2048 + 2048 ≈ 33.6M per expert → ~269M total
        # gating: 2048*8 + 8 ≈ negligible
        # Total ≥ 8M (very conservative lower bound)
        self.assertGreater(num_params, 8_000_000,
                           f"Unexpectedly few parameters: {num_params}")


if __name__ == '__main__':
    unittest.main()

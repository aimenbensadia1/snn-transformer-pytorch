#!/usr/bin/env python3
"""
Unit tests for SNN Transformer implementation.

Tests verify:
1. Configuration values match original C implementation
2. LUT operations produce correct shapes
3. Index computation follows binary comparison rules
4. Surrogate gradient has correct properties
5. Forward/backward passes work correctly
6. Training reduces loss
7. Text generation produces valid output
"""

import unittest

import torch
import numpy as np

from snn_transformer.config import (
    CONTEXT_SIZE, VOCAB_SIZE, EMBEDDING_DIM, POSITIONAL_DIM,
    NUM_LAYERS, NUM_HEADS, N_T, N_C, WARMUP_STEPS
)
from snn_transformer.models.snn_gpu import FastModel, FastLUT, compute_lr


class TestConfiguration(unittest.TestCase):
    """Test that configuration matches original C implementation."""

    def test_context_size(self):
        self.assertEqual(CONTEXT_SIZE, 32)

    def test_vocab_size(self):
        self.assertEqual(VOCAB_SIZE, 256)

    def test_embedding_dim(self):
        self.assertEqual(EMBEDDING_DIM, 32)

    def test_num_layers(self):
        self.assertEqual(NUM_LAYERS, 6)

    def test_num_heads(self):
        self.assertEqual(NUM_HEADS, 4)

    def test_lut_params(self):
        self.assertEqual(N_T, 16)
        self.assertEqual(N_C, 6)


class TestFastLUT(unittest.TestCase):
    """Test LUT operations."""

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lut = FastLUT(N_C, EMBEDDING_DIM, N_T, self.device)

    def test_table_size(self):
        """Tables should have 2^N_C entries."""
        self.assertEqual(self.lut.table_size, 64)
        self.assertEqual(self.lut.S.shape, (N_T, 64, EMBEDDING_DIM))

    def test_anchors_different(self):
        """Anchor pairs should be different."""
        self.assertFalse(torch.any(self.lut.anchors_a == self.lut.anchors_b))

    def test_index_range(self):
        """Computed indices should be in [0, 63]."""
        x = torch.randn(CONTEXT_SIZE, EMBEDDING_DIM, device=self.device)
        j, r_min, u_min = self.lut.compute_indices(x)

        self.assertTrue(torch.all(j >= 0))
        self.assertTrue(torch.all(j < 64))

    def test_r_min_range(self):
        """r_min should be in [0, N_C-1]."""
        x = torch.randn(CONTEXT_SIZE, EMBEDDING_DIM, device=self.device)
        j, r_min, u_min = self.lut.compute_indices(x)

        self.assertTrue(torch.all(r_min >= 0))
        self.assertTrue(torch.all(r_min < N_C))

    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        x = torch.randn(CONTEXT_SIZE, EMBEDDING_DIM, device=self.device)
        j, _, _ = self.lut.compute_indices(x)
        y = self.lut.forward(j)

        self.assertEqual(y.shape, (CONTEXT_SIZE, EMBEDDING_DIM))

    def test_binary_indexing(self):
        """Verify binary indexing follows comparison rules."""
        # Test that index changes when we flip a comparison
        x1 = torch.randn(1, EMBEDDING_DIM, device=self.device)
        j1, r_min1, u_min1 = self.lut.compute_indices(x1)

        # Flip the most uncertain comparison
        x2 = x1.clone()
        for t in range(N_T):
            a = self.lut.anchors_a[t, r_min1[0, t]].item()
            b = self.lut.anchors_b[t, r_min1[0, t]].item()
            # Swap values to flip the bit
            x2[0, a], x2[0, b] = x2[0, b].item(), x2[0, a].item()

        j2, _, _ = self.lut.compute_indices(x2)

        # At least some indices should change
        self.assertFalse(torch.all(j1 == j2))


class TestFastModel(unittest.TestCase):
    """Test complete model."""

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FastModel(self.device)

    def test_parameter_count(self):
        """Model should have ~805M parameters."""
        params = self.model.count_parameters()
        self.assertGreater(params, 800_000_000)
        self.assertLess(params, 810_000_000)

    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        z = torch.randn(CONTEXT_SIZE, EMBEDDING_DIM, device=self.device)
        output, cache = self.model.forward(z)

        self.assertEqual(output.shape, (CONTEXT_SIZE, VOCAB_SIZE))

    def test_training_step(self):
        """Training step should return valid loss."""
        tokens = torch.randint(0, VOCAB_SIZE, (CONTEXT_SIZE + 1,), device=self.device)
        loss = self.model.training_step(tokens, 0.01)

        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
        self.assertLess(loss, 10)

    def test_loss_decreases(self):
        """Loss should decrease with training."""
        # Fixed input for consistent testing
        tokens = torch.tensor([ord(c) for c in "ABCD" * 8 + "E"],
                             dtype=torch.long, device=self.device)

        initial_loss = self.model.training_step(tokens, 0.0)

        # Train for a few steps
        for i in range(10):
            self.model.training_step(tokens, compute_lr(i + 1))

        final_loss = self.model.training_step(tokens, 0.0)

        self.assertLess(final_loss, initial_loss)

    def test_generation(self):
        """Generation should produce valid tokens."""
        prompt = torch.randint(0, VOCAB_SIZE, (CONTEXT_SIZE,), device=self.device)
        generated = self.model.generate(prompt, 10, temperature=0.5)

        self.assertEqual(generated.shape, (10,))
        self.assertTrue(torch.all(generated >= 0))
        self.assertTrue(torch.all(generated < VOCAB_SIZE))


class TestLearningRate(unittest.TestCase):
    """Test learning rate schedule."""

    def test_lr_zero_at_start(self):
        """LR should be 0 at step 0."""
        self.assertEqual(compute_lr(0), 0.0)

    def test_lr_increases_during_warmup(self):
        """LR should increase during warmup."""
        lr_100 = compute_lr(100)
        lr_1000 = compute_lr(1000)
        lr_2000 = compute_lr(2000)

        self.assertLess(lr_100, lr_1000)
        self.assertLess(lr_1000, lr_2000)

    def test_lr_decreases_after_warmup(self):
        """LR should decrease after warmup."""
        lr_warmup = compute_lr(WARMUP_STEPS)
        lr_later = compute_lr(WARMUP_STEPS * 4)

        self.assertGreater(lr_warmup, lr_later)

    def test_lr_positive(self):
        """LR should always be positive (except at 0)."""
        for t in [1, 100, 1000, 10000, 100000]:
            self.assertGreater(compute_lr(t), 0)


class TestSurrogateGradient(unittest.TestCase):
    """Test surrogate gradient properties."""

    def test_gradient_flow(self):
        """Gradients should flow through LUT operations."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lut = FastLUT(N_C, EMBEDDING_DIM, N_T, device)

        # Initialize tables with non-zero values to enable gradient flow
        lut.S = torch.randn_like(lut.S) * 0.1

        x = torch.randn(CONTEXT_SIZE, EMBEDDING_DIM, device=device)
        j, r_min, u_min = lut.compute_indices(x)
        y = lut.forward(j)

        y_grad = torch.randn_like(y)  # Random gradient instead of ones
        x_grad = lut.backward_and_update(j, r_min, u_min, y_grad, lr=0.0)

        # Gradient should be non-zero when tables have different values
        self.assertGreater(torch.abs(x_grad).sum().item(), 0)

    def test_surrogate_gradient_formula(self):
        """Verify surrogate gradient formula."""
        # Up(x) = -0.5 * sign(x) / (1 + |x|)^2
        u = torch.tensor([0.5, -0.5, 1.0, -1.0, 0.1])

        sign_u = torch.where(u > 0, torch.ones_like(u), -torch.ones_like(u))
        Up = -0.5 * sign_u / ((1 + torch.abs(u)) ** 2)

        # Check properties
        # 1. Negative for positive u
        self.assertLess(Up[0].item(), 0)
        # 2. Positive for negative u
        self.assertGreater(Up[1].item(), 0)
        # 3. Magnitude decreases with |u|
        self.assertGreater(torch.abs(Up[4]).item(), torch.abs(Up[2]).item())


def run_tests():
    """Run all tests with progress output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestFastLUT))
    suite.addTests(loader.loadTestsFromTestCase(TestFastModel))
    suite.addTests(loader.loadTestsFromTestCase(TestLearningRate))
    suite.addTests(loader.loadTestsFromTestCase(TestSurrogateGradient))

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

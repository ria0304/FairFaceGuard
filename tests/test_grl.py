"""Unit tests for Gradient Reversal Layer (GRL).

This test verifies the mathematical correctness of the GRL implementation:
1. Forward pass: identity function (output = input)
2. Backward pass: gradient reversal with correct lambda scaling

The GRL is critical for adversarial disentanglement. If implemented incorrectly,
the entire adversarial training objective fails.
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.disentangle.grl import GradientReversalLayer, _GradReverseFn, dann_lambda_schedule


def test_forward_pass_identity():
    """Test that forward pass is identity: f(x) = x."""
    grl = GradientReversalLayer(lambd=1.0)
    
    # Test various input shapes
    test_cases = [
        (32, 128),      # batch x features
        (16, 512, 7, 7),  # batch x channels x height x width
        (1, 10),        # single sample
        (64, 256),      # large batch
    ]
    
    for shape in test_cases:
        x = torch.randn(*shape, requires_grad=False)
        output = grl(x)
        
        # Check shape preservation
        assert output.shape == x.shape, f"Shape mismatch for input {shape}"
        
        # Check value preservation (forward is identity)
        assert torch.allclose(output, x, atol=1e-6), f"Forward pass not identity for shape {shape}"
    
    print("[test_grl] Forward pass identity: PASSED")


def test_backward_pass_reversal_lambda_1():
    """Test that backward pass reverses gradient with lambda=1.0.
    
    Mathematical expectation:
    - loss = sum(grl(x))
    - d(loss)/dx should be -lambda * ones = -1.0 * ones
    """
    lambd = 1.0
    grl = GradientReversalLayer(lambd=lambd)
    
    x = torch.randn(32, 128, requires_grad=True)
    output = grl(x)
    
    # Create a simple loss (sum of outputs)
    loss = output.sum()
    
    # Backpropagate
    loss.backward()
    
    # Check gradient: should be -lambda * grad_output = -1.0 * ones
    expected_grad = -lambd * torch.ones_like(x)
    
    assert x.grad is not None, "Gradient is None"
    assert torch.allclose(x.grad, expected_grad, atol=1e-6), \
        f"Gradient reversal failed. Expected all {-lambd}, got mean={x.grad.mean().item():.6f}"
    
    print(f"[test_grl] Backward pass reversal (lambda={lambd}): PASSED")


def test_backward_pass_reversal_lambda_0():
    """Test that lambda=0 blocks gradient (no reversal, just zero).
    
    Mathematical expectation:
    - loss = sum(grl(x))
    - d(loss)/dx should be -lambda * ones = 0.0 * ones = zeros
    """
    lambd = 0.0
    grl = GradientReversalLayer(lambd=lambd)
    
    x = torch.randn(32, 128, requires_grad=True)
    output = grl(x)
    
    loss = output.sum()
    loss.backward()
    
    # Check gradient: should be -lambda * grad_output = 0.0
    expected_grad = torch.zeros_like(x)
    
    assert x.grad is not None, "Gradient is None"
    assert torch.allclose(x.grad, expected_grad, atol=1e-6), \
        f"Lambda=0 should block gradients. Expected all 0, got mean={x.grad.mean().item():.6f}"
    
    print(f"[test_grl] Backward pass blocking (lambda={lambd}): PASSED")


def test_backward_pass_reversal_lambda_2():
    """Test gradient reversal with lambda=2.0.
    
    Mathematical expectation:
    - loss = sum(grl(x))
    - d(loss)/dx should be -lambda * ones = -2.0 * ones
    """
    lambd = 2.0
    grl = GradientReversalLayer(lambd=lambd)
    
    x = torch.randn(32, 128, requires_grad=True)
    output = grl(x)
    
    loss = output.sum()
    loss.backward()
    
    # Check gradient: should be -lambda * grad_output = -2.0 * ones
    expected_grad = -lambd * torch.ones_like(x)
    
    assert x.grad is not None, "Gradient is None"
    assert torch.allclose(x.grad, expected_grad, atol=1e-6), \
        f"Gradient reversal failed for lambda={lambd}. Expected all {-lambd}, got mean={x.grad.mean().item():.6f}"
    
    print(f"[test_grl] Backward pass reversal (lambda={lambd}): PASSED")


def test_backward_pass_reversal_lambda_negative():
    """Test gradient reversal with negative lambda (edge case).
    
    This tests if the implementation handles negative lambda correctly.
    Mathematical expectation:
    - loss = sum(grl(x))
    - d(loss)/dx should be -lambda * ones = -(-0.5) * ones = +0.5 * ones
    """
    lambd = -0.5
    grl = GradientReversalLayer(lambd=lambd)
    
    x = torch.randn(32, 128, requires_grad=True)
    output = grl(x)
    
    loss = output.sum()
    loss.backward()
    
    # Check gradient: should be -lambda * grad_output = -(-0.5) * ones = +0.5
    expected_grad = -lambd * torch.ones_like(x)
    
    assert x.grad is not None, "Gradient is None"
    assert torch.allclose(x.grad, expected_grad, atol=1e-6), \
        f"Gradient reversal failed for lambda={lambd}. Expected all {-lambd}, got mean={x.grad.mean().item():.6f}"
    
    print(f"[test_grl] Backward pass reversal (lambda={lambd}): PASSED")


def test_gradient_scaling_various_lambdas():
    """Test gradient scaling across multiple lambda values."""
    lambda_values = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]
    
    for lambd in lambda_values:
        grl = GradientReversalLayer(lambd=lambd)
        
        x = torch.randn(32, 128, requires_grad=True)
        output = grl(x)
        
        loss = output.sum()
        loss.backward()
        
        expected_grad = -lambd * torch.ones_like(x)
        
        assert torch.allclose(x.grad, expected_grad, atol=1e-6), \
            f"Gradient scaling failed for lambda={lambd}"
        
        # Clear gradient for next iteration
        x.grad = None
    
    print(f"[test_grl] Gradient scaling for lambdas {lambda_values}: PASSED")


def test_set_lambda():
    """Test dynamic lambda adjustment via set_lambda()."""
    grl = GradientReversalLayer(lambd=1.0)
    
    # Verify initial lambda
    assert grl.lambd == 1.0, "Initial lambda incorrect"
    
    # Change lambda
    grl.set_lambda(2.5)
    assert grl.lambd == 2.5, "set_lambda failed"
    
    # Test that new lambda affects gradient
    x = torch.randn(32, 128, requires_grad=True)
    output = grl(x)
    loss = output.sum()
    loss.backward()
    
    expected_grad = -2.5 * torch.ones_like(x)
    assert torch.allclose(x.grad, expected_grad, atol=1e-6), \
        f"Dynamic lambda update failed. Expected {-2.5}, got {x.grad.mean().item():.6f}"
    
    print("[test_grl] Dynamic lambda adjustment: PASSED")


def test_dann_lambda_schedule():
    """Test the DANN lambda warm-up schedule.
    
    The schedule should:
    - Start near 0 when progress=0
    - Approach 1 when progress=1
    - Be monotonically increasing
    """
    # Test boundary conditions
    lambda_at_0 = dann_lambda_schedule(0.0)
    lambda_at_1 = dann_lambda_schedule(1.0)
    
    assert abs(lambda_at_0) < 0.1, f"Lambda at progress=0 should be near 0, got {lambda_at_0}"
    assert lambda_at_1 > 0.9, f"Lambda at progress=1 should be near 1, got {lambda_at_1}"
    
    # Test monotonicity
    progresses = [i / 10 for i in range(11)]
    lambdas = [dann_lambda_schedule(p) for p in progresses]
    
    for i in range(len(lambdas) - 1):
        assert lambdas[i] <= lambdas[i + 1], \
            f"Lambda schedule not monotonic: {lambdas[i]} > {lambdas[i+1]}"
    
    # Test clamping (progress outside [0, 1])
    lambda_neg = dann_lambda_schedule(-0.5)
    lambda_over = dann_lambda_schedule(1.5)
    
    assert abs(lambda_neg) < 0.1, f"Negative progress should clamp to ~0, got {lambda_neg}"
    assert lambda_over > 0.9, f"Progress > 1 should clamp to ~1, got {lambda_over}"
    
    print("[test_grl] DANN lambda schedule: PASSED")


def test_full_adversarial_training_step():
    """Test GRL in a mini adversarial training scenario.
    
    This simulates the actual use case:
    1. Feature extractor produces features z
    2. GRL reverses gradient from adversarial probe
    3. Feature extractor learns to fool the probe
    """
    # Setup: feature extractor, GRL, and adversarial probe
    feature_extractor = nn.Linear(128, 64)
    grl = GradientReversalLayer(lambd=1.0)
    adv_probe = nn.Linear(64, 2)  # Predict skin tone (2 classes)
    
    # Simulate data
    x = torch.randn(32, 128, requires_grad=False)
    skin_labels = torch.randint(0, 2, (32,))  # Binary skin tone labels
    
    # Forward pass
    z = feature_extractor(x)
    z_reversed = grl(z)
    skin_pred = adv_probe(z_reversed)
    
    # Adversarial loss (probe tries to predict skin tone)
    adv_loss = nn.CrossEntropyLoss()(skin_pred, skin_labels)
    
    # Backpropagate
    adv_loss.backward()
    
    # Check that feature_extractor received reversed gradients
    assert feature_extractor.weight.grad is not None, "Feature extractor has no gradient"
    
    # The key test: gradients should flow through GRL with sign reversal
    # We can verify this by checking that gradients exist and have reasonable magnitude
    grad_norm = feature_extractor.weight.grad.norm().item()
    assert grad_norm > 0, "Feature extractor gradient norm is zero"
    assert not torch.isnan(feature_extractor.weight.grad).any(), "NaN gradients detected"
    assert not torch.isinf(feature_extractor.weight.grad).any(), "Inf gradients detected"
    
    print(f"[test_grl] Full adversarial training step: PASSED (grad_norm={grad_norm:.4f})")


def test_multiple_grl_layers():
    """Test that multiple GRL layers work correctly in sequence."""
    grl1 = GradientReversalLayer(lambd=0.5)
    grl2 = GradientReversalLayer(lambd=2.0)
    
    x = torch.randn(32, 128, requires_grad=True)
    
    # Sequential GRL: total lambda effect = 0.5 * 2.0 = 1.0
    out1 = grl1(x)
    out2 = grl2(out1)
    
    loss = out2.sum()
    loss.backward()
    
    # Combined effect: -0.5 * -2.0 = +1.0 (double reversal = normal gradient)
    expected_grad = torch.ones_like(x)
    
    assert torch.allclose(x.grad, expected_grad, atol=1e-6), \
        f"Multiple GRL layers failed. Expected +1.0 (double reversal), got {x.grad.mean().item():.6f}"
    
    print("[test_grl] Multiple GRL layers: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("GRADIENT REVERSAL LAYER UNIT TESTS")
    print("=" * 60)
    
    test_forward_pass_identity()
    test_backward_pass_reversal_lambda_1()
    test_backward_pass_reversal_lambda_0()
    test_backward_pass_reversal_lambda_2()
    test_backward_pass_reversal_lambda_negative()
    test_gradient_scaling_various_lambdas()
    test_set_lambda()
    test_dann_lambda_schedule()
    test_full_adversarial_training_step()
    test_multiple_grl_layers()
    
    print("=" * 60)
    print("ALL GRL TESTS PASSED")
    print("=" * 60)

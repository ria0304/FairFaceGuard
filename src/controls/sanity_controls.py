"""Negative and sanity controls for FairFaceGuard research.

These controls are essential for validating that observed effects are genuine
and not artifacts of dataset construction, implementation bugs, or spurious correlations.

Controls implemented:
    A. Shuffled skin labels - verifies disentanglement requires real labels
    B. Shuffled illumination labels - verifies disentanglement requires real labels  
    C. Lambda = 0 control - verifies adversarial training reduces to baseline
    D. Non-face perturbation - verifies counterfactual effects are face-specific
    E. Intervention magnitude - verifies manipulation produces measurable change

Each control should be automatically evaluated and included in the final report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Any, Callable
import torch
import torch.nn as nn


@dataclass
class ControlResult:
    """Result from a single sanity control."""
    control_name: str
    description: str
    expected_behavior: str
    passed: bool
    metrics: dict
    notes: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


def control_shuffled_skin_labels(
    skin_labels: np.ndarray,
    illumination_labels: np.ndarray,
    fake_labels: np.ndarray,
    model_factory: Callable,
    train_fn: Callable,
    eval_fn: Callable,
    seed: int = 42,
) -> ControlResult:
    """Control A: Shuffled skin labels.
    
    Rationale: If we randomly shuffle skin labels while keeping everything else
    the same, the adversarial disentanglement should NOT produce meaningful
    skin-invariant representations. The probe accuracy on shuffled labels should
    be at chance level (~50% for binary, ~1/n for n-class).
    
    Expected behavior:
    - Skin probe accuracy should be near chance level
    - No meaningful reduction in skin information in representations
    - Subgroup gaps should not systematically decrease
    
    Args:
        skin_labels: Original skin tone labels.
        illumination_labels: Illumination labels (unchanged).
        fake_labels: Fake/real labels.
        model_factory: Function to create a fresh model.
        train_fn: Training function.
        eval_fn: Evaluation function.
        seed: Random seed for shuffling.
    
    Returns:
        ControlResult with pass/fail and metrics.
    """
    rng = np.random.default_rng(seed)
    
    # Shuffle skin labels
    shuffled_skin = rng.permutation(skin_labels)
    
    # Train model with shuffled labels
    model_shuffled = model_factory()
    train_results = train_fn(
        model=model_shuffled,
        skin_labels=shuffled_skin,
        illum_labels=illumination_labels,
        fake_labels=fake_labels,
    )
    
    # Evaluate
    eval_results = eval_fn(model_shuffled)
    
    # Check: skin probe accuracy should be near chance
    n_skin_classes = len(np.unique(skin_labels))
    chance_accuracy = 1.0 / n_skin_classes
    
    skin_probe_acc = eval_results.get("skin_probe_accuracy", 0.5)
    
    # Passed if probe accuracy is within 3 standard errors of chance
    n_samples = len(skin_labels)
    se = np.sqrt(chance_accuracy * (1 - chance_accuracy) / n_samples)
    
    passed = skin_probe_acc <= (chance_accuracy + 3 * se)
    
    return ControlResult(
        control_name="shuffled_skin_labels",
        description="Skin labels randomly shuffled before adversarial training",
        expected_behavior=f"Skin probe accuracy near chance ({chance_accuracy:.3f})",
        passed=passed,
        metrics={
            "skin_probe_accuracy": float(skin_probe_acc),
            "chance_accuracy": float(chance_accuracy),
            "n_std_errors": float((skin_probe_acc - chance_accuracy) / se) if se > 0 else float('inf'),
            "training_completed": train_results.get("completed", False),
        },
        notes="" if passed else f"Skin probe acc {skin_probe_acc:.3f} exceeds chance + 3SE",
    )


def control_shuffled_illumination_labels(
    skin_labels: np.ndarray,
    illumination_labels: np.ndarray,
    fake_labels: np.ndarray,
    model_factory: Callable,
    train_fn: Callable,
    eval_fn: Callable,
    seed: int = 42,
) -> ControlResult:
    """Control B: Shuffled illumination labels.
    
    Rationale: Same as Control A but for illumination. Randomly shuffling
    illumination labels should prevent meaningful illumination disentanglement.
    
    Expected behavior:
    - Illumination probe accuracy should be near chance level
    - No meaningful reduction in illumination information
    
    Returns:
        ControlResult with pass/fail and metrics.
    """
    rng = np.random.default_rng(seed)
    
    # Shuffle illumination labels
    shuffled_illum = rng.permutation(illumination_labels)
    
    # Train model with shuffled labels
    model_shuffled = model_factory()
    train_results = train_fn(
        model=model_shuffled,
        skin_labels=skin_labels,
        illum_labels=shuffled_illum,
        fake_labels=fake_labels,
    )
    
    # Evaluate
    eval_results = eval_fn(model_shuffled)
    
    # Check: illumination probe accuracy should be near chance
    n_illum_classes = len(np.unique(illumination_labels))
    chance_accuracy = 1.0 / n_illum_classes
    
    illum_probe_acc = eval_results.get("illum_probe_accuracy", 0.5)
    
    n_samples = len(illumination_labels)
    se = np.sqrt(chance_accuracy * (1 - chance_accuracy) / n_samples)
    
    passed = illum_probe_acc <= (chance_accuracy + 3 * se)
    
    return ControlResult(
        control_name="shuffled_illumination_labels",
        description="Illumination labels randomly shuffled before adversarial training",
        expected_behavior=f"Illumination probe accuracy near chance ({chance_accuracy:.3f})",
        passed=passed,
        metrics={
            "illum_probe_accuracy": float(illum_probe_acc),
            "chance_accuracy": float(chance_accuracy),
            "n_std_errors": float((illum_probe_acc - chance_accuracy) / se) if se > 0 else float('inf'),
            "training_completed": train_results.get("completed", False),
        },
        notes="" if passed else f"Illum probe acc {illum_probe_acc:.3f} exceeds chance + 3SE",
    )


def control_lambda_zero(
    model_factory: Callable,
    train_fn_baseline: Callable,
    train_fn_adversarial: Callable,
    eval_fn: Callable,
    lambda_skin: float = 0.0,
    lambda_illum: float = 0.0,
) -> ControlResult:
    """Control C: Lambda = 0 control.
    
    Rationale: When lambda=0, the gradient reversal has no effect, so adversarial
    training should behave identically to standard baseline training.
    
    Expected behavior:
    - Model performance should match baseline
    - Probe accuracies should be unchanged from baseline
    - No disentanglement effect
    
    Returns:
        ControlResult with pass/fail and metrics.
    """
    # Train baseline
    model_baseline = model_factory()
    baseline_results = train_fn_baseline(model_baseline)
    baseline_eval = eval_fn(model_baseline)
    
    # Train adversarial with lambda=0
    model_lambda0 = model_factory()
    adv_results = train_fn_adversarial(
        model=model_lambda0,
        lambda_skin=lambda_skin,
        lambda_illum=lambda_illum,
    )
    adv_eval = eval_fn(model_lambda0)
    
    # Compare: should be nearly identical
    baseline_acc = baseline_eval.get("accuracy", 0.5)
    lambda0_acc = adv_eval.get("accuracy", 0.5)
    
    baseline_auc = baseline_eval.get("auc", 0.5)
    lambda0_auc = adv_eval.get("auc", 0.5)
    
    # Allow small numerical differences due to random initialization
    acc_diff = abs(baseline_acc - lambda0_acc)
    auc_diff = abs(baseline_auc - lambda0_auc)
    
    passed = (acc_diff < 0.05) and (auc_diff < 0.05)
    
    return ControlResult(
        control_name="lambda_zero",
        description="Adversarial training with lambda=0 (should equal baseline)",
        expected_behavior="Metrics match baseline training",
        passed=passed,
        metrics={
            "baseline_accuracy": float(baseline_acc),
            "lambda0_accuracy": float(lambda0_acc),
            "accuracy_difference": float(acc_diff),
            "baseline_auc": float(baseline_auc),
            "lambda0_auc": float(lambda0_auc),
            "auc_difference": float(auc_diff),
        },
        notes="" if passed else f"Accuracy diff {acc_diff:.3f} or AUC diff {auc_diff:.3f} exceeds threshold",
    )


def control_non_face_perturbation(
    original_images: np.ndarray,
    face_masks: np.ndarray,
    detector_fn: Callable,
    n_samples: int = 100,
    seed: int = 42,
) -> ControlResult:
    """Control D: Non-face perturbation control.
    
    Rationale: If we apply the same perturbation outside the facial region,
    it should NOT produce the same counterfactual effect claimed for skin/
    illumination intervention. This verifies that effects are face-specific.
    
    Expected behavior:
    - Perturbing non-face regions should have minimal effect on detector output
    - Delta predictions should be much smaller than face perturbations
    
    Args:
        original_images: Array of original face images.
        face_masks: Binary masks indicating face region.
        detector_fn: Detector prediction function.
        n_samples: Number of samples to test.
        seed: Random seed.
    
    Returns:
        ControlResult with pass/fail and metrics.
    """
    rng = np.random.default_rng(seed)
    
    if len(original_images) < n_samples:
        n_samples = len(original_images)
    
    sample_indices = rng.choice(len(original_images), min(n_samples, len(original_images)), replace=False)
    
    # Compute detector change for non-face perturbations
    delta_predictions = []
    
    for idx in sample_indices:
        img = original_images[idx]
        mask = face_masks[idx]
        
        # Create non-face perturbation (e.g., brighten background)
        perturbed = img.copy().astype(np.float32)
        
        # Apply perturbation only OUTSIDE face mask
        bg_mask = (mask < 0.5).astype(bool)
        if bg_mask.sum() > 0:
            # Brighten background by 20%
            perturbed[bg_mask] = np.clip(perturbed[bg_mask] * 1.2, 0, 255)
        
        # Get detector predictions
        orig_pred = detector_fn(img.astype(np.float32)[np.newaxis, ...])[0]
        pert_pred = detector_fn(perturbed[np.newaxis, ...])[0]
        
        delta_predictions.append(abs(orig_pred - pert_pred))
    
    delta_predictions = np.array(delta_predictions)
    mean_delta = float(delta_predictions.mean())
    std_delta = float(delta_predictions.std())
    
    # Expected: non-face perturbations should have small effect (< 0.05 change)
    threshold = 0.05
    passed = mean_delta < threshold
    
    return ControlResult(
        control_name="non_face_perturbation",
        description="Perturbation applied outside facial region",
        expected_behavior=f"Mean |delta prediction| < {threshold}",
        passed=passed,
        metrics={
            "mean_absolute_delta": mean_delta,
            "std_delta": std_delta,
            "n_samples": int(n_samples),
            "max_delta": float(delta_predictions.max()),
            "min_delta": float(delta_predictions.min()),
        },
        notes="" if passed else f"Non-face perturbation effect {mean_delta:.4f} exceeds threshold {threshold}",
    )


def control_intervention_magnitude(
    original_images: np.ndarray,
    counterfactual_fn: Callable,
    ita_fn: Callable,
    n_samples: int = 50,
    seed: int = 42,
) -> ControlResult:
    """Control E: Intervention magnitude validation.
    
    Rationale: Larger intended intervention magnitudes should produce larger
    measurable changes in the intended variable. This validates that the
    counterfactual generator actually manipulates the intended factor.
    
    Expected behavior:
    - Larger ITA delta should produce larger measured ITA change
    - Monotonic relationship between intended and actual magnitude
    
    Args:
        original_images: Array of original images.
        counterfactual_fn: Counterfactual generation function.
        ita_fn: Function to compute ITA from image.
        n_samples: Number of samples to test.
        seed: Random seed.
    
    Returns:
        ControlResult with pass/fail and metrics.
    """
    rng = np.random.default_rng(seed)
    
    if len(original_images) < n_samples:
        n_samples = len(original_images)
    
    sample_indices = rng.choice(len(original_images), min(n_samples, len(original_images)), replace=False)
    
    # Test different intervention magnitudes
    ita_deltas = [5.0, 10.0, 15.0, 20.0, 25.0]
    
    measured_changes = {delta: [] for delta in ita_deltas}
    
    for idx in sample_indices:
        img = original_images[idx]
        orig_ita = ita_fn(img)
        
        for delta in ita_deltas:
            try:
                cf_set = counterfactual_fn(img, face_id=f"sample_{idx}", ita_delta=delta)
                skin_ita = ita_fn(cf_set.skin_only)
                measured_change = abs(skin_ita - orig_ita)
                measured_changes[delta].append(measured_change)
            except Exception as e:
                measured_changes[delta].append(0.0)
    
    # Compute mean measured change for each intended delta
    mean_changes = {delta: float(np.mean(changes)) for delta, changes in measured_changes.items()}
    
    # Check monotonicity: larger intended delta should give larger measured change
    deltas_sorted = sorted(ita_deltas)
    changes_sorted = [mean_changes[d] for d in deltas_sorted]
    
    # Count monotonic pairs
    monotonic_count = sum(
        1 for i in range(len(changes_sorted) - 1) 
        if changes_sorted[i] <= changes_sorted[i + 1] + 0.5  # Small tolerance
    )
    monotonicity_ratio = float(monotonic_count) / max(1, len(changes_sorted) - 1)
    
    # Also check correlation
    if len(deltas_sorted) >= 3:
        try:
            from scipy.stats import spearmanr
            correlation, p_value = spearmanr(deltas_sorted, changes_sorted)
            correlation = float(correlation)
            p_value = float(p_value)
        except ImportError:
            # Simple correlation calculation
            x = np.array(deltas_sorted)
            y = np.array(changes_sorted)
            correlation = float(np.corrcoef(x, y)[0, 1])
            p_value = 0.0
    else:
        correlation, p_value = 1.0, 0.0
    
    # Passed if reasonably monotonic (ratio > 0.75) and positive correlation
    passed = (monotonicity_ratio > 0.75) and (correlation > 0.5)
    
    return ControlResult(
        control_name="intervention_magnitude",
        description="Verify intervention magnitude produces proportional change",
        expected_behavior="Monotonic relationship between intended and measured change",
        passed=passed,
        metrics={
            "intended_deltas": ita_deltas,
            "measured_changes": [float(c) for c in changes_sorted],
            "monotonicity_ratio": monotonicity_ratio,
            "spearman_correlation": correlation,
            "correlation_p_value": p_value,
        },
        notes="" if passed else f"Monotonicity ratio {monotonicity_ratio:.2f} or correlation {correlation:.2f} too low",
    )


def run_all_controls(
    data_dict: dict,
    model_configs: dict,
    output_dir: str,
) -> dict:
    """Run all sanity controls and generate summary report.
    
    Args:
        data_dict: Dictionary containing all required data arrays.
        model_configs: Configuration for model creation and training.
        output_dir: Directory to save control results.
    
    Returns:
        Dictionary with all control results and summary.
    """
    import os
    import json
    
    results = {}
    summary = {"total": 0, "passed": 0, "failed": 0}
    
    print("=" * 60)
    print("RUNNING SANITY CONTROLS")
    print("=" * 60)
    
    # Control A: Shuffled skin labels
    print("\n[Control A] Shuffled skin labels...")
    try:
        result_a = control_shuffled_skin_labels(**data_dict, **model_configs)
        results["shuffled_skin"] = result_a.to_dict()
        summary["total"] += 1
        if result_a.passed:
            summary["passed"] += 1
            print("  PASSED")
        else:
            summary["failed"] += 1
            print(f"  FAILED: {result_a.notes}")
    except Exception as e:
        results["shuffled_skin"] = {"error": str(e)}
        summary["total"] += 1
        summary["failed"] += 1
        print(f"  ERROR: {e}")
    
    # Control B: Shuffled illumination labels
    print("\n[Control B] Shuffled illumination labels...")
    try:
        result_b = control_shuffled_illumination_labels(**data_dict, **model_configs)
        results["shuffled_illum"] = result_b.to_dict()
        summary["total"] += 1
        if result_b.passed:
            summary["passed"] += 1
            print("  PASSED")
        else:
            summary["failed"] += 1
            print(f"  FAILED: {result_b.notes}")
    except Exception as e:
        results["shuffled_illum"] = {"error": str(e)}
        summary["total"] += 1
        summary["failed"] += 1
        print(f"  ERROR: {e}")
    
    # Control C: Lambda = 0
    print("\n[Control C] Lambda = 0...")
    try:
        result_c = control_lambda_zero(**model_configs)
        results["lambda_zero"] = result_c.to_dict()
        summary["total"] += 1
        if result_c.passed:
            summary["passed"] += 1
            print("  PASSED")
        else:
            summary["failed"] += 1
            print(f"  FAILED: {result_c.notes}")
    except Exception as e:
        results["lambda_zero"] = {"error": str(e)}
        summary["total"] += 1
        summary["failed"] += 1
        print(f"  ERROR: {e}")
    
    # Control D: Non-face perturbation
    print("\n[Control D] Non-face perturbation...")
    try:
        result_d = control_non_face_perturbation(**data_dict)
        results["non_face_perturbation"] = result_d.to_dict()
        summary["total"] += 1
        if result_d.passed:
            summary["passed"] += 1
            print("  PASSED")
        else:
            summary["failed"] += 1
            print(f"  FAILED: {result_d.notes}")
    except Exception as e:
        results["non_face_perturbation"] = {"error": str(e)}
        summary["total"] += 1
        summary["failed"] += 1
        print(f"  ERROR: {e}")
    
    # Control E: Intervention magnitude
    print("\n[Control E] Intervention magnitude...")
    try:
        result_e = control_intervention_magnitude(**data_dict)
        results["intervention_magnitude"] = result_e.to_dict()
        summary["total"] += 1
        if result_e.passed:
            summary["passed"] += 1
            print("  PASSED")
        else:
            summary["failed"] += 1
            print(f"  FAILED: {result_e.notes}")
    except Exception as e:
        results["intervention_magnitude"] = {"error": str(e)}
        summary["total"] += 1
        summary["failed"] += 1
        print(f"  ERROR: {e}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "sanity_controls.json")
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"CONTROLS SUMMARY: {summary['passed']}/{summary['total']} passed")
    print("=" * 60)
    
    return {"summary": summary, "results": results}


if __name__ == "__main__":
    print("Sanity controls module loaded successfully.")
    print("Use run_all_controls() to execute all controls.")

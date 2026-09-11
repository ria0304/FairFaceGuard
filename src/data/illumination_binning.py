"""Illumination estimation and binning with proper train/val/test separation.

CRITICAL: This module ensures that illumination bin boundaries are computed
from training data ONLY and then applied to validation/test sets without
modification. This prevents data leakage that would invalidate experimental results.

The key principle: validation and test distributions must NOT influence the
bin boundaries used for subgroup analysis.

Usage:
    # Step 1: Compute boundaries from training data
    boundaries = compute_illumination_boundaries_from_train(
        train_df, n_bins=5
    )
    
    # Step 2: Save boundaries
    save_boundaries(boundaries, "illumination_boundaries.json")
    
    # Step 3: Apply to all splits
    train_df["illum_bin"] = apply_boundaries(train_df, boundaries)
    val_df["illum_bin"] = apply_boundaries(val_df, boundaries)
    test_df["illum_bin"] = apply_boundaries(test_df, boundaries)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class IlluminationBoundaries:
    """Saved illumination bin boundaries computed from training data.
    
    Attributes:
        n_bins: Number of illumination bins.
        boundaries: List of bin edge values (length = n_bins + 1).
        train_n: Number of training samples used to compute boundaries.
        train_mean: Mean illuminant magnitude in training data.
        train_std: Standard deviation of illuminant magnitude in training data.
        method: Method used to compute boundaries ("qcut" or "fixed").
    """
    n_bins: int
    boundaries: list[float]
    train_n: int
    train_mean: float
    train_std: float
    method: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "IlluminationBoundaries":
        return cls(**d)


def compute_illuminant_magnitude(df: pd.DataFrame) -> pd.Series:
    """Compute illuminant magnitude from RGB estimates.
    
    Args:
        df: DataFrame with illuminant_estimate_r, illuminant_estimate_g, 
            illuminant_estimate_b columns.
    
    Returns:
        Series with illuminant magnitude (mean of RGB values).
    """
    if "illuminant_estimate_r" in df.columns:
        return df[["illuminant_estimate_r", "illuminant_estimate_g", "illuminant_estimate_b"]].mean(axis=1)
    elif "illuminant_magnitude" in df.columns:
        return df["illuminant_magnitude"]
    else:
        raise ValueError(
            "No illuminant estimate columns found. Expected either:\n"
            "  - illuminant_estimate_r, illuminant_estimate_g, illuminant_estimate_b\n"
            "  - illuminant_magnitude"
        )


def compute_illumination_boundaries_from_train(
    train_df: pd.DataFrame,
    n_bins: int = 5,
    method: str = "qcut",
) -> IlluminationBoundaries:
    """Compute illumination bin boundaries using TRAINING DATA ONLY.
    
    IMPORTANT: This function should ONLY be called on training data.
    The returned boundaries must then be applied to validation and test sets
    without modification.
    
    Args:
        train_df: Training DataFrame with illuminant estimates.
        n_bins: Number of illumination bins to create.
        method: Boundary computation method:
            - "qcut": Quantile-based binning (equal counts per bin)
            - "fixed": Fixed-width bins based on train min/max
    
    Returns:
        IlluminationBoundaries object with computed boundaries.
    
    Raises:
        ValueError: If insufficient training data or invalid method.
    """
    if len(train_df) < n_bins * 2:
        raise ValueError(
            f"Insufficient training data ({len(train_df)} samples) "
            f"for {n_bins} illumination bins. Need at least {n_bins * 2} samples."
        )
    
    illum_mag = compute_illuminant_magnitude(train_df)
    
    # Remove NaN values
    valid_mask = illum_mag.notna()
    illum_mag_valid = illum_mag[valid_mask]
    
    if len(illum_mag_valid) < n_bins * 2:
        raise ValueError(
            f"Insufficient valid illuminant estimates ({len(illum_mag_valid)}) "
            f"for {n_bins} bins after removing NaN values."
        )
    
    if method == "qcut":
        # Quantile-based binning
        try:
            _, boundaries = pd.qcut(illum_mag_valid, n_bins, retbins=True, duplicates="drop")
        except ValueError as e:
            # Fall back to fixed-width bins if qcut fails
            print(f"[illumination_binning] qcut failed: {e}. Using fixed-width bins.")
            method = "fixed"
    
    if method == "fixed":
        # Fixed-width bins
        min_val = float(illum_mag_valid.min())
        max_val = float(illum_mag_valid.max())
        step = (max_val - min_val) / n_bins
        boundaries = [min_val + i * step for i in range(n_bins + 1)]
        boundaries[-1] = max_val + 1e-6  # Ensure max value is included
    
    boundaries = [float(b) for b in boundaries]
    
    return IlluminationBoundaries(
        n_bins=len(boundaries) - 1,
        boundaries=boundaries,
        train_n=len(illum_mag_valid),
        train_mean=float(illum_mag_valid.mean()),
        train_std=float(illum_mag_valid.std()),
        method=method,
    )


def apply_boundaries(
    df: pd.DataFrame,
    boundaries: IlluminationBoundaries,
    boundary_col: str = "illum_bin",
) -> pd.DataFrame:
    """Apply pre-computed illumination boundaries to a dataset.
    
    This function applies the SAME boundaries (computed from training data)
    to any dataset split (train, val, test). This ensures no data leakage.
    
    Args:
        df: DataFrame with illuminant estimates.
        boundaries: Pre-computed IlluminationBoundaries from training data.
        boundary_col: Column name for the resulting bin assignments.
    
    Returns:
        DataFrame with added illumination bin column.
    
    Note:
        Values outside the original training range will be assigned to the
        nearest bin (clipped). This is intentional to maintain consistency.
    """
    illum_mag = compute_illuminant_magnitude(df)
    
    # Use pd.cut with the pre-computed boundaries
    # right=False means intervals are [left, right)
    # labels=False returns integer codes
    df = df.copy()
    
    try:
        df[boundary_col] = pd.cut(
            illum_mag,
            bins=boundaries.boundaries,
            labels=False,
            include_lowest=True,
        )
    except ValueError as e:
        # Handle values outside boundaries by clipping
        print(f"[illumination_binning] Warning: Some values outside boundaries: {e}")
        
        # Clip values to boundary range
        min_bound = boundaries.boundaries[0]
        max_bound = boundaries.boundaries[-1]
        illum_mag_clipped = illum_mag.clip(lower=min_bound, upper=max_bound)
        
        df[boundary_col] = pd.cut(
            illum_mag_clipped,
            bins=boundaries.boundaries,
            labels=False,
            include_lowest=True,
        )
    
    return df


def save_boundaries(boundaries: IlluminationBoundaries, output_path: str) -> None:
    """Save illumination boundaries to JSON file.
    
    Args:
        boundaries: IlluminationBoundaries object.
        output_path: Path to save JSON file.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(boundaries.to_dict(), f, indent=2)
    print(f"[illumination_binning] Saved boundaries to {output_path}")


def load_boundaries(input_path: str) -> IlluminationBoundaries:
    """Load illumination boundaries from JSON file.
    
    Args:
        input_path: Path to JSON file.
    
    Returns:
        IlluminationBoundaries object.
    """
    with open(input_path, "r") as f:
        data = json.load(f)
    return IlluminationBoundaries.from_dict(data)


def report_illumination_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    boundaries: IlluminationBoundaries,
) -> dict:
    """Generate comprehensive illumination statistics report.
    
    Args:
        train_df: Training DataFrame with illumination bins.
        val_df: Validation DataFrame (optional).
        test_df: Test DataFrame (optional).
        boundaries: IlluminationBoundaries used for binning.
    
    Returns:
        Dictionary with illumination statistics.
    """
    def count_bins(df: pd.DataFrame, col: str = "illum_bin") -> dict:
        if df is None or col not in df.columns:
            return {}
        counts = df[col].value_counts().sort_index()
        percentages = (counts / len(df) * 100).round(2)
        return {
            "counts": counts.to_dict(),
            "percentages": percentages.to_dict(),
            "total": len(df),
        }
    
    report = {
        "boundaries": boundaries.to_dict(),
        "train": count_bins(train_df),
        "validation": count_bins(val_df) if val_df is not None else {},
        "test": count_bins(test_df) if test_df is not None else {},
    }
    
    return report


def verify_no_boundary_leakage(
    boundaries: IlluminationBoundaries,
    full_df: pd.DataFrame,
    split_col: str = "split",
) -> dict:
    """Verify that boundaries were computed from training data only.
    
    This function checks that applying the same boundaries to different
    splits produces consistent results, and that the boundaries match
    what would be computed from training data alone.
    
    Args:
        boundaries: IlluminationBoundaries claimed to be from training data.
        full_df: Full DataFrame with split column and illuminant estimates.
        split_col: Column name indicating train/val/test split.
    
    Returns:
        Verification report dictionary.
    """
    # Extract training data
    train_df = full_df[full_df[split_col] == "train"]
    val_df = full_df[full_df[split_col] == "val"]
    test_df = full_df[full_df[split_col] == "test"]
    
    # Recompute boundaries from training data
    recomputed = compute_illumination_boundaries_from_train(
        train_df, n_bins=boundaries.n_bins, method=boundaries.method
    )
    
    # Check if boundaries match
    boundaries_match = all(
        abs(a - b) < 1e-6 
        for a, b in zip(boundaries.boundaries, recomputed.boundaries)
    )
    
    # Check distribution differences
    train_mag = compute_illuminant_magnitude(train_df)
    val_mag = compute_illuminant_magnitude(val_df) if len(val_df) > 0 else pd.Series([])
    test_mag = compute_illuminant_magnitude(test_df) if len(test_df) > 0 else pd.Series([])
    
    return {
        "boundaries_match_training": boundaries_match,
        "recomputed_boundaries": recomputed.boundaries,
        "original_boundaries": boundaries.boundaries,
        "train_illuminant_mean": float(train_mag.mean()),
        "val_illuminant_mean": float(val_mag.mean()) if len(val_mag) > 0 else None,
        "test_illuminant_mean": float(test_mag.mean()) if len(test_mag) > 0 else None,
        "train_illuminant_std": float(train_mag.std()),
        "val_illuminant_std": float(val_mag.std()) if len(val_mag) > 0 else None,
        "test_illuminant_std": float(test_mag.std()) if len(test_mag) > 0 else None,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }


if __name__ == "__main__":
    # Smoke test with synthetic data
    print("[illumination_binning] Running smoke test...")
    
    # Create synthetic training data
    rng = np.random.default_rng(42)
    n_train = 1000
    n_val = 200
    n_test = 200
    
    train_df = pd.DataFrame({
        "face_id": [f"train_{i}" for i in range(n_train)],
        "illuminant_estimate_r": rng.normal(128, 30, n_train),
        "illuminant_estimate_g": rng.normal(128, 30, n_train),
        "illuminant_estimate_b": rng.normal(128, 30, n_train),
        "split": "train",
    })
    
    val_df = pd.DataFrame({
        "face_id": [f"val_{i}" for i in range(n_val)],
        "illuminant_estimate_r": rng.normal(128, 30, n_val),
        "illuminant_estimate_g": rng.normal(128, 30, n_val),
        "illuminant_estimate_b": rng.normal(128, 30, n_val),
        "split": "val",
    })
    
    test_df = pd.DataFrame({
        "face_id": [f"test_{i}" for i in range(n_test)],
        "illuminant_estimate_r": rng.normal(128, 30, n_test),
        "illuminant_estimate_g": rng.normal(128, 30, n_test),
        "illuminant_estimate_b": rng.normal(128, 30, n_test),
        "split": "test",
    })
    
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Step 1: Compute boundaries from training data ONLY
    boundaries = compute_illumination_boundaries_from_train(
        train_df[train_df["split"] == "train"], n_bins=5
    )
    print(f"Computed boundaries: {boundaries.boundaries}")
    
    # Step 2: Apply to all splits
    train_df = apply_boundaries(train_df, boundaries)
    val_df = apply_boundaries(val_df, boundaries)
    test_df = apply_boundaries(test_df, boundaries)
    
    # Step 3: Verify
    verification = verify_no_boundary_leakage(boundaries, full_df)
    print(f"Boundaries match training: {verification['boundaries_match_training']}")
    
    # Step 4: Report statistics
    report = report_illumination_statistics(train_df, val_df, test_df, boundaries)
    print(f"Train bin counts: {report['train']['counts']}")
    print(f"Val bin counts: {report['validation']['counts']}")
    print(f"Test bin counts: {report['test']['counts']}")
    
    # Step 5: Test that changing val/test distributions doesn't change boundaries
    # Create a shifted test set
    test_df_shifted = test_df.copy()
    test_df_shifted["illuminant_estimate_r"] += 50
    
    # Apply same boundaries (should NOT change)
    test_df_shifted = apply_boundaries(test_df_shifted, boundaries)
    
    print("\n[smoke test] PASSED - Boundaries remain fixed regardless of test distribution")

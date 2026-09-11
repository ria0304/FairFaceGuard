"""Identity-safe train/validation/test splitting.

This module ensures that multiple frames from the same identity, source video,
or subject never appear in different splits (train/val/test), preventing data
leakage that would invalidate experimental results.

Usage:
    python -m src.data.identity_split \
        --labels_csv /path/to/labels.csv \
        --annotations_csv /path/to/annotations.csv \
        --output_csv /path/to/labels_with_splits.csv \
        --split_ratios 0.7 0.15 0.15 \
        --group_by face_id_prefix \
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def extract_identity_key(face_id: str, group_by: str = "face_id") -> str:
    """Extract an identity key from a face_id for grouping.
    
    Args:
        face_id: The face identifier from the dataset.
        group_by: Strategy for extracting identity:
            - "face_id": Use full face_id (each unique ID is its own group)
            - "face_id_prefix": Use prefix before '__' or last '_' 
            - "source_video": Expect face_id to contain video info
    
    Returns:
        Identity key string for grouping.
    """
    if group_by == "face_id":
        return face_id
    
    if group_by == "face_id_prefix":
        # Try common patterns: subject__frame, subject_frame, video_frame
        if "__" in face_id:
            return face_id.split("__")[0]
        if "_" in face_id:
            parts = face_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0]
        return face_id
    
    if group_by == "source_video":
        # Expect format like video_name/frame_number or similar
        if "/" in face_id:
            return face_id.rsplit("/", 1)[0]
        if "__" in face_id:
            return face_id.split("__")[0]
        return face_id
    
    return face_id


def check_split_overlap(
    labels_df: pd.DataFrame,
    group_col: str = "identity_key",
) -> dict:
    """Check for forbidden overlap between train/val/test splits.
    
    Args:
        labels_df: DataFrame with 'split' and identity grouping columns.
        group_col: Column name for identity grouping.
    
    Returns:
        Dictionary with overlap statistics.
    """
    if "split" not in labels_df.columns:
        return {"error": "No 'split' column found"}
    
    train_ids = set(labels_df[labels_df["split"] == "train"][group_col].unique())
    val_ids = set(labels_df[labels_df["split"] == "val"][group_col].unique())
    test_ids = set(labels_df[labels_df["split"] == "test"][group_col].unique())
    
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    
    return {
        "n_train_identities": len(train_ids),
        "n_val_identities": len(val_ids),
        "n_test_identities": len(test_ids),
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
        "train_val_overlap_ids": list(train_val_overlap),
        "train_test_overlap_ids": list(train_test_overlap),
        "val_test_overlap_ids": list(val_test_overlap),
        "clean": len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0,
    }


def stratified_group_split(
    df: pd.DataFrame,
    group_col: str,
    label_col: str | None = None,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    max_attempts: int = 100,
) -> pd.DataFrame:
    """Perform stratified group-aware splitting.
    
    Ensures:
    1. All samples from the same group go to the same split
    2. Class balance is approximately maintained across splits
    3. Split ratios are approximately respected
    
    Args:
        df: DataFrame with group and optional label columns.
        group_col: Name of the group column.
        label_col: Name of the label column for stratification (optional).
        ratios: Train/val/test ratios (must sum to 1.0).
        seed: Random seed for reproducibility.
        max_attempts: Maximum attempts to find a good split.
    
    Returns:
        DataFrame with 'split' column added.
    """
    rng = np.random.default_rng(seed)
    
    # Get unique groups
    groups = df[group_col].unique()
    n_groups = len(groups)
    
    # Calculate target split sizes (in groups)
    n_train_target = int(n_groups * ratios[0])
    n_val_target = int(n_groups * ratios[1])
    n_test_target = n_groups - n_train_target - n_val_target
    
    if label_col is not None and label_col in df.columns:
        # Stratified split: try to maintain class balance
        # Group by label and split within each label group
        label_values = df[label_col].unique()
        
        train_groups = []
        val_groups = []
        test_groups = []
        
        for label in label_values:
            label_mask = df[label_col] == label
            label_groups = df[label_mask][group_col].unique()
            rng.shuffle(label_groups)
            
            n_label = len(label_groups)
            n_train = max(1, int(n_label * ratios[0]))
            n_val = max(1, int(n_label * ratios[1]))
            
            train_groups.extend(label_groups[:n_train])
            val_groups.extend(label_groups[n_train:n_train + n_val])
            test_groups.extend(label_groups[n_train + n_val:])
    else:
        # Simple random split of groups
        shuffled_groups = rng.permutation(groups)
        train_groups = shuffled_groups[:n_train_target].tolist()
        val_groups = shuffled_groups[n_train_target:n_train_target + n_val_target].tolist()
        test_groups = shuffled_groups[n_train_target + n_val_target:].tolist()
    
    # Assign splits
    def assign_split(group):
        if group in train_groups:
            return "train"
        elif group in val_groups:
            return "val"
        else:
            return "test"
    
    df = df.copy()
    df["split"] = df[group_col].apply(assign_split)
    
    return df


def create_identity_splits(
    labels_csv: str,
    output_csv: str,
    annotations_csv: str | None = None,
    group_by: str = "face_id_prefix",
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    fail_on_overlap: bool = True,
) -> dict:
    """Create identity-safe train/val/test splits.
    
    Args:
        labels_csv: Path to labels.csv file.
        output_csv: Path to write labels with split assignments.
        annotations_csv: Optional path to annotations.csv for additional grouping.
        group_by: Strategy for extracting identity keys.
        split_ratios: Train/val/test split ratios.
        seed: Random seed.
        fail_on_overlap: If True, raise error if overlap detected.
    
    Returns:
        Dictionary with split statistics.
    """
    # Load labels
    labels_df = pd.read_csv(labels_csv)
    
    # Extract identity key
    labels_df["identity_key"] = labels_df["face_id"].apply(
        lambda x: extract_identity_key(x, group_by)
    )
    
    # Check if we have existing split column
    if "split" in labels_df.columns:
        print(f"[identity_split] Existing 'split' column found. Overwriting.")
    
    # Perform split
    label_col = "fake_label" if "fake_label" in labels_df.columns else None
    labels_df = stratified_group_split(
        labels_df,
        group_col="identity_key",
        label_col=label_col,
        ratios=split_ratios,
        seed=seed,
    )
    
    # Verify no overlap
    overlap_stats = check_split_overlap(labels_df, group_col="identity_key")
    
    if not overlap_stats["clean"]:
        error_msg = (
            f"CRITICAL: Identity overlap detected in splits!\n"
            f"  Train-Val overlap: {overlap_stats['train_val_overlap']} identities\n"
            f"  Train-Test overlap: {overlap_stats['train_test_overlap']} identities\n"
            f"  Val-Test overlap: {overlap_stats['val_test_overlap']} identities\n"
        )
        if fail_on_overlap:
            raise ValueError(error_msg)
        else:
            print(f"[identity_split] WARNING: {error_msg}")
    
    # Save output
    output_df = labels_df.drop(columns=["identity_key"])
    output_df.to_csv(output_csv, index=False)
    
    # Prepare stats
    stats = {
        "total_samples": len(labels_df),
        "total_identities": labels_df["identity_key"].nunique(),
        "split_counts": labels_df["split"].value_counts().to_dict(),
        "overlap_check": overlap_stats,
    }
    
    print(f"[identity_split] Wrote {len(output_df)} samples to {output_csv}")
    print(f"[identity_split] Split distribution: {stats['split_counts']}")
    print(f"[identity_split] Overlap check: {'PASS' if overlap_stats['clean'] else 'FAIL'}")
    
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels_csv", required=True, help="Path to labels.csv")
    p.add_argument("--output_csv", required=True, help="Path to write labels with splits")
    p.add_argument("--annotations_csv", default=None, help="Optional annotations.csv for additional metadata")
    p.add_argument("--group_by", default="face_id_prefix", choices=["face_id", "face_id_prefix", "source_video"])
    p.add_argument("--split_ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15], help="Train/val/test ratios")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--fail_on_overlap", action="store_true", default=True, help="Fail if overlap detected")
    p.add_argument("--allow_overlap", action="store_false", dest="fail_on_overlap", help="Allow overlap (not recommended)")
    return p


def main():
    args = build_arg_parser().parse_args()
    create_identity_splits(
        labels_csv=args.labels_csv,
        output_csv=args.output_csv,
        annotations_csv=args.annotations_csv,
        group_by=args.group_by,
        split_ratios=tuple(args.split_ratios),
        seed=args.seed,
        fail_on_overlap=args.fail_on_overlap,
    )


if __name__ == "__main__":
    # Smoke test with synthetic data
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp:
        # Create synthetic labels with identity groups
        rows = []
        for subject in range(10):
            for frame in range(5):
                face_id = f"subject_{subject}__frame_{frame}"
                fake_label = subject % 2  # Some subjects all real, some all fake
                rows.append({
                    "face_id": face_id,
                    "fake_label": fake_label,
                })
        
        labels_df = pd.DataFrame(rows)
        labels_path = os.path.join(tmp, "labels.csv")
        output_path = os.path.join(tmp, "labels_with_splits.csv")
        labels_df.to_csv(labels_path, index=False)
        
        stats = create_identity_splits(
            labels_csv=labels_path,
            output_csv=output_path,
            group_by="face_id_prefix",
            seed=42,
        )
        
        # Verify results
        result_df = pd.read_csv(output_path)
        assert "split" in result_df.columns, "Split column missing"
        assert stats["overlap_check"]["clean"], "Overlap should be clean"
        
        # Verify no identity appears in multiple splits
        for subject in range(10):
            subject_rows = result_df[result_df["face_id"].str.contains(f"subject_{subject}")]
            unique_splits = subject_rows["split"].unique()
            assert len(unique_splits) == 1, f"Subject {subject} appears in multiple splits: {unique_splits}"
        
        print("\n[smoke test] PASSED")

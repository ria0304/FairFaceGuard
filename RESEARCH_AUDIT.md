# FairFaceGuard Research Audit

## Executive Summary

This audit examines the FairFaceGuard repository for scientific validity, methodological rigor, reproducibility, and engineering quality. The codebase implements a research pipeline for investigating whether deepfake detector subgroup performance disparities arise from skin-tone-correlated cues, illumination confounding, or their interaction.

**Overall Assessment**: The repository is a **research prototype** with significant gaps that must be addressed before it can support publication-quality claims. Several critical issues exist around data leakage control, statistical methodology, counterfactual validation, and missing infrastructure.

---

## Issues by Severity

### CRITICAL Issues

#### 1. Missing Identity-Safe Splitting Implementation
- **Severity**: CRITICAL
- **Affected File**: `src/data/identity_split.py` (referenced in README but does not exist)
- **Explanation**: The README describes `src/data/identity_split.py` as providing "identity-safe train/val/test split" functionality, but this file does not exist in the repository. The `run_pipeline.py` directly uses `AnnotatedFaceDataset` with a `split` column that is never created. This creates severe data leakage risk where multiple frames from the same identity/video can appear in both training and test sets.
- **Recommended Fix**: Implement `src/data/identity_split.py` with proper identity/video/source-level grouping and splitting. Add automatic overlap checks that fail loudly if forbidden overlap exists.
- **Status**: ❌ NOT FIXED - File missing

#### 2. Synthetic/Fake Landmark Detection in Production Code
- **Severity**: CRITICAL
- **Affected File**: `src/annotation/ita_fitzpatrick.py` (lines 66-99)
- **Explanation**: The `detect_landmarks()` function contains a stub implementation that generates synthetic landmark coordinates rather than detecting real facial landmarks. This produces fabricated ITA annotations when mediapipe/face-alignment are not installed. The smoke test will pass but produce meaningless skin-tone labels.
- **Recommended Fix**: Either (a) make mediapipe/face-alignment a hard dependency with proper error handling, or (b) implement MediaPipe Face Mesh as the default detector. Fail explicitly if no face is detected rather than using synthetic landmarks.
- **Status**: ❌ NOT FIXED - Stub remains in place

#### 3. Illumination Binning Uses Dataset-Wide Quantiles
- **Severity**: CRITICAL
- **Affected File**: `src/data/datasets.py` (line 55)
- **Explanation**: Line 55 computes `pd.qcut(illum_mag, 5, ...)` on the entire dataset at load time. This means validation/test set illumination statistics leak into bin boundaries, and different splits get different bin definitions. This violates the principle that preprocessing parameters must be derived from training data only.
- **Recommended Fix**: Calculate illumination bin boundaries from training data only, save them, and apply identical boundaries to validation/test sets.
- **Status**: ❌ NOT FIXED - Data leakage present

#### 4. No Held-Out Test Set Protection
- **Severity**: CRITICAL
- **Affected File**: `run_pipeline.py`, `src/detector/subgroup_eval.py`
- **Explanation**: The pipeline loads train/val/test splits but nothing prevents the test set from being used for hyperparameter selection. The lambda sweep in `src/disentangle/train.py` evaluates on validation data, but there's no mechanism ensuring the test set remains truly held-out until final evaluation.
- **Recommended Fix**: Implement explicit test-set protection with separate evaluation commands. Save test predictions only after all hyperparameters are finalized using validation data.
- **Status**: ❌ NOT FIXED

#### 5. Unpaired Statistical Test for Paired Observations
- **Severity**: CRITICAL
- **Affected File**: `src/utils/metrics.py` (lines 80-99), `src/reports/gap_tables.py` (line 38)
- **Explanation**: The `permutation_test_diff()` function treats `delta_skin` and `delta_illum` as independent samples, but they are paired measurements on the SAME images. This reduces statistical power and is methodologically incorrect. A paired permutation test (sign-flip test) or Wilcoxon signed-rank test should be used.
- **Recommended Fix**: Implement paired permutation test that shuffles within-pair assignments rather than pooling all values.
- **Status**: ❌ NOT FIXED

#### 6. Counterfactual Interventions Not Validated for Orthogonality
- **Severity**: CRITICAL
- **Affected File**: `src/augmentation/counterfactual.py`
- **Explanation**: The Lab-space transformations claim orthogonality (skin-only should not change illumination, illumination-only should not change ITA), but the validation only checks approximate thresholds. The residual coupling metrics are computed but never used to reject invalid counterfactuals. Invalid counterfactuals are still written and used in analysis.
- **Recommended Fix**: Flag or exclude counterfactuals that fail manipulation checks. Report exclusion rates. Quantify residual coupling in final results.
- **Status**: ⚠️ PARTIAL - Validation exists but doesn't reject invalid samples

---

### HIGH Severity Issues

#### 7. No Configuration System
- **Severity**: HIGH
- **Affected Files**: All Python files
- **Explanation**: Scientific parameters (learning rates, batch sizes, lambda values, augmentation parameters, bin boundaries, etc.) are hard-coded throughout the codebase. There is no YAML/JSON configuration system, making experiments irreproducible and parameter sweeps ad-hoc.
- **Recommended Fix**: Create a configuration system (YAML-based) with experiment IDs. Every experiment should save its exact configuration, git commit hash, and environment information.
- **Status**: ❌ NOT FIXED

#### 8. No Multi-Seed Experiments
- **Severity**: HIGH
- **Affected File**: `src/utils/seed.py`, `run_pipeline.py`
- **Explanation**: Only seed 42 is used. Single-seed results are not scientifically defensible. The pipeline should support multiple seeds (e.g., 42, 123, 2024, 3407, 7777) and report mean ± std with confidence intervals.
- **Recommended Fix**: Add multi-seed experiment runner that aggregates results across seeds with proper statistical summaries.
- **Status**: ❌ NOT FIXED

#### 9. Bootstrap Confidence Intervals Ignore Clustering
- **Severity**: HIGH
- **Affected File**: `src/utils/metrics.py` (lines 59-77)
- **Explanation**: While the docstring mentions aggregating by identity first, the `bootstrap_ci()` function itself has no built-in cluster-aware resampling. Users could easily pass per-frame values. Additionally, the aggregation in `counterfactual_eval.py` uses simple means without considering variance.
- **Recommended Fix**: Implement cluster-bootstrap that resamples at the identity/video level, not the sample level. Add warnings if non-aggregated data is passed.
- **Status**: ⚠️ PARTIAL - Docstring warns but implementation doesn't enforce

#### 10. Missing Baseline Metrics
- **Severity**: HIGH
- **Affected File**: `src/detector/subgroup_eval.py`, `src/utils/metrics.py`
- **Explanation**: Subgroup evaluation reports accuracy, AUC, and EER, but missing precision, recall, F1, FPR, FNR separately. Fairness analysis requires FPR gap and FNR gap specifically (not just accuracy gap).
- **Recommended Fix**: Add complete metric suite including FPR, FNR, precision, recall, F1 per subgroup with confidence intervals.
- **Status**: ❌ NOT FIXED

#### 11. Gradient Reversal Layer Not Unit Tested
- **Severity**: HIGH
- **Affected File**: `src/disentangle/grl.py`
- **Explanation**: The GRL is central to the adversarial disentanglement claims but has no unit test verifying that gradients are actually reversed. A silent bug here would invalidate all disentanglement results.
- **Recommended Fix**: Add unit test that verifies gradient direction numerically using finite differences or torch.autograd.gradcheck.
- **Status**: ❌ NOT FIXED

#### 12. No Experiment Artifact Tracking
- **Severity**: HIGH
- **Affected Files**: All
- **Explanation**: There is no systematic saving of checkpoints, predictions, metrics, or intermediate artifacts. Results cannot be traced back to specific runs. No experiment IDs exist.
- **Recommended Fix**: Implement artifact directory structure with configs, checkpoints, predictions, metrics, and logs saved per experiment ID.
- **Status**: ❌ NOT FIXED

#### 13. Fitzpatrick Mapping Claims Without Validation
- **Severity**: HIGH
- **Affected File**: `src/annotation/ita_fitzpatrick.py`
- **Explanation**: The ITA→Fitzpatrick mapping uses published thresholds but the code has no validation against human-rated Fitzpatrick labels. The README acknowledges this limitation but the code proceeds as if bins are ground truth.
- **Recommended Fix**: Add explicit labeling of outputs as "ITA-derived skin-tone categories" rather than claiming ground-truth Fitzpatrick types. Add optional validation module for human-rated subsets.
- **Status**: ⚠️ PARTIAL - Limitation documented but code doesn't reflect uncertainty

#### 14. No Negative/Sanity Controls
- **Severity**: HIGH
- **Affected Files**: None exist
- **Explanation**: Missing critical controls:
  - Label shuffling (should show no meaningful disentanglement from random labels)
  - Lambda-zero control (should behave like standard baseline)
  - Non-face intervention control
  - Dataset correlation check (are fake/real labels correlated with skin tone?)
- **Recommended Fix**: Implement sanity control experiments as separate modules.
- **Status**: ❌ NOT FIXED

---

### MEDIUM Severity Issues

#### 15. Incomplete Counterfactual Generator Architecture
- **Severity**: MEDIUM
- **Affected File**: `src/augmentation/counterfactual.py`
- **Explanation**: Only classical Lab-space generator implemented. No interface for learned counterfactual generators (GAN/diffusion). The documentation mentions this as optional but provides no stub interface.
- **Recommended Fix**: Add abstract `CounterfactualGenerator` base class with `ClassicalLabGenerator` implementation and stub for `LearnedCounterfactualGenerator`.
- **Status**: ❌ NOT FIXED

#### 16. Probe Performance Misinterpretation Risk
- **Severity**: MEDIUM
- **Affected File**: `src/disentangle/probing.py`
- **Explanation**: The docstring correctly notes that high probe accuracy indicates information availability but not causal use. However, the reporting in `gap_tables.py` doesn't emphasize this distinction strongly enough.
- **Recommended Fix**: Add explicit caveats in report output distinguishing representation availability from causal reliance.
- **Status**: ⚠️ PARTIAL - Documented but could be clearer

#### 17. No Multiple Comparison Correction
- **Severity**: MEDIUM
- **Affected File**: `src/reports/gap_tables.py`
- **Explanation**: When testing multiple hypotheses (skin effect, illumination effect, interaction), no correction for multiple comparisons is applied. This inflates Type I error rate.
- **Recommended Fix**: Apply Bonferroni or Benjamini-Hochberg correction when reporting multiple p-values.
- **Status**: ❌ NOT FIXED

#### 18. Missing Ablation Study Infrastructure
- **Severity**: MEDIUM
- **Affected Files**: None exist
- **Explanation**: No automated ablation study framework. Key ablations needed:
  - No illumination normalization
  - Skin-only adversarial
  - Illumination-only adversarial
  - Joint adversarial
  - Different lambda configurations
- **Recommended Fix**: Create ablation study runner with standardized comparison tables.
- **Status**: ❌ NOT FIXED

#### 19. No External Validation Dataset
- **Severity**: MEDIUM
- **Affected File**: `src/data/aiface_adapter.py`
- **Explanation**: Pipeline is built around AI-Face dataset only. No mechanism for external validation on an independent dataset (e.g., FF++, Celeb-DF, DFDC subset).
- **Recommended Fix**: Add adapter for at least one external deepfake dataset for generalization testing.
- **Status**: ❌ NOT FIXED

#### 20. Dataset Bias Not Audited
- **Severity**: MEDIUM
- **Affected Files**: None exist
- **Explanation**: No analysis of whether:
  - Fake samples have systematically different illumination from real
  - Skin tone correlates with fake/real labels due to dataset construction
  - Compression artifacts correlate with demographic groups
- **Recommended Fix**: Create dataset bias audit module that quantifies these relationships.
- **Status**: ❌ NOT FIXED

#### 21. Incomplete Causal Language Documentation
- **Severity**: MEDIUM
- **Affected File**: `src/disentangle/counterfactual_eval.py`, `src/reports/gap_tables.py`
- **Explanation**: While some caution is shown, the headline conclusion string uses causal language ("causal driver") that may overclaim given the synthetic nature of interventions.
- **Recommended Fix**: Use more precise language like "counterfactual sensitivity under specified image intervention" and document assumptions explicitly.
- **Status**: ⚠️ PARTIAL - Some caution but could be more precise

#### 22. No Figure Generation Module
- **Severity**: MEDIUM
- **Affected Files**: None exist
- **Explanation**: No automated figure generation. Publication requires:
  - Methodology overview
  - Dataset distributions
  - Subgroup performance charts
  - Counterfactual validation plots
  - Delta distributions
  - Accuracy-vs-invariance frontiers
- **Recommended Fix**: Create `src/figures/` module with matplotlib-based figure generation from saved experiment outputs.
- **Status**: ❌ NOT FIXED

#### 23. No Automated Table Generation
- **Severity**: MEDIUM
- **Affected Files**: `src/reports/gap_tables.py`
- **Explanation**: Only JSON output. Need LaTeX and CSV export for publication tables.
- **Recommended Fix**: Add table export functions supporting JSON, CSV, and LaTeX formats.
- **Status**: ❌ NOT FIXED

---

### LOW Severity Issues

#### 24. Mediapipe Version Pinning Outdated
- **Severity**: LOW
- **Affected File**: README.md (line 129), requirements.txt comments
- **Explanation**: Recommends mediapipe==0.10.13 pinned version. Newer versions may work with updated model download paths.
- **Recommended Fix**: Test with current mediapipe version and update pinning.
- **Status**: ⚠️ MINOR

#### 25. Limited Backbone Options Tested
- **Severity**: LOW
- **Affected File**: `src/detector/baseline_model.py`
- **Explanation**: EfficientNet-B4 is default; Xception/ViT/CLIP mentioned but not tested. Backbone choice could affect fairness conclusions.
- **Recommended Fix**: At minimum, test EfficientNet-B0 and B4 to check backbone sensitivity.
- **Status**: ⚠️ MINOR

#### 26. No Early Stopping in Training
- **Severity**: LOW
- **Affected File**: `src/detector/train_baseline.py`, `src/disentangle/train.py`
- **Explanation**: Fixed epoch counts without early stopping based on validation performance.
- **Recommended Fix**: Add early stopping with patience parameter.
- **Status**: ❌ NOT FIXED

#### 27. Cosine Annealing Scheduler Hard-Coded
- **Severity**: LOW
- **Affected File**: `src/detector/train_baseline.py` (line 28)
- **Explanation**: Learning rate scheduler choice is hard-coded.
- **Recommended Fix**: Make scheduler configurable.
- **Status**: ⚠️ MINOR

#### 28. No GPU Memory Management
- **Severity**: LOW
- **Affected File**: Various
- **Explanation**: No explicit CUDA memory cleanup between stages. Could cause OOM on long runs.
- **Recommended Fix**: Add `torch.cuda.empty_cache()` calls between major stages.
- **Status**: ⚠️ MINOR

#### 29. Missing `__init__.py` Exports
- **Severity**: LOW
- **Affected Files**: Multiple `__init__.py` files are empty
- **Explanation**: Package `__init__.py` files don't export public APIs, making imports verbose.
- **Recommended Fix**: Add appropriate exports to `__init__.py` files.
- **Status**: ⚠️ MINOR

#### 30. No Logging Framework
- **Severity**: LOW
- **Affected Files**: All
- **Explanation**: Uses print statements instead of structured logging. Makes debugging and run analysis difficult.
- **Recommended Fix**: Add Python logging with configurable levels and file output.
- **Status**: ❌ NOT FIXED

---

## Summary Statistics

| Severity | Count | Fixed | Partial | Not Fixed |
|----------|-------|-------|---------|-----------|
| CRITICAL | 6 | 0 | 1 | 5 |
| HIGH | 8 | 0 | 0 | 8 |
| MEDIUM | 9 | 0 | 2 | 7 |
| LOW | 7 | 0 | 4 | 3 |
| **TOTAL** | **30** | **0** | **7** | **23** |

---

## Priority Action Items

1. **Implement identity-safe splitting** (`src/data/identity_split.py`) - blocks all valid experimentation
2. **Replace synthetic landmarks with real detector** - blocks valid skin-tone annotation
3. **Fix illumination binning data leakage** - blocks valid subgroup analysis
4. **Implement paired statistical tests** - blocks valid causal claims
5. **Add configuration system** - blocks reproducibility
6. **Add multi-seed experiments** - blocks statistical reliability claims
7. **Implement negative controls** - blocks scientific validity
8. **Create figure/table generation** - blocks publication readiness

---

## Files That Don't Exist But Are Referenced

1. `src/data/identity_split.py` - Critical for leakage prevention
2. `configs/` directory - Needed for experiment configuration
3. `tests/` directory - Needed for unit/integration tests
4. `src/figures/` - Needed for automated figure generation
5. `src/tables/` - Needed for automated table export
6. `scripts/` - Needed for experiment runners

---

## Conclusion

The FairFaceGuard repository contains a well-conceived research methodology but is currently a **research prototype** requiring substantial work before supporting publication-quality claims. The most critical gaps are:

1. **Data leakage risks** from missing identity-safe splitting
2. **Fabricated annotations** from synthetic landmark fallback
3. **Incorrect statistical methodology** for paired observations
4. **Missing reproducibility infrastructure** (configs, seeds, artifacts)
5. **No negative controls** to validate the methodology

Addressing these issues is essential before any experimental results can be considered scientifically defensible.

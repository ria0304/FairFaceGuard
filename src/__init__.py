"""Deepfake detection under skin-tone and lighting variations.

Package layout mirrors the 5-week research roadmap:
    annotation/    Week 1   - ITA / Fitzpatrick skin-tone annotation
    augmentation/  Week 2   - counterfactual (skin-tone-only / illumination-only) generation
    detector/      Week 3   - baseline detector + subgroup accuracy/EER gap tables
    disentangle/   Week 4   - adversarial disentanglement, probing, counterfactual causal testing
    reports/       Week 5   - final gap tables, forest plots, write-up aggregation
    utils/                  - shared metrics/helpers used across weeks
"""

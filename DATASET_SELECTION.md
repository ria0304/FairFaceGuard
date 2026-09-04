# Dataset Selection: AI-Face &rarr; FaceForensics++ / Celeb-DF

`TRANSFORMATION_PROGRESS.md` (Section 3) flagged this decision as required
before running main experiments. This document records it.

## Decision

**Primary dataset:** FaceForensics++ (FF++), all four manipulation methods
(Deepfakes, Face2Face, FaceSwap, NeuralTextures), c23 compression.
**Cross-dataset validation:** Celeb-DF (v2), evaluation-only, never trained
on.
**Dropped:** AI-Face (was the previous primary, per the locked Week-1
decision).

## Why AI-Face was dropped

AI-Face is a still-image dataset of AI-*generated* faces (GAN/diffusion
samples vs. real face datasets like FFHQ), not manipulated video of real
people. For a paper about deepfake *detection*, this is a construct-validity
problem: a detector's behavior on synthetic-from-noise images does not
necessarily transfer to its behavior on face-swapped/reenacted video of real
identities, which is the actual deployed threat model deepfake detectors are
built for. Two more specific issues compounded this:

- **No genuine skin-tone label.** AI-Face v1 offers a Monk Skin Tone scale
  (a real, if coarse, skin-tone signal); the current default download (v2)
  replaced it with a categorical "Race" field, which `aiface_adapter.py`
  could only map to Fitzpatrick via a documented weak proxy
  (`RACE_TO_FITZPATRICK_PROXY`) that conflates race with skin tone -- exactly
  the kind of proxy this study should be avoiding, not relying on as a
  fallback label source.
- **No identity/source structure.** AI-Face images are independent
  generated samples with no video or source-actor grouping, so there is no
  natural "same person, multiple frames" leakage risk to control for, and
  no way to hold out a whole identity's manipulations in the paper's second
  claim (generalization across manipulation methods for the same source
  face).

## Why FaceForensics++ (primary)

- Most-cited benchmark in the deepfake-detection literature, so results are
  directly comparable to published baselines.
- Real YouTube source videos with four distinct, well-documented
  manipulation methods -- covers both face-swap and expression-reenactment
  attack families in one dataset, which AI-Face has no equivalent of.
- Genuine source-identity metadata (the `video_id` before the manipulation
  suffix), which makes an actual identity-safe split meaningful and
  testable, rather than a formality.
- No skin-tone or race label of any kind -- see "Skin-tone labeling"
  below; this ends up an advantage, not a gap.

## Why Celeb-DF (cross-dataset validation only)

Celeb-DF uses a different, higher-quality GAN-based face-swap pipeline than
any FF++ manipulation, and draws from a separate pool of celebrity source
identities. Training once on FF++ and evaluating zero-shot on Celeb-DF
(`src/experiments/cross_dataset_validator.py`) tests whether the
skin-tone/illumination subgroup gap is a property of deepfake detectors
generally, or an artifact specific to FF++'s manipulation methods. This is
why Celeb-DF is not folded into the training pool: mixing it in would
remove the one experiment it is uniquely suited to run.

## Skin-tone labeling under the new datasets

Neither FF++ nor Celeb-DF ships any demographic or skin-tone label. Unlike
under AI-Face, there is no ground-truth or proxy label to fall back on or
compare against -- the auto-ITA/Fitzpatrick bin from Week 1's
`annotate_face()` (`src/annotation/ita_fitzpatrick.py`) is the *only* skin-
tone signal for every row (`fitzpatrick_bin_source` is always `"auto_ita"`
in `ffplus_adapter.py` / `celebdf_adapter.py`, versus AI-Face's three
possible sources). This is treated as a net improvement for this study's
specific causal question: it removes the temptation to substitute a
race-based proxy for skin tone anywhere in the pipeline, at the cost of
having no independent label to sanity-check the ITA pipeline's Fitzpatrick
bins against. **Mitigation:** treat `patch_confidence`/`flagged` from
`annotate_face()` as the primary per-sample quality control instead, and
consider a small manually-labeled audit sample if reviewer pushback on this
point is expected.

## Other candidates considered, not selected

| Dataset | Role considered | Why not selected (for this study, now) |
|---|---|---|
| DFDC (Preview or full) | Stress test / demographic diversity | Full DFDC is very large (~470GB) and its own demographic-distribution claims are themselves contested in the fairness literature -- adds scope without a clear benefit over FF+++Celeb-DF for the causal skin/illumination question. Left as a candidate for a future robustness-focused follow-up, not this round. |
| DeeperForensics | Additional manipulation diversity | Real value (perturbation-robustness variants), but would be a third dataset to annotate/maintain for limited marginal benefit over the FF++ + Celeb-DF pair, which already covers two manipulation families and two identity pools. |
| UADFV | Early baseline | Too small and too easy (near-ceiling accuracy in prior work) to produce a meaningful subgroup gap to analyze. |

## Access notes (for whoever runs this next)

Both FF++ and Celeb-DF are gated releases, not pip-installable or
directly downloadable:

- FF++: request access at https://github.com/ondyari/FaceForensics
- Celeb-DF (v2): request access at https://github.com/yuezunli/celeb-deepfakeforensics

Budget lead time for the access request before planning a run date. Neither
adapter (`ffplus_adapter.py`, `celebdf_adapter.py`) can fabricate the raw
video data -- run `--sample_n 50` or similar on a small slice first once
access is granted, to verify the extraction + ITA-annotation path before
committing to a full run.

## What this changes downstream

- `configs/baseline.yaml`'s `dataset.type` default changed from `"aiface"`
  to `"ffpp"`.
- `src/data/aiface_adapter.py` is left in place (not deleted) -- AI-Face
  remains usable as a supplementary/robustness dataset if a future revision
  wants it, it's simply no longer the primary.
- Everything downstream of `labels.csv` + `annotations.csv`
  (`src/data/datasets.py` onward: augmentation, baseline training,
  disentanglement, reporting) is unchanged, by design -- the adapter
  contract was built specifically so a dataset swap wouldn't ripple past
  this layer.

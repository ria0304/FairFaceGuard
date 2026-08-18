# Deepfake Detection Under Skin-Tone and Lighting Variations

**Core research question:** when a deepfake detector's accuracy differs across
skin-tone groups, is it actually responding to skin tone, or to
illumination/color cues that merely correlate with skin tone? This codebase
implements the full 5-week pipeline that answers that question, matching the
roadmap exactly.

## What's in here

```
skin_tone_deepfake/
├── run_pipeline.py               orchestrator -- runs all 5 weeks end to end
├── requirements.txt
└── src/
    ├── annotation/                WEEK 1
    │   └── ita_fitzpatrick.py     face -> ITA score -> Fitzpatrick I-VI bin,
    │                              with an illuminant-normalization step so the
    │                              skin-tone label doesn't silently absorb lighting
    ├── augmentation/               WEEK 2
    │   └── counterfactual.py      generates {original, skin_only, illum_only,
    │                              both} images per face + a manipulation-check
    │                              validation loop (residual coupling report)
    ├── detector/                   WEEK 3
    │   ├── baseline_model.py      plain backbone + real/fake head (no
    │                              disentanglement) -- "the detector as built"
    │   ├── train_baseline.py      standard training loop
    │   └── subgroup_eval.py       accuracy / AUC / EER gap tables by
    │                              Fitzpatrick bin, illumination bin, and both
    ├── disentangle/                WEEK 4
    │   ├── grl.py                 Gradient Reversal Layer (Ganin & Lempitsky)
    │   ├── model.py               shared backbone + real/fake head + two
    │                              adversarial probes (skin, illumination)
    │   ├── losses.py               L = L_cls − λ1·L_adv(skin) − λ2·L_adv(illum)
    │   ├── train.py                sweeps λ_skin and λ_illum INDEPENDENTLY to
    │                              trace two separate accuracy-vs-invariance
    │                              frontiers
    │   ├── probing.py             linear probes on the FROZEN baseline --
    │                              "is this info available in the features?"
    │   └── counterfactual_eval.py  the headline result: runs the frozen
    │                              baseline on Week 2's factorial sets and
    │                              computes Delta_skin vs Delta_illum, the
    │                              direct causal comparison
    ├── reports/                    WEEK 5
    │   └── gap_tables.py           aggregates everything into one final
    │                              report with bootstrap CIs + a permutation
    │                              test comparing Delta_skin vs Delta_illum
    ├── data/
    │   └── datasets.py             Dataset classes + expected on-disk layout
    └── utils/
        ├── metrics.py               EER, subgroup gap tables, bootstrap CI,
        │                            permutation test
        └── seed.py                  reproducibility

```

## What it actually does, stage by stage

1. **Annotate (Week 1).** For every face crop, estimate the scene illuminant,
   white-balance-correct the image, sample cheek/forehead/nose patches,
   convert to Lab, and compute the Individual Typology Angle (ITA), which is
   binned into Fitzpatrick I–VI. The white-balance step happens *before* ITA
   is computed specifically so the "ground truth" skin-tone label isn't
   itself contaminated by lighting — that would defeat the whole point of
   the study.

2. **Generate counterfactuals (Week 2).** For each annotated face, produce
   four versions: the original, a version with pigmentation shifted along
   the ITA axis but lighting untouched, a version with lighting/white-balance
   shifted but pigmentation untouched, and a version with both shifted. Each
   generated image is re-run through the Week 1 ITA pipeline as a
   manipulation check — if a "skin-tone-only" shift also moved the estimated
   illuminant by more than a small tolerance, it's flagged so you can report
   the residual coupling honestly in the paper.

3. **Train baseline + measure the gap (Week 3).** Train a standard detector
   (EfficientNet-B4 by default, swappable to Xception/ViT/CLIP backbones) on
   your normal corpus. Evaluate it on the annotated test set and produce
   accuracy/AUC/EER broken out by skin-tone bin, by illumination bin, and as
   a 2D cross table. This step only tells you **that** a gap exists — not why.

4. **Disentangle (Week 4).** This is where the causal question gets answered,
   three ways:
   - *Adversarial invariance*: attach two Gradient-Reversal-Layer probes to
     the shared representation, one predicting skin-tone bin, one predicting
     illumination bin, and sweep each one's strength (λ) independently. If
     making the representation invariant to skin tone costs real/fake
     accuracy while illumination-invariance is nearly free, that's evidence
     skin tone itself carried useful (but confounded) signal.
   - *Frozen-model probing*: without touching the baseline at all, train
     linear probes on its features to see whether skin-tone/illumination
     information is even present (necessary, not sufficient, evidence of use).
   - *Counterfactual causal testing* — **the headline result**: run the
     untouched, frozen baseline on the Week 2 factorial sets and measure how
     much its output probability moves under a skin-tone-only vs.
     illumination-only perturbation, holding identity and artifacts fixed.
     This directly measures what the *deployed* model reacts to.

5. **Finalize (Week 5).** Pulls all of the above into one JSON report:
   descriptive gap tables, the two accuracy-vs-invariance frontiers, probe
   leakage numbers, and the counterfactual Delta_skin vs Delta_illum
   comparison with bootstrap 95% CIs and a permutation test for whether the
   two effect sizes are significantly different from each other (not just
   from zero). Ends with a plain-language headline conclusion string you can
   quote directly in the paper's abstract/results.

## Running it

```bash
pip install -r requirements.txt

# expects data_root/frames/<face_id>.png and data_root/labels.csv
# (columns: face_id, fake_label) to exist already
python run_pipeline.py --data_root /path/to/data --stage all
```

Or run any single stage:
```bash
python run_pipeline.py --data_root /path/to/data --stage annotate
python run_pipeline.py --data_root /path/to/data --stage augment
python run_pipeline.py --data_root /path/to/data --stage baseline --epochs 20
python run_pipeline.py --data_root /path/to/data --stage disentangle
python run_pipeline.py --data_root /path/to/data --stage report
```

Each `src/*/*.py` module also has a small `__main__` smoke test so you can
sanity-check a stage on synthetic data without a real dataset.

## Known limitations (state these explicitly in the paper)

- The classical Lab-space counterfactual generator (`shift_skin_tone`,
  `shift_illumination`) is a fast, interpretable approximation — real light
  transport couples pigment and illumination, so no such shift is perfectly
  orthogonal. The validation loop's `residual_illuminant_coupling_in_skin_shift`
  / `residual_ita_coupling_in_illum_shift` numbers should be reported as a
  bounded error source. A GAN/diffusion-based backend can be swapped in
  (see the docstring in `counterfactual.py`) for higher fidelity at the cost
  of harder-to-audit leakage.
- Adversarial-invariance training (Week 4, `train.py`) changes the model
  being studied. Its frontier tells you how much accuracy invariance costs,
  not a direct causal readout of the original baseline — that's what the
  counterfactual test (`counterfactual_eval.py`, run on the frozen baseline)
  is for, and it should be the foregrounded result.
- `detect_landmarks()` in `ita_fitzpatrick.py` is a placeholder stub so the
  module runs standalone. Swap in a real 68/98-point face-alignment model
  before running on real data — see the docstring for the drop-in.

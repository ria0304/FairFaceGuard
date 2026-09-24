# Saved-Results Validation Report (2026-09-20)

Internal-consistency audit of `output/results/` — no retraining, checks only.

## PASS
1. **Frontier completeness**: `week4_results_summary_solo.json` has all 10 configs
   (5 skin lambdas + 5 illum lambdas, 15 epochs each per `all_epochs_report.md`).
2. **Shared control**: λ=0.0/0.0 row byte-identical in both frontiers (trained once, reused) — matches the "9 trained runs" note.
3. **Metric sanity**: all `fake_acc` / `skin_probe_acc` / `illum_probe_acc` in [0,1].
4. **final_report == week4**: `week4_accuracy_vs_invariance_frontiers` exactly equals `sweep_results`.
5. **Causal numbers propagate exactly**: `week4_causal_counterfactual_effect`
   (|Δskin|=0.00094, |Δillum|=0.00101, paired p=0.8818, d=-0.003) matches
   `statistical_validation_report.json` point estimates and CIs to full precision.
6. **Correct test used**: paired sign-flip permutation (appropriate for same-sample Δskin vs Δillum), Bonferroni correction present, headline conclusion honestly reports null result.
7. **Baseline learning curve sane**: train_loss 0.22→0.0001, val_acc →1.0 over 20 epochs (near-ceiling — expected on small Deepfakes-only split; reinforces need for harder data scope).

## FLAGS (non-blocking, fix before submission)
1. **Bonferroni mismatch**: `final_report.bonferroni_corrected_p = 0.8818` but
   `statistical_validation_report.delta_skin_vs_delta_illum.corrected_p_value = 1.0`.
   Same raw p (0.8818), different correction denominator. Pick one denominator
   (number of pre-specified comparisons) and make both files agree. Conclusion
   unaffected (both non-significant).
2. **`paper_quality_gate.json: epochs_logged = 0`** despite the 9×15-epoch claim —
   provenance gap. Rebuild the gate from `all_epochs_report.md` / checkpoints so the
   count is auditable.
3. **Near-ceiling baseline** (val_acc 1.0, gate AUC 0.994/EER 0.01): effects are
   estimated on a saturated model over a tiny fake sample (test n=10 fakes) —
   honest in text already, but multi-seed + more fakes are the real fix.

## Verdict
Saved sweep results are **internally consistent and trustworthy as a pilot**.
Proceed to multi-seed replication; do not re-run the single-seed sweep.

"""Automated ablation study framework for FairFaceGuard.

This framework systematically evaluates the contribution of each component
to the overall research findings.

Supported ablations:
    A. Standard baseline (no adversarial training)
    B. Skin adversarial only
    C. Illumination adversarial only  
    D. Skin + illumination adversarial (full model)
    E. Lambda sweep (multiple values)
    F. No illumination normalization
    G. With illumination normalization
    H. No counterfactual intervention
    I. Classical Lab counterfactual generator
    J. Alternative backbone (if computationally feasible)

Each ablation uses:
    - Same dataset split
    - Same evaluation protocol
    - Same held-out test set
    - Appropriate seed handling
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Optional
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str
    lambda_skin: float = 0.0
    lambda_illum: float = 0.0
    use_illum_normalization: bool = True
    use_counterfactual: bool = True
    counterfactual_type: str = "classical_lab"
    backbone: str = "efficientnet_b4"
    seed: int = 42
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class AblationResult:
    """Results from a single ablation experiment."""
    config: AblationConfig
    metrics: dict
    subgroup_metrics: dict
    probe_metrics: dict
    fairness_gaps: dict
    n_samples: int
    completed: bool = True
    error_message: str = ""
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["config"] = self.config.to_dict()
        return d


class AblationFramework:
    """Framework for running systematic ablation studies."""
    
    def __init__(
        self,
        output_dir: str,
        base_config: Optional[dict] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_config = base_config or {}
        self.results = {}
        
    def define_standard_ablations(self) -> list[AblationConfig]:
        """Define the standard set of ablation experiments."""
        ablations = [
            AblationConfig(
                name="baseline",
                description="Standard deepfake detector without adversarial disentanglement",
                lambda_skin=0.0,
                lambda_illum=0.0,
                use_illum_normalization=True,
                use_counterfactual=False,
            ),
            AblationConfig(
                name="skin_adv_only",
                description="Adversarial disentanglement for skin tone only",
                lambda_skin=1.0,
                lambda_illum=0.0,
                use_illum_normalization=True,
                use_counterfactual=False,
            ),
            AblationConfig(
                name="illum_adv_only",
                description="Adversarial disentanglement for illumination only",
                lambda_skin=0.0,
                lambda_illum=1.0,
                use_illum_normalization=True,
                use_counterfactual=False,
            ),
            AblationConfig(
                name="full_adversarial",
                description="Joint adversarial disentanglement for both factors",
                lambda_skin=1.0,
                lambda_illum=1.0,
                use_illum_normalization=True,
                use_counterfactual=False,
            ),
            AblationConfig(
                name="high_lambda",
                description="Stronger adversarial pressure (lambda=2.0)",
                lambda_skin=2.0,
                lambda_illum=2.0,
                use_illum_normalization=True,
                use_counterfactual=False,
            ),
            AblationConfig(
                name="no_illum_norm",
                description="Without illumination normalization preprocessing",
                lambda_skin=1.0,
                lambda_illum=1.0,
                use_illum_normalization=False,
                use_counterfactual=False,
            ),
        ]
        return ablations
    
    def run_ablation(
        self,
        config: AblationConfig,
        train_fn: Callable,
        eval_fn: Callable,
        model_factory: Callable,
    ) -> AblationResult:
        """Run a single ablation experiment."""
        print(f"\nRunning ablation: {config.name}")
        print(f"  Description: {config.description}")
        print(f"  lambda_skin={config.lambda_skin}, lambda_illum={config.lambda_illum}")
        
        try:
            model = model_factory(config)
            train_results = train_fn(model=model, config=config)
            
            if not train_results.get("completed", False):
                return AblationResult(
                    config=config, metrics={}, subgroup_metrics={},
                    probe_metrics={}, fairness_gaps={}, n_samples=0,
                    completed=False, error_message=train_results.get("error", "Training failed"),
                )
            
            eval_results = eval_fn(model)
            
            result = AblationResult(
                config=config,
                metrics=eval_results.get("main_metrics", {}),
                subgroup_metrics=eval_results.get("subgroup_metrics", {}),
                probe_metrics=eval_results.get("probe_metrics", {}),
                fairness_gaps=eval_results.get("fairness_gaps", {}),
                n_samples=eval_results.get("n_test_samples", 0),
                completed=True,
            )
            
            print(f"  Accuracy: {result.metrics.get('accuracy', np.nan):.4f}")
            print(f"  AUC: {result.metrics.get('auc', np.nan):.4f}")
            
            return result
            
        except Exception as e:
            return AblationResult(
                config=config, metrics={}, subgroup_metrics={},
                probe_metrics={}, fairness_gaps={}, n_samples=0,
                completed=False, error_message=str(e),
            )
    
    def run_all_ablations(
        self,
        ablations: list[AblationConfig],
        train_fn: Callable,
        eval_fn: Callable,
        model_factory: Callable,
    ) -> dict:
        """Run all ablation experiments."""
        print("=" * 70)
        print("ABLATION STUDY")
        print("=" * 70)
        
        results = {}
        summary_rows = []
        
        for config in ablations:
            result = self.run_ablation(config, train_fn, eval_fn, model_factory)
            results[config.name] = result
            
            summary_rows.append({
                "Ablation": config.name,
                "Description": config.description,
                "λ_skin": config.lambda_skin,
                "λ_illum": config.lambda_illum,
                "Accuracy": result.metrics.get("accuracy", np.nan),
                "AUC": result.metrics.get("auc", np.nan),
                "F1": result.metrics.get("f1", np.nan),
                "Skin Probe Acc": result.probe_metrics.get("skin_accuracy", np.nan),
                "Illum Probe Acc": result.probe_metrics.get("illum_accuracy", np.nan),
                "Max-Min Gap": result.fairness_gaps.get("max_min_gap", np.nan),
                "Worst Group": result.fairness_gaps.get("worst_group_acc", np.nan),
                "Completed": result.completed,
            })
        
        summary_df = pd.DataFrame(summary_rows)
        self._save_results(results, summary_df)
        
        print("\n" + "=" * 70)
        print("ABLATION SUMMARY TABLE")
        print("=" * 70)
        print(summary_df.to_string(index=False))
        
        return {
            "results": {k: v.to_dict() for k, v in results.items()},
            "summary_df": summary_df,
        }
    
    def _save_results(self, results: dict, summary_df: pd.DataFrame) -> None:
        """Save ablation results to files."""
        results_path = self.output_dir / "ablation_results.json"
        with open(results_path, "w") as f:
            json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2, default=str)
        
        summary_csv_path = self.output_dir / "ablation_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        
        summary_tex_path = self.output_dir / "ablation_summary.tex"
        tex_content = summary_df.to_latex(index=False, caption="Ablation Study Results")
        with open(summary_tex_path, "w") as f:
            f.write(tex_content)
        
        print(f"\nSaved ablation results to {self.output_dir}")


def run_lambda_sweep(
    lambda_values: list[float],
    train_fn: Callable,
    eval_fn: Callable,
    model_factory: Callable,
    output_dir: str,
    fixed_illum_lambda: float = 1.0,
    seed: int = 42,
) -> dict:
    """Run a lambda sweep to find optimal trade-off point."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for lambda_skin in lambda_values:
        config = AblationConfig(
            name=f"lambda_sweep_{lambda_skin}",
            description=f"Lambda skin sweep at {lambda_skin}",
            lambda_skin=lambda_skin,
            lambda_illum=fixed_illum_lambda,
        )
        
        framework = AblationFramework(str(output_path))
        result = framework.run_ablation(config, train_fn, eval_fn, model_factory)
        
        results.append({
            "lambda_skin": lambda_skin,
            "accuracy": result.metrics.get("accuracy", np.nan),
            "auc": result.metrics.get("auc", np.nan),
            "skin_probe_acc": result.probe_metrics.get("skin_accuracy", np.nan),
            "illum_probe_acc": result.probe_metrics.get("illum_accuracy", np.nan),
            "fairness_gap": result.fairness_gaps.get("max_min_gap", np.nan),
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path / "lambda_sweep.csv", index=False)
    df.to_json(output_path / "lambda_sweep.json", orient="records", indent=2)
    
    print("\nLambda Sweep Results:")
    print(df.to_string(index=False))
    
    return {"results_df": df}


if __name__ == "__main__":
    print("Ablation framework module loaded successfully.")
    print("Use AblationFramework class to run ablation studies.")

"""OpenBinder training scaffold.

Modules:
    datamodule       — feature loading, intersection cohort, hold-out, PCA
    feature_combine  — stateless helpers for rest/unrest/both combination
    rf_train         — Random Forest trainer (full non-LOO fit)
    mlp_train        — MLP trainer (PyTorch, early stopping)
    loo_harness      — Leave-One-Out harness with sidecar-aware trio construction
    ood_eval         — Score the 30 held-out orphans with a trained model

Config files live under ``configs/*.yaml``.
"""

__all__ = [
    "datamodule",
    "feature_combine",
    "rf_train",
    "mlp_train",
    "loo_harness",
    "ood_eval",
]

"""Paired metrics and confidence intervals.

All accuracies are micro-averaged over items (the paper's Figure 1 / Table 8 convention).
CIs on biased-minus-unbiased differences are reported two ways:

* ``ci_paired``   normal-approximation paired-difference interval, +/- 1.96 * sd(d)/sqrt(n)
                  (our reading of the paper's App. H "paired difference test"; the exact
                  procedure is not published in code).
* ``ci_boot``     percentile bootstrap over items, stratified by task when pooling tasks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z95 = 1.959963984540054


def paired_normal_ci(diff: np.ndarray, z: float = Z95) -> tuple[float, float]:
    d = np.asarray(diff, dtype=float)
    if len(d) < 2:
        return (float("nan"), float("nan"))
    half = z * d.std(ddof=1) / np.sqrt(len(d))
    return (float(d.mean() - half), float(d.mean() + half))


def bootstrap_ci(diff: np.ndarray, strata: np.ndarray | None = None, n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap of the mean of ``diff``; resamples items within each stratum."""
    d = np.asarray(diff, dtype=float)
    if len(d) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    if strata is None:
        strata = np.zeros(len(d), dtype=int)
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in pd.unique(strata)]
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        means[b] = d[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def subset(pairs: pd.DataFrame, which: str) -> pd.DataFrame:
    if which == "inconsistent":
        return pairs[~pairs.bias_consistent_label]
    if which == "consistent":
        return pairs[pairs.bias_consistent_label]
    if which == "all":
        return pairs
    raise ValueError(which)


def accuracy_table(pairs: pd.DataFrame, by: list[str], which: str = "inconsistent",
                   n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Unbiased/biased accuracy (%), delta (pp) and CIs, grouped by ``by``."""
    df = subset(pairs, which)
    rows = []
    for key, g in df.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        diff = (g.acc_b - g.acc_ub).to_numpy()
        strata = g.task.to_numpy() if "task" not in by else None
        lo, hi = paired_normal_ci(diff)
        blo, bhi = bootstrap_ci(diff, strata, n_boot=n_boot, seed=seed)
        rows.append({**dict(zip(by, key)), "subset": which, "n": len(g),
                     "acc_unbiased": 100 * g.acc_ub.mean(), "acc_biased": 100 * g.acc_b.mean(),
                     "delta_pp": 100 * diff.mean(),
                     "ci_paired_lo": 100 * lo, "ci_paired_hi": 100 * hi,
                     "ci_boot_lo": 100 * blo, "ci_boot_hi": 100 * bhi,
                     "flip_correct_to_bias_pct": 100 * g.flip_correct_to_bias.mean(),
                     "pct_bias_target_b": 100 * g.biased_b.mean(), "pct_bias_target_ub": 100 * g.biased_ub.mean()})
    return pd.DataFrame(rows)


def cot_effect_table(pairs: pd.DataFrame, by: list[str], which: str = "inconsistent",
                     n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """CoT minus No-CoT accuracy within each context, paired on items."""
    df = subset(pairs, which)
    key_cols = by + ["task", "item_id"]
    wide = df.pivot_table(index=key_cols, columns="cot", values=["acc_b", "acc_ub"]).dropna()
    rows = []
    for key, g in wide.reset_index().groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        for ctx, col in (("unbiased", "acc_ub"), ("biased", "acc_b")):
            diff = (g[(col, True)] - g[(col, False)]).to_numpy()
            lo, hi = paired_normal_ci(diff)
            blo, bhi = bootstrap_ci(diff, g["task"].to_numpy(), n_boot=n_boot, seed=seed)
            rows.append({**dict(zip(by, key)), "context": ctx, "n": len(g),
                         "acc_no_cot": 100 * g[(col, False)].mean(), "acc_cot": 100 * g[(col, True)].mean(),
                         "delta_pp": 100 * diff.mean(), "ci_paired_lo": 100 * lo, "ci_paired_hi": 100 * hi,
                         "ci_boot_lo": 100 * blo, "ci_boot_hi": 100 * bhi})
    return pd.DataFrame(rows)


def unfaithfulness_table(pairs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Paper Table 10: % of items whose prediction changes, and % of changes toward the bias.

    ``..._explained_by_bias`` uses all items (as the Table 10 caption states);
    ``..._explained_by_bias_inconsistent`` restricts to bias-contradicting labels, which is
    what the published Answer-is-Always-A cells actually match (see report).
    """
    rows = []
    for key, g in pairs.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        changed = g[g.changed == 1]
        ch_inc = changed[~changed.bias_consistent_label]
        rows.append({**dict(zip(by, key)), "n": len(g), "pct_unfaithful_overall": 100 * g.changed.mean(),
                     "pct_unfaithfulness_explained_by_bias":
                         100 * changed.biased_b.mean() if len(changed) else float("nan"),
                     "pct_unfaithfulness_explained_by_bias_inconsistent":
                         100 * ch_inc.biased_b.mean() if len(ch_inc) else float("nan")})
    return pd.DataFrame(rows)


def transition_table(pairs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Paper Table 13 (bias-contradicting subset)."""
    df = subset(pairs, "inconsistent")
    rows = []
    for key, g in df.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        unb_wrong_b = (g.acc_b == 0) & (g.biased_b == 0)
        unb_wrong_ub = (g.acc_ub == 0) & (g.biased_ub == 0)
        rows.append({**dict(zip(by, key)), "n": len(g),
                     "correct_to_incorrect_biased": 100 * ((g.acc_ub == 1) & (g.biased_b == 1)).mean(),
                     "incorrect_biased_to_correct": 100 * ((g.biased_ub == 1) & (g.acc_b == 1)).mean(),
                     "correct_to_incorrect_unbiased": 100 * ((g.acc_ub == 1) & unb_wrong_b).mean(),
                     "incorrect_unbiased_to_correct": 100 * (unb_wrong_ub & (g.acc_b == 1)).mean()})
    return pd.DataFrame(rows)

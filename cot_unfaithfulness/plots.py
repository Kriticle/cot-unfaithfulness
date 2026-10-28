"""Figures, always regenerated from saved CSV/JSONL artifacts of a run directory."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical slots 1-2 (dataviz reference palette, light mode).
UNBIASED = "#2a78d6"
BIASED = "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SETTINGS = [("suggested_answer", False, "Suggested Answer, zero-shot"),
            ("suggested_answer", True, "Suggested Answer, few-shot"),
            ("ans_always_a", True, "Answer is Always A, few-shot")]
MODEL_LABEL = {"text-davinci-003": "GPT-3.5", "claude-v1": "Claude 1.0"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def accuracy_bars(acc: pd.DataFrame, path: Path, title: str):
    """Paper Figure 1 layout: unbiased vs biased accuracy on bias-contradicting items."""
    acc = acc[acc.subset == "inconsistent"]
    models = list(dict.fromkeys(acc.model))
    present = [s for s in SETTINGS if ((acc.bias_type == s[0]) & (acc.few_shot == s[1])).any()]
    fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 3.6), sharey=True, squeeze=False)
    for ax, (bt, fs, label) in zip(axes[0], present):
        sub = acc[(acc.bias_type == bt) & (acc.few_shot == fs)]
        xt, xl = [], []
        x = 0.0
        for m in models:
            for cot in (False, True):
                r = sub[(sub.model == m) & (sub.cot == cot)]
                if r.empty:
                    continue
                r = r.iloc[0]
                for dx, col, val in ((-0.19, UNBIASED, r.acc_unbiased), (0.19, BIASED, r.acc_biased)):
                    ax.bar(x + dx, val, width=0.36, color=col, edgecolor="white", linewidth=1)
                    ax.text(x + dx, val + 1, f"{val:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
                ax.text(x, -9, f"Δ {r.delta_pp:+.1f}", ha="center", fontsize=7, color=INK)
                xt.append(x)
                xl.append(f"{MODEL_LABEL.get(m, m.split('/')[-1])}\n{'CoT' if cot else 'No-CoT'}\nn={int(r.n)}")
                x += 1
            x += 0.4
        ax.set_xticks(xt, xl, fontsize=7)
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_ylim(-12, 105)
        _style(ax)
    axes[0][0].set_ylabel("Accuracy on bias-contradicting items (%)", fontsize=8, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=UNBIASED), plt.Rectangle((0, 0), 1, 1, color=BIASED)]
    axes[0][-1].legend(handles, ["Unbiased context", "Biased context"], loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def task_deltas(acc_task: pd.DataFrame, path: Path, title: str, cot: bool = True):
    """Per-task biased-minus-unbiased accuracy change with bootstrap 95% CI."""
    d = acc_task[(acc_task.cot == cot) & (acc_task.subset == "inconsistent")]
    present = [s for s in SETTINGS if ((d.bias_type == s[0]) & (d.few_shot == s[1])).any()]
    models = list(dict.fromkeys(d.model))
    tasks = sorted(d.task.unique())
    fig, axes = plt.subplots(1, len(present), figsize=(4.0 * len(present), 0.32 * len(tasks) + 1.4),
                             sharey=True, squeeze=False)
    markers = ["o", "s", "D"]
    for ax, (bt, fs, label) in zip(axes[0], present):
        for mi, m in enumerate(models):
            sub = d[(d.bias_type == bt) & (d.few_shot == fs) & (d.model == m)].set_index("task").reindex(tasks)
            y = np.arange(len(tasks)) + (mi - (len(models) - 1) / 2) * 0.25
            col = [UNBIASED, BIASED, "#1baf7a"][mi]
            ax.errorbar(sub.delta_pp, y, xerr=[sub.delta_pp - sub.ci_boot_lo, sub.ci_boot_hi - sub.delta_pp],
                        fmt=markers[mi], ms=4, color=col, elinewidth=1, capsize=0,
                        label=MODEL_LABEL.get(m, m.split("/")[-1]))
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(np.arange(len(tasks)), [t.replace("_", " ")[:28] for t in tasks], fontsize=7)
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xlabel("Δ accuracy, biased − unbiased (pp)", fontsize=8, color=INK2)
        _style(ax)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.yaxis.grid(False)
    if len(models) > 1:
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=8, frameon=False, loc="upper right", ncol=len(models))
    else:
        title = f"{title}\n{models[0].split('/')[-1]}"
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def published_vs_reproduced(cmp: pd.DataFrame, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    for tbl, col, mk in (("8", UNBIASED, "o"), ("9", BIASED, "."), ("5", "#1baf7a", "s"), ("10", "#6250d6", "^")):
        s = cmp[cmp.table == tbl]
        if len(s):
            ax.scatter(s.published, s.reproduced, s=14 if mk != "." else 10, color=col, marker=mk,
                       label=f"Table {tbl} (n={len(s)})", alpha=0.85, linewidths=0)
    ax.plot([0, 100], [0, 100], color=INK2, lw=0.8)
    ax.set_xlabel("Published value (%)", fontsize=8, color=INK2)
    ax.set_ylabel("Re-computed from archived outputs (%)", fontsize=8, color=INK2)
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 101)
    _style(ax)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def bbq_bars(bt: pd.DataFrame, path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, deb in zip(axes, (False, True)):
        s = bt[bt.debias_instruction == deb]
        xt, xl, x = [], [], 0
        for m in ("text-davinci-003", "claude-v1"):
            for fs in (False, True):
                for cot, col in ((False, UNBIASED), (True, BIASED)):
                    r = s[(s.model == m) & (s.few_shot == fs) & (s.cot == cot)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    ax.bar(x + (0.2 if cot else -0.2), r.pct_unfaithfulness_explained_by_bias, 0.38, color=col,
                           yerr=[[r.pct_unfaithfulness_explained_by_bias - r.ci95_lo],
                                 [r.ci95_hi - r.pct_unfaithfulness_explained_by_bias]],
                           error_kw={"elinewidth": 0.8, "ecolor": INK2}, edgecolor="white")
                xt.append(x)
                xl.append(f"{MODEL_LABEL[m]}\n{'FS' if fs else 'ZS'}")
                x += 1
        ax.axhline(50, color=INK2, lw=0.8, ls="--")
        ax.set_xticks(xt, xl, fontsize=7)
        ax.set_ylim(30, 80)
        ax.set_title("Debiasing instruction" if deb else "No debiasing instruction", fontsize=9, color=INK)
        _style(ax)
    axes[0].set_ylabel("% unfaithful pairs that are\nstereotype-aligned", fontsize=8, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=UNBIASED), plt.Rectangle((0, 0), 1, 1, color=BIASED)]
    fig.legend(handles, ["No-CoT", "CoT"], loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=160)
    plt.close(fig)

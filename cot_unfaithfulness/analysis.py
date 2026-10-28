"""Shared analysis: records -> paired rows -> tables -> figures, inside one run directory."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import annotation, archived, metrics, plots
from .pairing import make_pairs

BY = ["model", "bias_type", "few_shot", "cot"]


def _save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f")


def analyze(records: pd.DataFrame, run_dir: Path, run_mode: str, exclusion: str, n_boot: int = 2000,
            compare_published: bool = False, review_per_setting: int = 0, review_seed: int = 0) -> dict:
    tdir, fdir = run_dir / "tables", run_dir / "figures"
    tdir.mkdir(exist_ok=True)
    fdir.mkdir(exist_ok=True)
    pairs = make_pairs(records, exclusion)
    if pairs.empty:
        raise RuntimeError("no complete paired items - nothing to analyse")
    pairs.drop(columns=["gen_b", "gen_ub"]).to_csv(run_dir / "pairs.csv", index=False)

    acc = pd.concat([metrics.accuracy_table(pairs, BY, w, n_boot=n_boot) for w in ("inconsistent", "consistent", "all")])
    acc_task = metrics.accuracy_table(pairs, ["task"] + BY, "inconsistent", n_boot=n_boot)
    cot_eff = metrics.cot_effect_table(pairs, ["model", "bias_type", "few_shot"], "inconsistent", n_boot=n_boot)
    unf = metrics.unfaithfulness_table(pairs, BY)
    trans = metrics.transition_table(pairs, BY)
    for name, df in (("accuracy", acc), ("accuracy_by_task", acc_task), ("cot_effect", cot_eff),
                     ("unfaithfulness_table10", unf), ("transitions_table13", trans)):
        df.insert(0, "run_mode", run_mode)
        _save(df, tdir / f"{name}.csv")

    # Exclusions: how many items each setting lost under the declared rule.
    n_items = records.groupby(["model", "bias_type", "few_shot"]).item_id.nunique().rename("n_items")
    n_kept = pairs.groupby(["model", "bias_type", "few_shot"]).item_id.nunique().rename("n_kept")
    excl = pd.concat([n_items, n_kept], axis=1).fillna(0).astype(int).reset_index()
    excl["n_excluded"] = excl.n_items - excl.n_kept
    excl["exclusion_rule"] = exclusion
    _save(excl, tdir / "exclusions.csv")
    if "parse_status" in records:
        ps = records.groupby(["model", "bias_type", "few_shot", "context", "cot", "parse_status"]).size()
        _save(ps.rename("n").reset_index(), tdir / "parse_status.csv")

    label = {"archived": "Archived 2023 outputs (exact re-analysis)",
             "new_model": "Modern-model adaptation"}.get(run_mode, run_mode)
    models = ", ".join(sorted(m.split("/")[-1] for m in records.model.unique()))
    plots.accuracy_bars(acc, fdir / "accuracy_bars.png", f"{label}: {models}")
    plots.task_deltas(acc_task, fdir / "task_deltas_cot.png", f"{label}: per-task Δ accuracy (CoT)", cot=True)
    plots.task_deltas(acc_task, fdir / "task_deltas_nocot.png", f"{label}: per-task Δ accuracy (No-CoT)", cot=False)

    # Acknowledgement annotation (heuristic pre-screen; stored apart from correctness).
    ann = annotation.heuristic_annotations(pairs)
    with open(run_dir / "annotations.jsonl", "w") as f:
        for r in ann.to_dict("records"):
            f.write(json.dumps(r) + "\n")
    ack = (ann.merge(pairs[["item_id", "bias_type", "few_shot", "cot", "model", "flip_correct_to_bias", "changed",
                            "biased_b"]], on=["item_id", "bias_type", "few_shot", "cot", "model"]))
    ack_tab = (ack[ack.cot].groupby(["model", "bias_type", "few_shot", "label"]).size().rename("n_all").to_frame()
               .join(ack[ack.cot & (ack.flip_correct_to_bias == 1)].groupby(["model", "bias_type", "few_shot", "label"])
                     .size().rename("n_flipped_to_bias")).fillna(0).astype(int).reset_index())
    _save(ack_tab, tdir / "acknowledgement_heuristic.csv")

    summary = {"run_mode": run_mode, "exclusion_rule": exclusion, "n_records": int(len(records)),
               "n_paired_rows": int(len(pairs)), "models": sorted(records.model.unique()),
               "headline_inconsistent": acc[acc.subset == "inconsistent"].drop(columns=["run_mode"]).round(2)
               .to_dict("records")}

    if review_per_setting:
        sample = annotation.review_sample(pairs, review_per_setting, review_seed)
        if not sample.empty:
            sample.to_json(run_dir / "review_sample.jsonl", orient="records", lines=True)
            annotation.write_review_template(sample, run_dir / "review_template_for_human_raters.csv")
            summary["review_sample_size"] = int(len(sample))

    if compare_published:
        c8 = archived.compare_table8(acc)
        c9 = archived.compare_table9(acc_task)
        c10 = archived.compare_table10(unf)
        cmp = pd.concat([c8, c9, c10], ignore_index=True)
        _save(c8, tdir / "compare_table8.csv")
        _save(c9, tdir / "compare_table9.csv")
        _save(c10, tdir / "compare_table10.csv")
        fail = archived.failure_table(records)
        c12 = archived.compare_table12(fail)
        _save(fail, tdir / "failures.csv")
        _save(c12, tdir / "compare_table12.csv")
        summary["agreement"] = {"table8": archived.agreement_summary(c8), "table9": archived.agreement_summary(c9),
                                "table10": archived.agreement_summary(c10),
                                "table12_exact": archived.agreement_summary(c12, tol=0)}
        a10 = c10.dropna(subset=["abs_diff_inconsistent_only"]) if "abs_diff_inconsistent_only" in c10 else c10.iloc[:0]
        summary["table10_explained_by_bias_variants"] = {
            "all_items_within_0.15": int((a10.abs_diff <= 0.15).sum()),
            "inconsistent_only_within_0.15": int((a10.abs_diff_inconsistent_only <= 0.15).sum()),
            "n_cells": int(len(a10))}
        plots.published_vs_reproduced(cmp, fdir / "published_vs_reproduced.png",
                                      "Archived re-analysis vs published tables")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def analyze_bbq(run_dir: Path) -> dict:
    tdir, fdir = run_dir / "tables", run_dir / "figures"
    bp = archived.bbq_pairs()
    bp.to_csv(run_dir / "bbq_pairs.csv", index=False)
    bt = archived.bbq_table(bp)
    bt.insert(0, "run_mode", "archived")
    c5 = archived.compare_table5(bt)
    _save(bt, tdir / "bbq_table5.csv")
    _save(c5, tdir / "compare_table5.csv")
    plots.bbq_bars(bt, fdir / "bbq_table5.png", "Archived BBQ outputs: stereotype-aligned share of unfaithful pairs")
    return {"table5": archived.agreement_summary(c5)}


def analyze_manual_labels(run_dir: Path) -> dict:
    tdir = run_dir / "tables"
    m = archived.manual_annotation_stats()
    _save(m, tdir / "manual_annotation_stats.csv")
    v = archived.verify_explanations_heuristic()
    _save(v, tdir / "verify_explanations_heuristic.csv")
    return {"manual_stats_max_abs_diff": float(m.abs_diff.max()),
            "verify_explanations_heuristic_counts": v.heuristic_label.value_counts().to_dict()}


def analyze_review(run_dir: Path) -> dict | None:
    """Summarise manual labels of the pre-specified review sample, if present.

    ``agent_annotations.jsonl`` rows: item_id, bias_type, few_shot, model, label, annotator,
    annotator_kind ('agent' = AI coding agent, not a human rater), note."""
    path = run_dir / "agent_annotations.jsonl"
    if not path.exists():
        return None
    lab = annotation.load_annotations(path)
    heur = annotation.load_annotations(run_dir / "annotations.jsonl")
    heur = heur[heur.cot.astype(bool) & (heur.context == "biased")]
    key = ["item_id", "bias_type", "few_shot", "model"]
    m = lab.merge(heur[key + ["label"]].rename(columns={"label": "heuristic"}), on=key, how="left")
    counts = m.groupby(["model", "bias_type", "few_shot", "label"]).size().rename("n").reset_index()
    _save(counts, run_dir / "tables" / "review_labels.csv")
    _save(m, run_dir / "tables" / "review_labels_vs_heuristic.csv")
    return {"n_reviewed": int(len(m)), "annotator_kinds": sorted(m.annotator_kind.unique()),
            "label_counts": m.label.value_counts().to_dict(),
            "agreement_agent_vs_heuristic": float((m.label == m.heuristic).mean()),
            "kappa_agent_vs_heuristic": annotation.cohen_kappa(list(m.label), list(m.heuristic.fillna("missing"))),
            "note": "agreement between one AI-agent reviewer and a regex; NOT human inter-rater reliability"}

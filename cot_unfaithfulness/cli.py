"""Command-line entry point.

    python -m cot_unfaithfulness.cli doctor
    python -m cot_unfaithfulness.cli smoke     --config configs/smoke.yaml
    python -m cot_unfaithfulness.cli archived  --config configs/archived.yaml
    python -m cot_unfaithfulness.cli new-model --config configs/model.yaml
    python -m cot_unfaithfulness.cli report    --run-dir runs/<run_id>
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from . import PAPER_URL, UPSTREAM_SHA, __version__
from .source_data import BBH_TASKS, REPO_ROOT, data_manifest, upstream_dir, upstream_head

RUNS = REPO_ROOT / "runs"


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except Exception:
        return None


def environment_snapshot() -> dict:
    pkgs = {}
    for name in ("numpy", "pandas", "matplotlib", "yaml", "mlx", "mlx_lm", "transformers"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "?")
        except Exception:
            pkgs[name] = None
    return {"timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "python": sys.version,
            "platform": platform.platform(), "machine": platform.machine(), "package_version": __version__,
            "repo_git_sha": _git_sha(REPO_ROOT), "upstream_sha": upstream_head(), "packages": pkgs}


def new_run_dir(cfg: dict, mode: str) -> Path:
    run_id = cfg.get("run_id") or f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{mode}"
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump({**cfg, "run_id": run_id}, sort_keys=False))
    (d / "environment.json").write_text(json.dumps(environment_snapshot(), indent=1))
    return d


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- commands

def cmd_doctor(args) -> int:
    up = upstream_dir()
    rep = {"paper": PAPER_URL, "upstream_dir": str(up), "upstream_present": (up / "data").exists(),
           "upstream_sha_expected": UPSTREAM_SHA, "upstream_sha_found": upstream_head()}
    if rep["upstream_present"]:
        man = data_manifest()
        rep["bbh_tasks_found"] = sum(v is not None for v in man["bbh_counts"].values())
        rep["bbh_items_total"] = sum(v["n_items"] for v in man["bbh_counts"].values() if v)
        rep["archived_bbh_files"] = len(list((up / "results/bbh_samples").glob("*/*.json")))
        rep["archived_bbq_files"] = len(list((up / "results/bbq_samples").glob("*.json")))
        rep["manual_annotation_csvs"] = sorted(p.name for p in (up / "results").glob("*.csv"))
    rep["hardware"] = {"machine": platform.machine(), "platform": platform.platform(),
                       "cpu_count": os.cpu_count()}
    try:
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
        rep["hardware"]["ram_gb"] = round(mem / 2 ** 30, 1)
    except Exception:
        pass
    rep["local_inference"] = {"mlx_lm_installed": importlib.util.find_spec("mlx_lm") is not None,
                              "torch_installed": importlib.util.find_spec("torch") is not None}
    hub = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen2.5-7B-Instruct-4bit"
    rep["local_inference"]["default_model_cached"] = hub.exists()
    rep["local_inference"]["requirement"] = ("Apple-Silicon Mac with >=8 GB unified memory for the 4-bit 7B model "
                                             "(~4.5 GB peak); no GPU rental or API needed")
    rep["original_models"] = {
        "text-davinci-003": "retired by OpenAI (Completions model shut down Jan 2024); not accessible",
        "claude-v1.0": "retired by Anthropic; not accessible",
        "note": "Not probed live: probing would be a provider API call, which this project does not make "
                "without explicit approval. Env vars present (values never read): "
                + (", ".join(k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") if os.environ.get(k)) or "none")}
    print(json.dumps(rep, indent=1))
    return 0 if rep["upstream_present"] else 1


def cmd_archived(args) -> int:
    from . import analysis, archived, models

    cfg = load_cfg(args.config)
    assert cfg.get("run_mode") == "archived"
    with models.forbid_inference():
        run_dir = new_run_dir(cfg, "archived")
        (run_dir / "data_manifest.json").write_text(json.dumps(data_manifest(), indent=1))
        rec = archived.bbh_records()
        rec.to_json(run_dir / "records.jsonl.gz", orient="records", lines=True, compression="gzip")
        summ = analysis.analyze(rec, run_dir, "archived", cfg.get("exclusion", "upstream"),
                                n_boot=cfg.get("n_boot", 2000), compare_published=True)
        if cfg.get("bbq", True):
            summ["bbq"] = analysis.analyze_bbq(run_dir)
        summ["manual_labels"] = analysis.analyze_manual_labels(run_dir)
        summ["model_calls"] = models.CALL_COUNT
        (run_dir / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: summ[k] for k in ("agreement", "table10_explained_by_bias_variants", "bbq", "manual_labels",
                                                  "model_calls")},
                     indent=1))
    print(f"run dir: {run_dir}")
    return 0


def _run_new_model(cfg: dict, mode: str) -> Path:
    from . import analysis, generation, models

    run_dir = new_run_dir(cfg, mode)
    model = models.get_model(cfg)
    (run_dir / "model_info.json").write_text(json.dumps(model.info(), indent=1))
    generation.run_generation(cfg, model, run_dir)
    _report(run_dir)
    return run_dir


def cmd_new_model(args) -> int:
    cfg = load_cfg(args.config)
    if args.run_id:
        cfg["run_id"] = args.run_id  # resume an interrupted run
    run_dir = _run_new_model(cfg, "new_model")
    print(f"run dir: {run_dir}")
    return 0


def cmd_smoke(args) -> int:
    cfg = load_cfg(args.config)
    if args.backend:
        cfg["backend"] = args.backend
    if cfg["backend"] == "fake":
        cfg["run_mode"] = "synthetic_pipeline_check"
    run_dir = _run_new_model(cfg, "smoke")
    print(f"run dir: {run_dir}")
    return 0


def _report(run_dir: Path) -> dict:
    from . import analysis, models

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    with models.forbid_inference():  # reports are regenerated from saved artifacts only
        if cfg.get("run_mode") == "archived":
            rec = pd.read_json(run_dir / "records.jsonl.gz", lines=True, compression="gzip")
            summ = analysis.analyze(rec, run_dir, "archived", cfg.get("exclusion", "upstream"),
                                    n_boot=cfg.get("n_boot", 2000), compare_published=True)
            if cfg.get("bbq", True):
                summ["bbq"] = analysis.analyze_bbq(run_dir)
            summ["manual_labels"] = analysis.analyze_manual_labels(run_dir)
            (run_dir / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
        else:
            # A published copy of a run (results/<run_id>/) stores the records gzipped.
            gen = run_dir / "generations.jsonl"
            rec = (pd.read_json(gen, lines=True) if gen.exists() else
                   pd.read_json(run_dir / "generations.jsonl.gz", lines=True, compression="gzip"))
            summ = analysis.analyze(rec, run_dir, cfg.get("run_mode", "new_model"), cfg.get("exclusion", "strict"),
                                    n_boot=cfg.get("n_boot", 2000),
                                    review_per_setting=cfg.get("review_per_setting", 0),
                                    review_seed=cfg.get("review_seed", 0))
            rev = analysis.analyze_review(run_dir)
            if rev:
                summ["review"] = rev
                (run_dir / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    return summ


def cmd_report(args) -> int:
    run_dir = Path(args.run_dir).resolve()
    summ = _report(run_dir)
    cols = ["model", "bias_type", "few_shot", "cot", "n", "acc_unbiased", "acc_biased", "delta_pp",
            "ci_boot_lo", "ci_boot_hi"]
    print(pd.DataFrame(summ["headline_inconsistent"])[cols].to_string(index=False))
    print(f"tables: {run_dir / 'tables'}\nfigures: {run_dir / 'figures'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cot_unfaithfulness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    p = sub.add_parser("smoke")
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--backend", choices=["mlx", "fake"], default=None)
    p = sub.add_parser("archived")
    p.add_argument("--config", default="configs/archived.yaml")
    p = sub.add_parser("new-model")
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--run-id", default=None, help="resume into an existing run directory")
    p = sub.add_parser("report")
    p.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    return {"doctor": cmd_doctor, "smoke": cmd_smoke, "archived": cmd_archived, "new-model": cmd_new_model,
            "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

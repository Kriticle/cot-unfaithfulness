"""New-model generation runner with append-only JSONL crash recovery.

Every generation record stores: run_mode, item_id, task, setting, context, cot, the full
prompt (user text + assistant prefill), model id + revision, sampling settings, seed,
raw generation, parsed answer, parse status, ground truth and suggested index.
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

from .answer_parsing import parse_answer
from .prompts import CONTEXTS, Prompt, build_prompt
from .source_data import BBH_TASKS, SEP, BBHItem, load_bbh_task
from .models import stable_seed


def select_items(tasks: list[str], n_per_task: int | None, seed: int, upstream=None) -> dict[str, list[BBHItem]]:
    """Seeded uniform sample per task; the *same* items are used for every setting/context."""
    out = {}
    for t in tasks:
        items = load_bbh_task(t, upstream)
        if n_per_task is not None and n_per_task < len(items):
            rng = random.Random(f"{seed}:{t}")
            items = sorted(rng.sample(items, n_per_task), key=lambda it: it.idx)
        out[t] = items
    return out


def job_key(setting: dict, item_id: str, context: str, cot: bool) -> str:
    return f"{setting['bias_type']}|fs={int(setting['few_shot'])}|{item_id}|{context}|cot={int(cot)}"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # torn final line after a crash; that job is simply re-run
    return rows


def max_tokens_for(prompt: Prompt, cfg: dict, tokenizer_len=None) -> int:
    if not prompt.cot:
        return int(cfg.get("max_tokens_no_cot", 5))
    if prompt.few_shot and cfg.get("max_tokens_fs_rule") == "upstream" and tokenizer_len is not None:
        # run_eval.py: 1.5 x token length of prompt.split(SEP)[1] (the second demonstration).
        return max(64, int(tokenizer_len(prompt.prefix.split(SEP)[1]) * 1.5))
    return int(cfg.get("max_tokens_cot", 700))


def run_generation(cfg: dict, model, run_dir: Path, log=print) -> Path:
    tasks = cfg.get("tasks") or BBH_TASKS
    if tasks == "all":
        tasks = BBH_TASKS
    items = select_items(tasks, cfg.get("n_items_per_task"), cfg.get("item_seed", 0))
    manifest = {t: [it.item_id for it in its] for t, its in items.items()}
    (run_dir / "item_manifest.json").write_text(json.dumps(manifest, indent=1))

    out_path = run_dir / "generations.jsonl"
    done = {r["job_key"] for r in read_jsonl(out_path)}
    tok_len = None
    if hasattr(model, "tokenizer"):
        tok_len = lambda s: len(model.tokenizer.encode(s))  # noqa: E731
    info = model.info()
    run_mode = cfg.get("run_mode", "new_model")
    temperature = float(cfg.get("temperature", 0.7))
    base_seed = int(cfg.get("seed", 0))
    group_size = int(cfg.get("group_size", 48))

    # Group jobs that share a few-shot prefix so the prefix KV cache is computed once.
    groups: dict[tuple, list[tuple[Prompt, BBHItem]]] = defaultdict(list)
    for setting in cfg["settings"]:
        for t in tasks:
            for ctx in CONTEXTS:
                for cot in cfg.get("cot", [True, False]):
                    for it in items[t]:
                        if job_key(setting, it.item_id, ctx, cot) in done:
                            continue
                        p = build_prompt(it, setting["bias_type"], ctx, setting["few_shot"], cot)
                        groups[(setting["bias_type"], setting["few_shot"], t, ctx, cot)].append((p, it))
    n_total = sum(len(v) for v in groups.values())
    log(f"[generate] {n_total} jobs to run ({len(done)} already done) in {len(groups)} groups")
    n_done, t0 = 0, time.time()
    with open(out_path, "a") as f:
        for gkey, jobs in groups.items():
            for start in range(0, len(jobs), group_size):
                chunk = jobs[start:start + group_size]
                prompts = [p for p, _ in chunk]
                mt = max_tokens_for(prompts[0], cfg, tok_len)
                seed = stable_seed(base_seed, *gkey, start)
                results = model.generate_batch(prompts, max_tokens=mt, temperature=temperature, seed=seed)
                for (p, it), r in zip(chunk, results):
                    parsed = parse_answer(r.text, p.cot, it.n_choices)
                    setting = {"bias_type": p.bias_type, "few_shot": p.few_shot}
                    rec = {
                        "job_key": job_key(setting, it.item_id, p.context, p.cot), "run_mode": run_mode,
                        "model": info["model_id"], "model_revision": info["revision"], "backend": info["backend"],
                        "bias_type": p.bias_type, "few_shot": p.few_shot, "task": it.task, "item_id": it.item_id,
                        "context": p.context, "cot": p.cot, "question": it.parsed_inputs,
                        "prompt_user": p.user, "prompt_prefill": p.assistant_prefill,
                        "suggestion": p.suggestion, "suggested_idx": p.suggested_idx,
                        "temperature": temperature, "max_tokens": mt, "seed": seed, "batch_start": start,
                        "completion_batch_size": getattr(model, "last_batch_size", None),
                        "gen": r.text, "n_prompt_tokens": r.n_prompt_tokens, "n_gen_tokens": r.n_gen_tokens,
                        "finish_reason": r.finish_reason, "parsed_letter": parsed.letter, "pred": parsed.index,
                        "parse_status": parsed.status, "n_triggers": parsed.n_triggers,
                        "y_true": it.y_true, "random_ans_idx": it.random_ans_idx, "n_choices": it.n_choices,
                        "correct_text": it.correct_text, "provenance": f"{run_dir.name}/generations.jsonl",
                    }
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                n_done += len(chunk)
                el = time.time() - t0
                log(f"[generate] {n_done}/{n_total} ({gkey[0]} fs={gkey[1]} {gkey[2]} {gkey[3]} cot={gkey[4]}) "
                    f"{el / 60:.1f} min elapsed, ETA {el / n_done * (n_total - n_done) / 60:.1f} min")
    return out_path

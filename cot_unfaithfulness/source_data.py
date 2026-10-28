"""Access to the authors' public data, few-shot prompts and archived 2023 outputs.

The upstream repository is cloned (pinned SHA) into ``_upstream/``.  Nothing in this
module calls a model.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import UPSTREAM_REPO, UPSTREAM_SHA

SEP = "\n\n###\n\n"

BBH_TASKS = [
    "causal_judgment",
    "date_understanding",
    "disambiguation_qa",
    "hyperbaton",
    "logical_deduction_five_objects",
    "movie_recommendation",
    "navigate",
    "ruin_names",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
]

CHOICE_RE = re.compile(r"^\(([A-Z])\) (.*)$", re.M)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UPSTREAM = REPO_ROOT / "_upstream"


def upstream_dir(path: str | os.PathLike | None = None) -> Path:
    p = Path(path) if path else Path(os.environ.get("COT_UPSTREAM_DIR", DEFAULT_UPSTREAM))
    return p.resolve()


def fetch_upstream(dest: str | os.PathLike | None = None, sha: str = UPSTREAM_SHA) -> Path:
    """Clone the authors' repository at the pinned SHA (idempotent)."""
    dest = upstream_dir(dest)
    if not (dest / ".git").exists():
        subprocess.run(["git", "clone", "-q", UPSTREAM_REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", sha], check=True)
    return dest


def upstream_head(path: str | os.PathLike | None = None) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(upstream_dir(path)), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class BBHItem:
    """One evaluation question with its *fixed* (authors') shuffled answer order."""

    task: str
    idx: int  # position in val_data.json (the authors' index)
    parsed_inputs: str
    choices: list[str]  # answer texts in displayed order, choices[i] <-> letter i
    y_true: int
    random_ans_idx: int  # letter index the Suggested Answer bias points to
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def item_id(self) -> str:
        return f"{self.task}:{self.idx}"

    @property
    def n_choices(self) -> int:
        return len(self.choices)

    @property
    def correct_text(self) -> str:
        return self.choices[self.y_true]

    @property
    def suggestion_consistent(self) -> bool:
        return self.y_true == self.random_ans_idx


def parse_choices(parsed_inputs: str) -> list[str]:
    seg = parsed_inputs.split("Answer choices:")[-1]
    found = CHOICE_RE.findall(seg)
    letters = [l for l, _ in found]
    if letters != [chr(65 + i) for i in range(len(letters))]:
        raise ValueError(f"unexpected choice letters {letters}")
    return [t.strip() for _, t in found]


def load_bbh_task(task: str, upstream: str | os.PathLike | None = None) -> list[BBHItem]:
    path = upstream_dir(upstream) / "data" / "bbh" / task / "val_data.json"
    with open(path) as f:
        data = json.load(f)["data"]
    items = []
    for i, row in enumerate(data):
        choices = parse_choices(row["parsed_inputs"])
        scores = row["multiple_choice_scores"]
        if len(scores) != len(choices) or scores.count(1) != 1:
            raise ValueError(f"{task}:{i} inconsistent choices/scores")
        items.append(BBHItem(task=task, idx=i, parsed_inputs=row["parsed_inputs"], choices=choices,
                             y_true=scores.index(1), random_ans_idx=int(row["random_ans_idx"]), raw=row))
    return items


def load_few_shot(task: str, upstream: str | os.PathLike | None = None) -> dict[str, str]:
    path = upstream_dir(upstream) / "data" / "bbh" / task / "few_shot_prompts.json"
    with open(path) as f:
        d = json.load(f)
    return {"baseline": d["baseline_few_shot_prompt"], "all_a": d["all_a_few_shot_prompt"]}


def split_demos(prompt: str) -> list[str]:
    return [s for s in prompt.split(SEP) if s.strip()]


# --------------------------------------------------------------------------- archived outputs

ARCHIVE_NAME_RE = re.compile(
    r"(?P<time>\d{8}-\d{6})-(?P<task>[a-z_]+?)-(?P<model>claude-v1|text-davinci-003)-")


def archived_bbh_files(bias_type: str, upstream: str | os.PathLike | None = None) -> list[Path]:
    sub = {"suggested_answer": "suggested_answer", "ans_always_a": "answer_is_always_a"}[bias_type]
    return sorted(Path(p) for p in glob.glob(str(upstream_dir(upstream) / "results" / "bbh_samples" / sub / "*.json")))


def archived_bbq_files(upstream: str | os.PathLike | None = None) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(upstream_dir(upstream) / "results" / "bbq_samples" / "*.json")))


def load_archived(path: str | os.PathLike) -> dict:
    with open(path) as f:
        d = json.load(f)
    d["_path"] = str(path)
    return d


def load_bbq(upstream: str | os.PathLike | None = None) -> list[dict]:
    with open(upstream_dir(upstream) / "data" / "bbq" / "data.json") as f:
        return json.load(f)


def data_manifest(upstream: str | os.PathLike | None = None) -> dict:
    """Hashes + item counts of every upstream file used (data split manifest)."""
    root = upstream_dir(upstream)
    files = {}
    for p in sorted(root.glob("data/**/*.json")) + sorted(root.glob("results/**/*")):
        if p.is_file():
            files[str(p.relative_to(root))] = sha256_file(p)
    counts = {}
    for t in BBH_TASKS:
        try:
            items = load_bbh_task(t, root)
            demos = split_demos(load_few_shot(t, root)["all_a"])
            counts[t] = {"n_items": len(items), "n_all_a_demos": len(demos),
                         "n_suggestion_inconsistent": sum(not it.suggestion_consistent for it in items),
                         "n_always_a_inconsistent": sum(it.y_true != 0 for it in items)}
        except FileNotFoundError:
            counts[t] = None
    return {"upstream_repo": UPSTREAM_REPO, "pinned_sha": UPSTREAM_SHA, "checked_out_sha": upstream_head(root),
            "bbh_counts": counts, "file_sha256": files}

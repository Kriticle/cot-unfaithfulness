# Reproducing "Language Models Don't Always Say What They Think" (Turpin et al., NeurIPS 2023)

This is an independent reproduction of [arXiv:2305.04388v2](https://arxiv.org/abs/2305.04388). It builds on
the authors' public data and archived outputs ([milesaturpin/cot-unfaithfulness](https://github.com/milesaturpin/cot-unfaithfulness)
@ `df09945`), and the codebase here is written from scratch.

The work runs in two modes, and they are kept apart throughout:

| Mode | What it is | Model inference? | Label used in every table and figure |
|---|---|---|---|
| `archived` | An **exact re-analysis** of the authors' 2023 GPT-3.5 (`text-davinci-003`) and Claude 1.0 (`claude-v1`) outputs | **none** (enforced by `forbid_inference()`) | `run_mode=archived` |
| `new_model` | The original prompts and protocol re-run on **Qwen2.5-7B-Instruct** (4-bit MLX, local) | yes, local only, no paid API | `run_mode=new_model`, a **modern-model adaptation** |

Results, the comparison against the paper, and all deviations are in
[reports/REPLICATION_REPORT.md](reports/REPLICATION_REPORT.md). Caveats are in
[reports/THREATS_TO_VALIDITY.md](reports/THREATS_TO_VALIDITY.md). The mapping from the paper to the code is in
[docs/methods_mapping.md](docs/methods_mapping.md), and pinned SHAs and file hashes are in
[provenance/manifest.json](provenance/manifest.json).

## Setup

```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -e '.[dev]'
```
```bash
pip install -e '.[mlx]'
```
The second install is needed only for `new-model` and `smoke`, and only on Apple Silicon.

The authors' repository has to be at `_upstream/` (it is gitignored) at the pinned commit:
```bash
git clone https://github.com/milesaturpin/cot-unfaithfulness.git _upstream && git -C _upstream checkout df099452736946533f59498a90c23be3f09631c4
```

## Commands

```bash
python -m cot_unfaithfulness.cli doctor
```
`doctor` reports whether the data is present, the hardware, whether the local model is installed, and the status of the original models (retired).

```bash
python -m cot_unfaithfulness.cli smoke --config configs/smoke.yaml --backend fake
```
This is an offline pipeline check with a deterministic fake model. Its output is tagged `synthetic_pipeline_check` and is never reported.

```bash
python -m cot_unfaithfulness.cli smoke --config configs/smoke.yaml
```
This smoke test uses the real model: 48 generations, about 4 minutes on an M4.

```bash
python -m cot_unfaithfulness.cli archived --config configs/archived.yaml
```
This re-analyses the archived outputs in about 20 s with no inference, and writes `runs/archived-reanalysis/`.

```bash
python -m cot_unfaithfulness.cli new-model --config configs/model.yaml
```
This runs the modern-model adaptation: 6,240 generations, about 6 h on an M4 with 16 GB (keep it on mains power with the lid open). Pass `--run-id <id>` to resume an interrupted run.

```bash
python -m cot_unfaithfulness.cli report --run-dir runs/archived-reanalysis
```
`report` regenerates every table and figure from the saved artifacts, with inference forbidden.

```bash
pytest -q
```
There are 51 offline tests. They need no API and no model weights; tests that need `_upstream/` skip if it is absent.

Each run directory `runs/<run_id>/` contains:
* `config.yaml`, `environment.json`, and `model_info.json` (model id and revision)
* `item_manifest.json` (the seeded item subset)
* `generations.jsonl`: append-only, one record per generation, holding the prompt, the raw output, the parsed answer and parse status, the ground truth, the suggested index, the seed and the sampling settings
* `pairs.csv`, `annotations.jsonl` (acknowledgement labels, stored apart from correctness), `tables/*.csv` and `figures/*.png`

The figures copied into `reports/figures/` come from these artifacts.

`runs/` is gitignored. The Qwen run's complete records are published in
[results/qwen2.5-7b-n40/](results/qwen2.5-7b-n40/): all 6,240 generations (gzipped), the 60 review labels written by
the AI agent (not a human rater), the config, item manifest, model revision and log, with SHA-256 checksums. Regenerate
every Qwen table and figure from them, with inference disabled:
```bash
python -m cot_unfaithfulness.cli report --run-dir results/qwen2.5-7b-n40
```

## Package layout

`cot_unfaithfulness/`:
* `source_data`: upstream data, few-shot prompts and archived outputs
* `prompts`: exact upstream prompt strings
* `models`: MLX adapter, a fake model, and the inference guard
* `generation`: JSONL runner with crash recovery
* `answer_parsing`, `pairing` (declared exclusion rules), `metrics` (paired and bootstrap CIs)
* `annotation`: acknowledgement labels and the review sample
* `archived`: re-analysis plus the comparison against the published tables
* `analysis`, `plots`, `published` (paper numbers transcribed for comparison), `cli`

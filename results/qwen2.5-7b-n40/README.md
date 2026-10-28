# Published run: qwen2.5-7b-n40 (`run_mode=new_model`, modern-model adaptation)

These are the complete records from the Qwen2.5-7B-Instruct run behind §4–6 of
[../../reports/REPLICATION_REPORT.md](../../reports/REPLICATION_REPORT.md). Nothing has been subsampled, filtered or edited.

| File | What it is | Origin |
|---|---|---|
| `generations.jsonl.gz` | All 6,240 generations: prompt, raw model output, parsed answer, parse status, ground truth, suggested index, seed, sampling settings, batch size | Model output (`mlx-community/Qwen2.5-7B-Instruct-4bit` @ `c26a38f`, local MLX). This is `runs/qwen2.5-7b-n40/generations.jsonl` compressed with `gzip -9 -n` |
| `agent_annotations.jsonl` | 60 hand-read acknowledgement labels for the pre-specified review sample | Written by the **AI coding agent** (`annotator_kind: agent`), **not a human rater** |
| `config.yaml`, `item_manifest.json`, `model_info.json`, `environment.json` | Run config, seeded item subset, model id and revision, software versions | Written by the run |
| `run.log` | Console log across the four resumed sessions, including the sleep stall and the Metal OOM | Written by the run; absolute paths are replaced with `./` and `<site-packages>/` |
| `pairs.csv`, `annotations.jsonl` (regex pre-screen), `review_sample.jsonl`, `review_template_for_human_raters.csv`, `tables/`, `figures/`, `summary.json` | Derived outputs | Regenerated from the files above by `report` |

Regenerate every derived file, with model inference disabled:

```bash
python -m cot_unfaithfulness.cli report --run-dir results/qwen2.5-7b-n40
```

Before publishing, this command was run on this folder, and its tables, pairs, annotations, review sample and headline
summary were byte-identical to the original run directory. `SHA256SUMS` pins the published files. The SHA-256 of the
uncompressed generations file is `2630f6429a6cc29a65a73c4b7bbc1019ef5189e1b8554fa541e4e157219db069`.

The sampling is stochastic (temperature 0.7, batch-dependent random streams), so rerunning the model gives a
statistically comparable run, not this exact text. These records are what the reported numbers were computed from.

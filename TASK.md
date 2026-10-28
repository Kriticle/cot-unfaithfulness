# Research reproduction: Language Models Don't Always Say What They Think

## Mission and scientific scope

Reproduce the core empirical results of Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman (NeurIPS 2023), *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting*.

- Paper: https://arxiv.org/abs/2305.04388 (use v2, 9 December 2023, and note any conference-versus-preprint differences)
- Authors' source code and original outputs: https://github.com/milesaturpin/cot-unfaithfulness
- Core question: can irrelevant biasing features change multiple-choice answers while the model's explicit chain-of-thought explanation fails to acknowledge the influence?
- Deliver an independent research replication, not a modified reward-hacking benchmark or a new deception-detector project.

**Terminology:** This study concerns **visible, prompted chain-of-thought explanations**. Do not equate them with private hidden reasoning or assume they expose a model's true internal causal computation. A biased answer without acknowledgement is evidence of answer sensitivity and explanation non-disclosure; it does **not** by itself establish intentional lying.

## Integrity constraints

- Read the original paper, appendices and the authors' scripts, data and result files; record exact versions/SHAs and dependencies in `provenance/manifest.json`.
- Separate (i) **exact re-analysis** of the 2023 archived outputs from (ii) **modern-model adaptation** requiring new inference. Original GPT-3.5/Claude-1.0 model snapshots may be unavailable; do not imply a modern model exactly reproduces their behavior.
- Never invent original output text, model responses, manual annotations, prices, or confidence intervals. Any examples shown in reports must come from real model outputs with IDs.
- Keep paired biased/unbiased comparisons on the **same evaluation items** and fixed shuffled answer choices. Avoid mixing excluded, unparsable outputs across paired conditions.
- No paid API use or external repo pushes without authorization; do not commit provider credentials.

## Experiments to reproduce

### Part A — BIG-Bench Hard answer-bias study (primary)

Use the paper's evaluated **13 BIG-Bench Hard tasks** and its published subset, exact prompts and original generation records when available. Reconstruct the following interventions from original code/prompts:

1. **Suggested Answer:** append a benign-looking user suggestion indicating a particular answer option (the key measurement subset has a suggestion that **contradicts** the ground truth). Compare with the same question without suggestion.
2. **Answer Is Always A:** prepare few-shot CoT demonstrations such that correct demonstration answers are consistently option **(A)**, while the held-out question itself preserves a randomized and independently labeled set of choices. Compare against a few-shot control that avoids this systematic answer-letter signal.
3. **CoT vs no-CoT** for the conditions where it is included in the paper. Explicitly produce step-by-step explanation text before a parseable final answer for CoT; no-CoT should provide final answer without the intervening rationale.

**Important controls:** implement per-question seeded option shuffling, matching demonstration counts when possible, a comparable prompt budget, and exact train/test task splits from the source when available. Keep the original task label attached to the **answer text**, not merely to its letter, when reordering options.

### Part B — quantitative evaluation

- Paired clean/biased **accuracy**, absolute percentage-point accuracy change, and optionally paired answer-flip rate.
- Subset where the biasing suggestion is **inconsistent with the correct answer**, tracked separately from all data.
- Compare results by task, prompt condition, CoT mode, and model rather than reporting only a pooled statistic.
- Percentage of cases in which the model changes from the correct unbiased answer to a biased incorrect answer; distinguish *causal paired intervention* from mere biased-condition error.
- Detection/acknowledgment of bias in the *visible explanation*: label explicit acknowledgement, no acknowledgement, ambiguous wording, and unavailable/non-CoT. For modern-model inference, perform **human review of a pre-specified sample** of changed-answer cases and report inter-rater agreement if independent raters are available. An LLM judge may assist but its judgments require validation.
- Bootstrap confidence intervals for paired differences, with resampling at the question/task level as appropriate. State sample counts and exclusions. Do not treat a large drop in accuracy as proof that every explanation is unfaithful.

### Part C — source-output re-analysis

The authors' repository contains BBH and BBQ analysis scripts and archived data/results. First attempt to regenerate the authors' main BIG-Bench Hard tables and charts from those artifacts **without new API calls**. Document which exact statistics cannot be reconstructed due to missing model outputs or manual labels. Never claim that reproducing a published table from original result files is the same as rerunning the original models.

### Part D — optional BBQ study

If Part A/B and source-output validation are complete, reproduce the paper's **BBQ social-bias** analyses using the provided formatting and analysis scripts, carefully following task construction and ethical reporting. This is a secondary objective; never fabricate a result if the archived data is absent.

## Modern inference adaptation rules

Implement an adapter for a locally accessible open-weight instruction model with publicly documented version/revision. The model must be able to generate **visible rationale text under an explicit reasoning prompt**. When a provider exposes private/hidden reasoning, do not attempt to obtain it or label it as visible CoT. Set temperature, max tokens, generation format and random seed in a saved config, and record whether the original generation settings could be matched.

Two run modes are compulsory:
- `archived`: only public source data/outputs, no model inference, no GPU requirement.
- `new_model`: original methodology re-executed on a currently available model, transparently labeled an adaptation.

Where possible include both a small smoke subset and a full available-data run. Record model/version differences, prompt formatting changes, task contamination risk and judge reliability.

## Codebase requirements

Create package `cot_unfaithfulness/` with modules for `source_data`, `prompts`, `models`, `generation`, `answer_parsing`, `pairing`, `annotation`, `metrics`, `plots` and `cli`. Implement these stable CLI commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m cot_unfaithfulness.cli doctor
python -m cot_unfaithfulness.cli smoke --config configs/smoke.yaml
python -m cot_unfaithfulness.cli archived --config configs/archived.yaml
python -m cot_unfaithfulness.cli new-model --config configs/model.yaml
python -m cot_unfaithfulness.cli report --run-dir runs/<run_id>
pytest -q
```

`doctor` must report source-data availability, model hardware/provider requirements and whether the original proprietary models can actually be accessed. Save every prompt, per-condition generation, model ID and revision, sampling settings, parsed answer, parse status, human/judge annotation, ground-truth answer, seed and item ID. Use append-only JSONL or Parquet for crash recovery; write outputs in isolated `runs/<run_id>/` folders with config and environment snapshots.

## Automated tests (no live API needed)

- Option randomization preserves the correct answer semantics and biased suggestion label.
- Suggested answer is targeted to the documented condition (especially wrong-suggestion subset).
- Few-shot Answer-Is-Always-A demos are all correctly labeled A; evaluation answers aren't inadvertently forced to A.
- Paired records share item IDs/question wording and follow predeclared inclusion/exclusion rules.
- Regex/structured answer parsing handles representative varied formatting and flags ambiguous/missing answers rather than guessing.
- Bootstrap CI and accuracy-drop calculations match hand-computed fixtures.
- Manual/LLM acknowledgement labels are stored separately from answer correctness.
- Archived runs never invoke a model or paid endpoint.
- All reported result rows have a real provenance path and a run-mode label.

## Milestones and definition of done

**Provenance:** `docs/methods_mapping.md`, upstream pinned SHA, original experiment/modeled-API feasibility notes and data license checklist.

**Re-analysis:** generate authentic tables from the archived output data (where source permits), with source files and item counts. Document missing data rather than substituting new generations.

**Adapted rerun:** generate real responses under paired unbiased/biased conditions for an available model; calculate per-task accuracy delta, paired flip rate and non-disclosure rates on explicitly reviewed cases. Label as modern-model adaptation.

**Report:** produce `reports/REPLICATION_REPORT.md` showing published claims, your observed results, exact-vs-adapted differences, examples with source IDs, uncertainties, failures and one-command reproduction instructions. Include `reports/THREATS_TO_VALIDITY.md` (prompt sensitivity, annotation ambiguity, dataset contamination, generation changes and lack of access to 2023 model snapshots).

**Done:** `doctor`, `smoke`, `archived`, `new-model`, `report`, and `pytest` work; every graph is generated from saved real artifacts; the README clearly distinguishes archived re-analysis from fresh model inference.

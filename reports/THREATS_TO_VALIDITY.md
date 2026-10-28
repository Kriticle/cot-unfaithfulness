# Threats to validity

This covers both parts of the work: **(A) re-analysing the authors' archived 2023 outputs**, and
**(B) a downscaled re-run on a modern open-weight model (Qwen2.5-7B-Instruct, 4-bit MLX)**.

## 1. No access to the 2023 model snapshots
`text-davinci-003` and `claude-v1.0` have been retired, so nothing in this repo re-runs them. Part A
recomputes statistics from the authors' saved predictions. That checks the *analysis*, not the
*generation*: if the archived files were wrong, Part A would reproduce the same errors. Part B
uses a different model family, size, training recipe, quantization and decoding stack. Its results
test whether the **phenomenon** generalizes. They are not a replication of the GPT-3.5 or Claude 1.0
numbers, and the two should never be compared as if they measured the same thing.

## 2. Generation-setting changes (Part B)
* **Chat template.** The authors' Human/Assistant text becomes Qwen's ChatML. The user turn and the
  assistant prefill (`Let's think step by step:` / `The best answer is: (`) are byte-identical to
  upstream (a test checks this). The Qwen template adds a default system prompt ("You are Qwen…").
* **Decoding.** We sample at temperature 0.7, the paper's GPT-3.5 setting; Claude 1.0 used 1.0.
  There is one sample per condition, as in the paper. Sampling noise therefore adds to the measured
  biased-vs-unbiased changes, though the unbiased-context "incorrect → correct" rates bound it.
  Batched sampling is seeded per batch, so token-level reruns are deterministic only for an
  identical batch composition and the same mlx/Metal versions.
* **4-bit quantization** can change behaviour relative to the bf16 model.
* **Batch size varied across resumed sessions:** 16, then 24, then 12, with an automatic halving to 6 after a
  Metal out-of-memory error on the longest Always-A prompts. Prompts, temperature and token limits were constant.
  Each record stores its seed and batch size. Batching changes which random stream a sequence gets, but not its
  distribution.
* **max_tokens** follows the upstream rule but counts with the Qwen tokenizer instead of GPT-2.
  Generations cut off at the limit are flagged (`finish_reason=length`). If they lack an answer
  trigger, they are excluded.

## 3. Downscaling and sample size (Part B)
We use 40 seeded items per task (520 of 3,299), and the same items in every setting. Only the
bias-contradicting subset enters the headline accuracy: 334, 331 and 309 items for Suggested Answer zero-shot,
Suggested Answer few-shot and Always A. CIs are therefore about 2× wider than the
paper's. Per-task estimates (≤40 items, ~25 inconsistent) are noisy and should be read only as
direction.

## 4. Prompt sensitivity
Effects of this kind depend strongly on wording, on where the suggestion is placed, and on the chat
format. We kept the authors' exact strings. We did not test alternative phrasings, so the size of
any effect applies only to these prompts.

## 5. Dataset contamination
BBH/BIG-Bench items carry a canary string. Data dated after 2023 (Qwen2.5 was released Sep 2024)
may still include BBH items, the authors' public repo, or this paper's examples. Contamination
mostly raises clean-context accuracy. It could also make a model more robust to the suggestion,
because it "knows" the answer. Both would shrink the measured effect, not inflate it.

## 6. Answer parsing and exclusions
* Part B uses a stricter parser than upstream. Several answer triggers naming different letters
  count as `ambiguous` and are excluded, where upstream took the last one. Exclusion rates per
  condition are in `tables/parse_status.csv` and `tables/exclusions.csv`.
* Exclusion is paired: an item with any failed generation is dropped from every one of its
  conditions. Models that fail to follow the format differently under bias could still bias the
  kept sample.
* For the archived data, the authors' code drops only CoT failures (a stale loop variable; see
  `docs/methods_mapping.md`). We reproduce that rule exactly. On these data the strict rule gives
  the same numbers.

## 7. Annotation ambiguity and annotator identity
* "Acknowledges the bias" has fuzzy edges. Qwen often addresses the user directly ("you think it
  might be A, but…"). That mentions the suggestion without conceding influence. We label it
  `explicit_ack` (the feature is mentioned), so our acknowledgement rates are **upper bounds**
  on verbalized influence.
* The regex pre-screen has unknown recall. On the 60 reviewed flips it agreed with the agent's labels 98.3% of the
  time, but κ is degenerate because almost every label is `no_ack`.
* "Explanation supports the final answer" is a single-reviewer judgement. It is the noisiest label, especially for
  logic tasks with multi-step errors.
* **No human raters were available.** The pre-specified review sample was read and labelled by the
  AI coding agent that built this repo (`annotator_kind: agent`). That is not human review, and the
  agent shares the risk of being lenient or strict in the same way the heuristic is. There is only
  one reviewer, so no inter-rater agreement can be reported. Agent-vs-heuristic agreement is
  reported, but it is not inter-rater reliability. A blank CSV template for two human raters is
  saved in the run directory.
* No acknowledgement plus a changed answer shows that the answer is sensitive to the feature and
  that the explanation did not disclose it. It does **not** show intent, deception, or what the
  model's internal computation was. Visible CoT is a prompted text output, not hidden reasoning.

## 8. Re-analysis-specific caveats (Part A)
* Four archived Always-A files are truncated (e.g. date_understanding has 217/300 rows). The paper's
  Table 12 counts the missing rows as "failed", and so do we. No missing generation was replaced.
* Table 10 "% explained by bias" for Always A matches the published numbers only when restricted to
  bias-contradicting items, even though the caption says all items. The Suggested-Answer cells match
  the all-items reading. We report both.
* The causal_judgment few-shot file has 7 demonstrations, but Table 12 says 8.
* The paper's Δ values are differences of rounded accuracies, so ours differ by ≤0.1 pp.
* Manual-label statistics (Table 7, App. C) are recomputed from the authors' label CSVs. We did not
  re-annotate them. The BBQ "strongly biased pairs" share comes out at 22.9% vs 21% published, and
  "supports prediction" at 85.4% vs 86% published (the paper's figure appears to be 100 − 9 − 5).

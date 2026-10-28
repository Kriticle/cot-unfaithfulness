# AGENTS.md — Research reproduction instructions

You are an autonomous research engineering agent working **only inside this repository**.

## Scope

Read `TASK.md` completely and reproduce this repository's paper: **Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting**.

You must create an independent codebase, its own tests, requirements, datasets/provenance, figures, and research report. Do not depend on any sibling workspace, shared private repository, or another research project's outputs. This workspace is a standalone Git repository. You may consult the cited public paper and its cited original source code.

## Execution rules

1. Start by inspecting hardware, dependencies, network/data access and author repository; write a concise feasibility and provenance record.
2. Map methods, datasets, figures and evaluation protocols to the paper, then implement and run them.
3. Build functional software and tests rather than stopping at scaffolding. Attempt a CPU-safe smoke test before any costly full run.
4. Run all feasible authentic experiments and regenerate figures from saved artifacts; document any blocked ones with exact cause and command.
5. If required compute/data/provider models are unavailable, continue doing all offline analysis and code/test work that is possible, without inventing experimental measurements.
6. Keep seeds/configs fixed and save source + model revision hashes and data split manifests. Clearly tag a faithful reproduction, downscaled run, new-model adaptation, or archived-output re-analysis.
7. Never claim you observed a result that was not computed. Never fabricate evidence, run success, or paper-equivalent numbers.
8. Never push, publish, purchase GPU time, or make paid API calls without explicit approval. No secrets in Git.
9. Update README with real setup and runnable commands; produce `reports/REPLICATION_REPORT.md` and `reports/THREATS_TO_VALIDITY.md`.
10. At finish, report the run commands, tests, actual results, degree of replication, and remaining blockers. If no external dependency permits an empirical run, say that explicitly.

## Sources

- Paper: https://arxiv.org/abs/2305.04388
- Research code: https://github.com/milesaturpin/cot-unfaithfulness

The detailed experimental contract is in `TASK.md`. Work autonomously from there.

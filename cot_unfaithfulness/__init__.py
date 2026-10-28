"""Independent reproduction of Turpin et al. (NeurIPS 2023).

Run modes:
  * ``archived``  - exact re-analysis of the authors' 2023 archived outputs, no inference.
  * ``new_model`` - the original prompts re-executed on a locally available open-weight
                    model; always labelled a *modern-model adaptation*.
"""

__version__ = "0.1.0"

UPSTREAM_REPO = "https://github.com/milesaturpin/cot-unfaithfulness"
UPSTREAM_SHA = "df099452736946533f59498a90c23be3f09631c4"
PAPER_URL = "https://arxiv.org/abs/2305.04388v2"

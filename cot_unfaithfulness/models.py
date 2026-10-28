"""Model adapters.

* ``MLXModel``   local open-weight chat model on Apple Silicon via ``mlx-lm`` (the only
                 backend used for reported new-model results).
* ``FakeModel``  deterministic stand-in used by tests and the pipeline self-check; its
                 outputs are tagged ``synthetic`` and are never reported as results.

No adapter talks to a paid or remote endpoint.  ``forbid_inference()`` makes any attempt
to construct or call a model raise, and is engaged for the ``archived`` run mode.
"""
from __future__ import annotations

import contextlib
import hashlib
import platform
import re
from dataclasses import dataclass

from .prompts import Prompt

_FORBIDDEN = False
CALL_COUNT = 0


class InferenceForbidden(RuntimeError):
    pass


@contextlib.contextmanager
def forbid_inference():
    global _FORBIDDEN
    prev, _FORBIDDEN = _FORBIDDEN, True
    try:
        yield
    finally:
        _FORBIDDEN = prev


def _guard():
    global CALL_COUNT
    if _FORBIDDEN:
        raise InferenceForbidden("model inference is forbidden in this run mode")
    CALL_COUNT += 1


@dataclass
class GenResult:
    text: str
    n_prompt_tokens: int
    n_gen_tokens: int
    finish_reason: str


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class FakeModel:
    """Sycophantic toy: follows a suggestion if present, else answers (A) after an all-A prefix,
    else answers (B).  Exists only to exercise the pipeline offline."""

    model_id = "fake/sycophant"
    revision = "synthetic"
    backend = "fake"

    def __init__(self, **_):
        _guard()

    def info(self) -> dict:
        return {"backend": self.backend, "model_id": self.model_id, "revision": self.revision}

    def generate_batch(self, prompts: list[Prompt], max_tokens: int, temperature: float, seed: int) -> list[GenResult]:
        _guard()
        out = []
        for p in prompts:
            if p.suggested_idx is not None:
                letter = chr(65 + p.suggested_idx)
            elif p.prefix and set(re.findall(r"best answer is: \(([A-Z])\)", p.prefix)) == {"A"}:
                letter = "A"
            else:
                letter = "B"
            if p.cot:
                text = f" Considering each option carefully, the evidence points to ({letter}).\nThe best answer is: ({letter})."
            else:
                text = f"{letter})"
            out.append(GenResult(text, len(p.user) // 4, len(text) // 4, "stop"))
        return out


class MLXModel:
    backend = "mlx"

    def __init__(self, model_id: str, revision: str, system_prompt: str | None = None, batch_size: int = 16,
                 min_cached_prefix: int = 256, **_):
        _guard()
        from mlx_lm import load  # imported lazily: archived mode never imports mlx

        self.model_id, self.revision = model_id, revision
        self.model, self.tokenizer = load(model_id, revision=revision)
        self.system_prompt = system_prompt
        self.batch_size = batch_size
        self.min_cached_prefix = min_cached_prefix

    def info(self) -> dict:
        import mlx.core as mx
        import mlx_lm

        return {"backend": self.backend, "model_id": self.model_id, "revision": self.revision,
                "mlx_lm_version": mlx_lm.__version__, "mlx_version": getattr(mx, "__version__", "?"),
                "device": str(mx.default_device()), "machine": platform.machine(),
                "system_prompt": self.system_prompt if self.system_prompt is not None else "<template default>"}

    def render(self, p: Prompt) -> str:
        msgs = ([{"role": "system", "content": self.system_prompt}] if self.system_prompt else []) + \
               [{"role": "user", "content": p.user}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + p.assistant_prefill

    def generate_batch(self, prompts: list[Prompt], max_tokens: int, temperature: float, seed: int) -> list[GenResult]:
        _guard()
        import mlx.core as mx
        from mlx_lm import batch_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        toks = [self.tokenizer.encode(self.render(p), add_special_tokens=False) for p in prompts]
        # Shared few-shot prefix -> compute its KV cache once and reuse it for the batch.
        common = min(len(t) for t in toks) - 1
        for t in toks[1:]:
            k = 0
            while k < common and t[k] == toks[0][k]:
                k += 1
            common = k
        caches = None
        if common >= self.min_cached_prefix:
            cache = make_prompt_cache(self.model)
            prefix = mx.array(toks[0][:common])[None]
            for s in range(0, common, 2048):
                self.model(prefix[:, s:s + 2048], cache=cache)
                mx.eval([c.state for c in cache])
            caches = [cache] * len(toks)
            inputs = [t[common:] for t in toks]
        else:
            inputs = toks
        # Long shared prefixes (Answer-is-Always-A, ~4k tokens) can exhaust Metal memory when many
        # sequences are prefilled together; on OOM retry the same batch with half the batch size.
        bs = self.batch_size
        while True:
            try:
                mx.random.seed(seed)
                resp = batch_generate(self.model, self.tokenizer, inputs, prompt_caches=caches,
                                      max_tokens=max_tokens, sampler=make_sampler(temp=temperature),
                                      completion_batch_size=bs, prefill_batch_size=min(bs, 8),
                                      return_token_ids=True)
                break
            except RuntimeError as e:
                if "Insufficient Memory" not in str(e) or bs == 1:
                    raise
                mx.clear_cache()
                bs = max(1, bs // 2)
                print(f"[mlx] Metal OOM; retrying batch with completion_batch_size={bs}", flush=True)
        self.last_batch_size = bs
        out = []
        for t, text, ids in zip(toks, resp.texts, getattr(resp, "token_ids", None) or [None] * len(toks)):
            n_gen = len(ids) if ids is not None else len(self.tokenizer.encode(text))
            out.append(GenResult(text, len(t), n_gen, "length" if n_gen >= max_tokens else "stop"))
        return out


def get_model(cfg: dict):
    _guard()
    backend = cfg.get("backend", "mlx")
    if backend == "fake":
        return FakeModel()
    if backend == "mlx":
        return MLXModel(cfg["model_id"], cfg["revision"], system_prompt=cfg.get("system_prompt"),
                        batch_size=cfg.get("batch_size", 16))
    raise ValueError(f"unknown backend {backend!r} (no remote/paid backends are implemented)")

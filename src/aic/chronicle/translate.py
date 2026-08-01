"""Caption translation (English -> Vietnamese) behind a registry.

The NLLB path uses transformers seq2seq generation with a forced target
language token — ``tokenizer.src_lang`` plus
``generate(forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang))`` —
which is the current API on transformers 4.57 (``lang_code_to_id`` was
removed in 4.46). The model loads lazily; on the CPU dev machine translation
is exercised through the fake backend in tests and for real on the GPU box.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from aic.config import TranslateConfig
from aic.models_cache import hf_cache_dir

logger = logging.getLogger(__name__)


class Translator(Protocol):
    def translate(self, texts: list[str]) -> list[str]:
        """Translate texts src->tgt, preserving order and length."""
        ...


class NLLBTranslator:
    """facebook/nllb-200 seq2seq translation via transformers."""

    def __init__(self, cfg: TranslateConfig, models_dir: Path) -> None:
        self._cfg = cfg
        self._cache_dir = hf_cache_dir(models_dir)
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            device = self._cfg.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._cfg.model_id,
                cache_dir=str(self._cache_dir),
                src_lang=self._cfg.src_lang,
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self._cfg.model_id, cache_dir=str(self._cache_dir)
            ).to(device)
            self._model.eval()
            self._device = device
            self._torch = torch
            forced = self._tokenizer.convert_tokens_to_ids(self._cfg.tgt_lang)
            if forced == self._tokenizer.unk_token_id:
                raise ValueError(
                    f"target language token {self._cfg.tgt_lang!r} is unknown "
                    f"to the {self._cfg.model_id} tokenizer"
                )
            self._forced_bos_token_id = forced
            logger.info("loaded %s on %s", self._cfg.model_id, device)
        return self._model

    def translate(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        model = self._ensure_loaded()
        outputs: list[str] = []
        for start in range(0, len(texts), self._cfg.batch_size):
            batch = texts[start : start + self._cfg.batch_size]
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(self._device)
            with self._torch.no_grad():
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=self._forced_bos_token_id,
                    max_new_tokens=self._cfg.max_new_tokens,
                )
            outputs.extend(
                self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return outputs


def build_translator(cfg: TranslateConfig, models_dir: Path) -> Translator:
    """Instantiate the translator (NLLB is the only family so far)."""
    return NLLBTranslator(cfg, models_dir)

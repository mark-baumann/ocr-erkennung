"""vLLM backend — the fast path for multi-page documents.

Adopted from ``DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py``. DeepSeek-OCR is
supported natively in vLLM from v0.11.1, which is the path used here: the
no-repeat-ngram logits processor ships with vLLM as
``NGramPerReqLogitsProcessor`` and is configured per request via
``SamplingParams.extra_args``.

The real win over the transformers backend is :meth:`infer_batch` — vLLM
schedules all pages of a PDF in one ``generate`` call instead of looping.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

from .base import BackendInfo, EngineError, EngineUnavailable, OcrEngine, OcrRequest, OcrResult, StreamCallback
from ..config import TABLE_TOKEN_WHITELIST, Settings


class VllmEngine(OcrEngine):
    key = "vllm"
    label = "vLLM (Batch)"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._llm = None
        self._logits_processor_cls = None

    # -- availability ------------------------------------------------------

    @classmethod
    def probe(cls, settings: Settings) -> BackendInfo:
        try:
            import vllm
        except ImportError:
            return BackendInfo(cls.key, cls.label, False, "vLLM nicht installiert.")

        try:
            import torch

            if not torch.cuda.is_available():
                return BackendInfo(cls.key, cls.label, False, "Keine CUDA-GPU sichtbar.")
        except ImportError:
            return BackendInfo(cls.key, cls.label, False, "PyTorch nicht installiert.")

        version = getattr(vllm, "__version__", "unbekannt")
        if not _has_native_support():
            return BackendInfo(
                cls.key,
                cls.label,
                False,
                f"vLLM {version} kennt DeepSeek-OCR noch nicht — benötigt >= 0.11.1.",
            )
        return BackendInfo(cls.key, cls.label, True, f"vLLM {version} · natives DeepSeek-OCR-Modell")

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        if self._llm is not None:
            return

        try:
            from vllm import LLM
        except ImportError as exc:
            raise EngineUnavailable("vLLM ist nicht installiert.") from exc

        self._logits_processor_cls = _load_logits_processor()
        if self._logits_processor_cls is None:
            raise EngineUnavailable(
                "Diese vLLM-Version stellt NGramPerReqLogitsProcessor nicht bereit (>= 0.11.1 nötig)."
            )

        try:
            self._llm = LLM(
                model=self.settings.model_path,
                enable_prefix_caching=False,
                logits_processors=[self._logits_processor_cls],
                trust_remote_code=True,
                max_model_len=self.settings.max_model_len,
                gpu_memory_utilization=self.settings.gpu_memory_utilization,
                max_num_seqs=self.settings.max_concurrency,
                limit_mm_per_prompt={"image": 1},
            )
        except Exception as exc:
            raise EngineError(f"vLLM-Engine konnte nicht starten: {exc}") from exc

    def unload(self) -> None:
        self._llm = None

    # -- inference ---------------------------------------------------------

    def _sampling_params(self, request: OcrRequest):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_new_tokens,
            skip_special_tokens=False,
            extra_args={
                "ngram_size": self.settings.ngram_size,
                "window_size": self.settings.ngram_window,
                "whitelist_token_ids": set(TABLE_TOKEN_WHITELIST),
            },
        )

    def infer(self, request: OcrRequest, on_token: Optional[StreamCallback] = None) -> OcrResult:
        results = self.infer_batch([request])
        result = results[0]
        if on_token is not None and result.text:
            on_token(result.text)
        return result

    def infer_batch(
        self,
        requests: List[OcrRequest],
        on_page: Optional[Callable[[int, OcrResult], None]] = None,
    ) -> List[OcrResult]:
        if not requests:
            return []

        self.load()
        if self._llm is None:  # pragma: no cover - defensive
            raise EngineError("vLLM-Engine ist nicht geladen.")

        inputs = [
            {"prompt": r.prompt, "multi_modal_data": {"image": r.image.convert("RGB")}} for r in requests
        ]

        start = time.perf_counter()
        try:
            outputs = self._llm.generate(inputs, self._sampling_params(requests[0]))
        except Exception as exc:
            raise EngineError(f"vLLM-Inferenz fehlgeschlagen: {exc}") from exc
        elapsed = time.perf_counter() - start

        # vLLM returns one RequestOutput per input, in input order.
        per_page = elapsed / max(1, len(requests))
        results: List[OcrResult] = []
        for index, output in enumerate(outputs):
            completion = output.outputs[0] if output.outputs else None
            text = completion.text if completion is not None else ""
            token_count = len(completion.token_ids) if completion is not None else None
            result = OcrResult(
                text=text,
                latency_s=per_page,
                backend=self.key,
                generated_tokens=token_count,
                metadata={
                    "mode": requests[index].mode.key,
                    "batch_size": len(requests),
                    "batch_latency_s": elapsed,
                },
            )
            results.append(result)
            if on_page is not None:
                on_page(index, result)

        return results


def _has_native_support() -> bool:
    try:
        from vllm.model_executor.models import registry  # noqa: F401
    except Exception:
        return False
    return _load_logits_processor() is not None


def _load_logits_processor():
    """Locate ``NGramPerReqLogitsProcessor`` across vLLM layouts."""
    candidates = (
        "vllm.model_executor.models.deepseek_ocr",
        "vllm.v1.sample.logits_processor",
    )
    for module_path in candidates:
        try:
            module = __import__(module_path, fromlist=["NGramPerReqLogitsProcessor"])
        except Exception:
            continue
        processor = getattr(module, "NGramPerReqLogitsProcessor", None)
        if processor is not None:
            return processor
    return None

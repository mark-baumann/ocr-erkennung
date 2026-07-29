"""HuggingFace ``transformers`` backend.

Wraps the upstream one-liner from ``DeepSeek-OCR-hf/run_dpsk_ocr.py``::

    model.infer(tokenizer, prompt=..., image_file=..., base_size=..., ...)

Changes made here:

* the model is loaded once and reused instead of per script run;
* ``infer`` writes to a temp dir and reads the result back, because the upstream
  ``model.infer`` prints to stdout and only returns text in some code paths —
  we capture stdout as the authoritative source and fall back to the files;
* flash-attention is optional: we fall back to eager attention rather than
  crashing on machines where ``flash_attn`` is not built.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from .base import BackendInfo, EngineError, EngineUnavailable, OcrEngine, OcrRequest, OcrResult, StreamCallback
from ..config import Settings


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise EngineUnavailable("PyTorch ist nicht installiert.") from exc


def _has_flash_attention() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


class TransformersEngine(OcrEngine):
    key = "transformers"
    label = "Transformers (HF)"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model = None
        self._tokenizer = None
        self._attn_impl = "eager"

    # -- availability ------------------------------------------------------

    @classmethod
    def probe(cls, settings: Settings) -> BackendInfo:
        try:
            import torch
        except ImportError:
            return BackendInfo(cls.key, cls.label, False, "PyTorch nicht installiert.")

        try:
            import transformers  # noqa: F401
        except ImportError:
            return BackendInfo(cls.key, cls.label, False, "transformers nicht installiert.")

        if not torch.cuda.is_available():
            return BackendInfo(
                cls.key,
                cls.label,
                False,
                "Keine CUDA-GPU sichtbar. DeepSeek-OCR benötigt eine NVIDIA-GPU.",
            )

        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        attn = "flash_attention_2" if _has_flash_attention() else "eager"
        return BackendInfo(
            cls.key,
            cls.label,
            True,
            f"{name} · {total_gb:.0f} GB · attention={attn}",
        )

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return

        torch = _torch()
        if not torch.cuda.is_available():
            raise EngineUnavailable("Keine CUDA-GPU verfügbar.")

        from transformers import AutoModel, AutoTokenizer

        model_path = self.settings.model_path
        self._attn_impl = "flash_attention_2" if _has_flash_attention() else "eager"

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        try:
            model = AutoModel.from_pretrained(
                model_path,
                _attn_implementation=self._attn_impl,
                trust_remote_code=True,
                use_safetensors=True,
            )
        except Exception as exc:
            if self._attn_impl == "eager":
                raise EngineError(f"Modell konnte nicht geladen werden: {exc}") from exc
            # flash-attn present but unusable (wrong CUDA/torch build) – retry eager.
            self._attn_impl = "eager"
            model = AutoModel.from_pretrained(
                model_path,
                _attn_implementation="eager",
                trust_remote_code=True,
                use_safetensors=True,
            )

        self._model = model.eval().cuda().to(torch.bfloat16)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        with contextlib.suppress(Exception):
            _torch().cuda.empty_cache()

    # -- inference ---------------------------------------------------------

    def infer(self, request: OcrRequest, on_token: Optional[StreamCallback] = None) -> OcrResult:
        self.load()
        if self._model is None or self._tokenizer is None:  # pragma: no cover - defensive
            raise EngineError("Modell ist nicht geladen.")

        mode = request.mode
        start = time.perf_counter()

        with tempfile.TemporaryDirectory(prefix="dsocr-") as workdir:
            image_path = Path(workdir) / "page.png"
            request.image.convert("RGB").save(image_path)
            output_dir = Path(workdir) / "out"
            output_dir.mkdir(parents=True, exist_ok=True)

            # ``model.infer`` streams to stdout; capture it so the UI gets the text
            # even on the code paths where the return value is None.
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer):
                    returned = self._model.infer(
                        self._tokenizer,
                        prompt=request.prompt,
                        image_file=str(image_path),
                        output_path=str(output_dir),
                        base_size=mode.base_size,
                        image_size=mode.image_size,
                        crop_mode=mode.crop_mode,
                        save_results=True,
                        test_compress=False,
                    )
            except Exception as exc:
                raise EngineError(f"Inferenz fehlgeschlagen: {exc}") from exc

            text = _first_non_empty(
                returned if isinstance(returned, str) else None,
                _read_if_exists(output_dir / "result_ori.mmd"),
                _read_if_exists(output_dir / "result.mmd"),
                _strip_progress(buffer.getvalue()),
            )

        latency = time.perf_counter() - start
        if on_token is not None and text:
            on_token(text)

        return OcrResult(
            text=text,
            latency_s=latency,
            backend=self.key,
            metadata={
                "mode": mode.key,
                "attn_implementation": self._attn_impl,
                "model_path": self.settings.model_path,
            },
        )


def _read_if_exists(path: Path) -> Optional[str]:
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None
    return None


def _strip_progress(raw: str) -> str:
    """Drop tqdm/status lines that ``model.infer`` prints around the answer."""
    keep = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            keep.append(line)
            continue
        if stripped.startswith(("=" * 5, "PATCHES", "BASE:", "image:", "valid image tokens")):
            continue
        if "it/s]" in stripped or "%|" in stripped:
            continue
        keep.append(line)
    return "\n".join(keep).strip()


def _first_non_empty(*candidates: Optional[str]) -> str:
    for candidate in candidates:
        if candidate and candidate.strip():
            return candidate.strip()
    return ""


# Keep CUDA device selection consistent with upstream when the caller pins one.
if "CUDA_VISIBLE_DEVICES" not in os.environ and os.environ.get("DSOCR_CUDA_DEVICE"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["DSOCR_CUDA_DEVICE"]

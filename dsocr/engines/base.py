"""Engine interface shared by the transformers, vLLM and demo backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from PIL import Image

from ..config import ResolutionMode, Settings

#: Called with each newly generated chunk of text so the UI can stream.
StreamCallback = Callable[[str], None]


class EngineUnavailable(RuntimeError):
    """The backend cannot run here (missing dependency, no GPU, no weights)."""


class EngineError(RuntimeError):
    """Inference started but failed."""


@dataclass
class OcrRequest:
    """Everything an engine needs for one page."""

    image: Image.Image
    prompt: str
    mode: ResolutionMode
    max_new_tokens: int = 8192
    min_crops: int = 2
    max_crops: int = 6
    #: Deterministic by default — OCR is not a creative task.
    temperature: float = 0.0


@dataclass
class OcrResult:
    """Raw model output plus timing telemetry."""

    text: str
    latency_s: float
    backend: str
    #: Number of tokens the backend reported generating, when known.
    generated_tokens: Optional[int] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> Optional[float]:
        if not self.generated_tokens or self.latency_s <= 0:
            return None
        return self.generated_tokens / self.latency_s


@dataclass(frozen=True)
class BackendInfo:
    """What the UI shows in the backend picker."""

    key: str
    label: str
    available: bool
    detail: str
    #: ``False`` for the demo backend — results are synthetic, say so loudly.
    is_real: bool = True


class OcrEngine(ABC):
    """A DeepSeek-OCR inference backend."""

    key: str = "base"
    label: str = "Base"
    is_real: bool = True

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # -- availability ------------------------------------------------------

    @classmethod
    @abstractmethod
    def probe(cls, settings: Settings) -> BackendInfo:
        """Report — without loading weights — whether this backend can run."""

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Load weights. Called lazily on first use; safe to call repeatedly."""

    def unload(self) -> None:
        """Release GPU memory."""

    # -- inference ---------------------------------------------------------

    @abstractmethod
    def infer(self, request: OcrRequest, on_token: Optional[StreamCallback] = None) -> OcrResult:
        """Run one page and return the raw model output."""

    def infer_batch(
        self,
        requests: List[OcrRequest],
        on_page: Optional[Callable[[int, OcrResult], None]] = None,
    ) -> List[OcrResult]:
        """Run several pages. Backends with real batching override this."""
        results: List[OcrResult] = []
        for index, request in enumerate(requests):
            result = self.infer(request)
            results.append(result)
            if on_page is not None:
                on_page(index, result)
        return results

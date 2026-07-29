"""Backend registry and auto-selection."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from .base import (
    BackendInfo,
    EngineError,
    EngineUnavailable,
    OcrEngine,
    OcrRequest,
    OcrResult,
    StreamCallback,
)
from .demo_engine import DemoEngine
from .transformers_engine import TransformersEngine
from .vllm_engine import VllmEngine
from ..config import SETTINGS, Settings

#: Preference order for ``backend="auto"``: batching first, then single-page,
#: then the CPU fallback.
ENGINES: Dict[str, Type[OcrEngine]] = {
    VllmEngine.key: VllmEngine,
    TransformersEngine.key: TransformersEngine,
    DemoEngine.key: DemoEngine,
}

AUTO_ORDER: List[str] = [VllmEngine.key, TransformersEngine.key, DemoEngine.key]


def probe_all(settings: Optional[Settings] = None) -> List[BackendInfo]:
    """Report availability of every backend without loading weights."""
    settings = settings or SETTINGS
    infos: List[BackendInfo] = []
    for key in AUTO_ORDER:
        try:
            infos.append(ENGINES[key].probe(settings))
        except Exception as exc:  # a broken probe must never kill the UI
            cls = ENGINES[key]
            infos.append(BackendInfo(key, cls.label, False, f"Probe fehlgeschlagen: {exc}", cls.is_real))
    return infos


def resolve_backend(preference: str, settings: Optional[Settings] = None) -> str:
    """Turn a preference (``"auto"`` or a key) into a concrete backend key."""
    settings = settings or SETTINGS
    infos = {info.key: info for info in probe_all(settings)}

    if preference != "auto":
        info = infos.get(preference)
        if info is None:
            raise EngineUnavailable(f"Unbekanntes Backend {preference!r}.")
        if not info.available:
            raise EngineUnavailable(f"Backend '{preference}' ist hier nicht verfügbar: {info.detail}")
        return preference

    for key in AUTO_ORDER:
        info = infos.get(key)
        if info is not None and info.available:
            return key

    raise EngineUnavailable("Kein Backend verfügbar — auch der Demo-Modus ist deaktiviert.")


def create_engine(preference: str = "auto", settings: Optional[Settings] = None) -> OcrEngine:
    """Instantiate the best available backend. Weights load lazily on first use."""
    settings = settings or SETTINGS
    return ENGINES[resolve_backend(preference, settings)](settings)


__all__ = [
    "AUTO_ORDER",
    "BackendInfo",
    "DemoEngine",
    "ENGINES",
    "EngineError",
    "EngineUnavailable",
    "OcrEngine",
    "OcrRequest",
    "OcrResult",
    "StreamCallback",
    "TransformersEngine",
    "VllmEngine",
    "create_engine",
    "probe_all",
    "resolve_backend",
]

"""Central configuration: resolution modes, prompt library and runtime settings.

Upstream DeepSeek-OCR keeps these as module level constants in
``DeepSeek-OCR-vllm/config.py`` which has to be edited by hand before every run.
Here they become first class objects so the Streamlit UI can offer them as
choices and the engines can consume them uniformly.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Vision encoder geometry (from the upstream DeepseekOCRProcessor)
# ---------------------------------------------------------------------------

PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4

#: A view of ``size`` pixels collapses to ``TOKEN_GRID(size)`` queries per axis.
def token_grid(size: int) -> int:
    """Number of latent queries along one axis for a ``size`` x ``size`` view."""
    return math.ceil((size / PATCH_SIZE) / DOWNSAMPLE_RATIO)


# ---------------------------------------------------------------------------
# Resolution modes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionMode:
    """One of the five resolution presets described in the DeepSeek-OCR paper.

    ``base_size`` drives the global (thumbnail) view, ``image_size`` drives the
    local tiles. ``crop_mode`` enables the dynamic tiling ("Gundam") path.
    """

    key: str
    label: str
    base_size: int
    image_size: int
    crop_mode: bool
    blurb: str

    @property
    def global_grid(self) -> int:
        return token_grid(self.base_size)

    @property
    def local_grid(self) -> int:
        return token_grid(self.image_size)

    @property
    def nominal_vision_tokens(self) -> int:
        """Vision tokens for a single global view (the number quoted in the paper)."""
        return self.global_grid**2


MODES: Dict[str, ResolutionMode] = {
    "tiny": ResolutionMode(
        key="tiny",
        label="Tiny · 512px · 64 Tokens",
        base_size=512,
        image_size=512,
        crop_mode=False,
        blurb="Schnellster Modus. Gut für saubere, textarme Bilder und Screenshots.",
    ),
    "small": ResolutionMode(
        key="small",
        label="Small · 640px · 100 Tokens",
        base_size=640,
        image_size=640,
        crop_mode=False,
        blurb="Guter Kompromiss für einspaltige Dokumente in normaler Schriftgröße.",
    ),
    "base": ResolutionMode(
        key="base",
        label="Base · 1024px · 256 Tokens",
        base_size=1024,
        image_size=1024,
        crop_mode=False,
        blurb="Standard für gescannte A4-Seiten mit moderater Informationsdichte.",
    ),
    "large": ResolutionMode(
        key="large",
        label="Large · 1280px · 400 Tokens",
        base_size=1280,
        image_size=1280,
        crop_mode=False,
        blurb="Für kleine Schrift und dichte Layouts ohne Tiling-Overhead.",
    ),
    "gundam": ResolutionMode(
        key="gundam",
        label="Gundam · 1024 + n×640 · dynamisch",
        base_size=1024,
        image_size=640,
        crop_mode=True,
        blurb="Dynamisches Tiling: globale Übersicht plus n lokale Kacheln. "
        "Beste Qualität für Zeitungen, Formulare und mehrspaltige Seiten.",
    ),
}

DEFAULT_MODE = "gundam"


def get_mode(key: str) -> ResolutionMode:
    try:
        return MODES[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unbekannter Resolution-Mode {key!r}. Erlaubt: {sorted(MODES)}") from exc


# ---------------------------------------------------------------------------
# Prompt library
# ---------------------------------------------------------------------------

IMAGE_TOKEN = "<image>"
GROUNDING_TOKEN = "<|grounding|>"


@dataclass(frozen=True)
class PromptPreset:
    """A named prompt from the upstream README, plus metadata for the UI."""

    key: str
    label: str
    template: str
    blurb: str
    #: ``True`` when the prompt asks the model to emit ``<|det|>`` bounding boxes.
    grounding: bool = False
    #: ``True`` when the template contains a ``{query}`` placeholder.
    needs_query: bool = False

    def render(self, query: str = "") -> str:
        text = self.template
        if self.needs_query:
            text = text.replace("{query}", query.strip())
        return text


PROMPTS: Dict[str, PromptPreset] = {
    "markdown": PromptPreset(
        key="markdown",
        label="Dokument → Markdown (mit Layout)",
        template=f"{IMAGE_TOKEN}\n{GROUNDING_TOKEN}Convert the document to markdown.",
        blurb="Volle Dokumentkonvertierung inklusive Layout-Boxen für jede Region.",
        grounding=True,
    ),
    "ocr_layout": PromptPreset(
        key="ocr_layout",
        label="OCR mit Layout-Boxen",
        template=f"{IMAGE_TOKEN}\n{GROUNDING_TOKEN}OCR this image.",
        blurb="Für Fotos, Schilder und Nicht-Dokumente – mit Positionsangaben.",
        grounding=True,
    ),
    "free_ocr": PromptPreset(
        key="free_ocr",
        label="Free OCR (nur Text)",
        template=f"{IMAGE_TOKEN}\nFree OCR.",
        blurb="Reiner Textauszug ohne Layout-Analyse. Am schnellsten.",
    ),
    "figure": PromptPreset(
        key="figure",
        label="Abbildung / Diagramm parsen",
        template=f"{IMAGE_TOKEN}\nParse the figure.",
        blurb="Extrahiert Datenpunkte aus Charts, Graphen und Geometriezeichnungen.",
    ),
    "describe": PromptPreset(
        key="describe",
        label="Bild beschreiben",
        template=f"{IMAGE_TOKEN}\nDescribe this image in detail.",
        blurb="Allgemeine Bildbeschreibung statt OCR.",
    ),
    "locate": PromptPreset(
        key="locate",
        label="Text lokalisieren (Grounding)",
        template=f"{IMAGE_TOKEN}\nLocate <|ref|>{{query}}<|/ref|> in the image.",
        blurb="Findet eine konkrete Textstelle und gibt ihre Bounding-Box zurück.",
        grounding=True,
        needs_query=True,
    ),
    "custom": PromptPreset(
        key="custom",
        label="Eigener Prompt",
        template="{query}",
        blurb="Freitext. Muss <image> enthalten, damit das Bild verarbeitet wird.",
        needs_query=True,
    ),
}

DEFAULT_PROMPT = "markdown"


# ---------------------------------------------------------------------------
# Runtime settings
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Everything that is normally an env var / CLI flag."""

    model_path: str = field(default_factory=lambda: os.environ.get("DSOCR_MODEL_PATH", "deepseek-ai/DeepSeek-OCR"))
    backend: str = field(default_factory=lambda: os.environ.get("DSOCR_BACKEND", "auto"))

    # Tiling bounds – upstream MIN_CROPS / MAX_CROPS.
    min_crops: int = field(default_factory=lambda: _env_int("DSOCR_MIN_CROPS", 2))
    max_crops: int = field(default_factory=lambda: _env_int("DSOCR_MAX_CROPS", 6))

    # Generation
    max_new_tokens: int = field(default_factory=lambda: _env_int("DSOCR_MAX_NEW_TOKENS", 8192))
    ngram_size: int = field(default_factory=lambda: _env_int("DSOCR_NGRAM_SIZE", 30))
    ngram_window: int = field(default_factory=lambda: _env_int("DSOCR_NGRAM_WINDOW", 90))

    # vLLM
    gpu_memory_utilization: float = field(default_factory=lambda: _env_float("DSOCR_GPU_MEM_UTIL", 0.85))
    max_model_len: int = field(default_factory=lambda: _env_int("DSOCR_MAX_MODEL_LEN", 8192))
    max_concurrency: int = field(default_factory=lambda: _env_int("DSOCR_MAX_CONCURRENCY", 32))

    # Documents
    pdf_dpi: int = field(default_factory=lambda: _env_int("DSOCR_PDF_DPI", 144))
    max_pages: int = field(default_factory=lambda: _env_int("DSOCR_MAX_PAGES", 100))
    max_upload_mb: int = field(default_factory=lambda: _env_int("DSOCR_MAX_UPLOAD_MB", 50))

    allow_demo_engine: bool = field(default_factory=lambda: _env_bool("DSOCR_ALLOW_DEMO", True))

    def crop_bounds(self) -> Tuple[int, int]:
        lo = max(1, self.min_crops)
        hi = max(lo, self.max_crops)
        return lo, hi


#: ``<td>`` / ``</td>`` – whitelisted so the no-repeat-ngram processor never
#: blocks legitimate table markup. Values come from the upstream tokenizer.
TABLE_TOKEN_WHITELIST: List[int] = [128821, 128822]


SETTINGS = Settings()

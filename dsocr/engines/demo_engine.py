"""CPU fallback so the app is usable — and testable — without a GPU.

DeepSeek-OCR needs CUDA. Rather than showing a dead UI on a laptop or in CI,
this backend produces output in exactly the same grounded format the real model
emits, so every downstream stage (grounding parser, overlay, markdown builder,
exports) runs against realistic input.

It works in two tiers:

* **Tesseract available** — real text recognition, with real word boxes merged
  into paragraph-level regions. Quality is nowhere near DeepSeek-OCR, but the
  text is genuine.
* **Nothing available** — layout-only: text blocks are located with a classic
  projection-profile analysis and returned with placeholder content.

Results are always flagged ``is_real = False``. Nothing here should ever be
mistaken for DeepSeek-OCR output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .base import BackendInfo, OcrEngine, OcrRequest, OcrResult, StreamCallback
from ..config import Settings
from ..postprocess.grounding import COORD_SCALE

DEMO_BANNER = (
    "> **Demo-Modus** — keine CUDA-GPU gefunden, daher läuft hier nicht DeepSeek-OCR, "
    "sondern ein CPU-Ersatz. Layout und Format entsprechen dem echten Modell, "
    "die Erkennungsqualität nicht.\n"
)


def _tesseract():
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return pytesseract
    except Exception:
        return None


@dataclass
class _Block:
    """A detected text block in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    text: str = ""

    def normalized(self, width: int, height: int) -> Tuple[int, int, int, int]:
        return (
            int(self.x1 / max(1, width) * COORD_SCALE),
            int(self.y1 / max(1, height) * COORD_SCALE),
            int(self.x2 / max(1, width) * COORD_SCALE),
            int(self.y2 / max(1, height) * COORD_SCALE),
        )

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class DemoEngine(OcrEngine):
    key = "demo"
    label = "Demo (CPU)"
    is_real = False

    @classmethod
    def probe(cls, settings: Settings) -> BackendInfo:
        if not settings.allow_demo_engine:
            return BackendInfo(cls.key, cls.label, False, "Per DSOCR_ALLOW_DEMO=0 deaktiviert.", is_real=False)
        engine = _tesseract()
        detail = (
            "Tesseract gefunden — echter Text, synthetisches Layout."
            if engine
            else "Nur Layout-Analyse (kein Tesseract installiert)."
        )
        return BackendInfo(cls.key, cls.label, True, detail, is_real=False)

    def infer(self, request: OcrRequest, on_token: Optional[StreamCallback] = None) -> OcrResult:
        start = time.perf_counter()
        image = request.image.convert("RGB")

        engine = _tesseract()
        if engine is not None:
            blocks = _blocks_from_tesseract(engine, image)
            source = "tesseract"
        else:
            blocks = _blocks_from_projection(image)
            source = "projection"

        text = _render_grounded_markdown(blocks, image.size, source=source, prompt=request.prompt)
        latency = time.perf_counter() - start

        if on_token is not None:
            on_token(text)

        return OcrResult(
            text=text,
            latency_s=latency,
            backend=self.key,
            metadata={"mode": request.mode.key, "demo_source": source, "blocks": len(blocks)},
        )


# ---------------------------------------------------------------------------
# Block detection
# ---------------------------------------------------------------------------


def _blocks_from_tesseract(pytesseract, image: Image.Image) -> List[_Block]:
    """Group Tesseract word boxes into paragraph-level blocks."""
    from pytesseract import Output

    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    groups: dict[tuple, List[int]] = {}

    for index, conf in enumerate(data["conf"]):
        try:
            confidence = float(conf)
        except (TypeError, ValueError):
            continue
        if confidence < 0 or not data["text"][index].strip():
            continue
        key = (data["block_num"][index], data["par_num"][index])
        groups.setdefault(key, []).append(index)

    blocks: List[_Block] = []
    for indices in groups.values():
        xs1 = [data["left"][i] for i in indices]
        ys1 = [data["top"][i] for i in indices]
        xs2 = [data["left"][i] + data["width"][i] for i in indices]
        ys2 = [data["top"][i] + data["height"][i] for i in indices]
        words = [data["text"][i].strip() for i in indices if data["text"][i].strip()]
        blocks.append(
            _Block(min(xs1), min(ys1), max(xs2), max(ys2), text=" ".join(words))
        )

    blocks.sort(key=lambda b: (b.y1, b.x1))
    return blocks


def _blocks_from_projection(image: Image.Image, *, max_blocks: int = 40) -> List[_Block]:
    """Locate text blocks with a horizontal projection profile.

    Rows whose ink density exceeds a global threshold are considered text; runs
    of such rows separated by whitespace become blocks. Crude but dependency
    free, and good enough to demonstrate the layout overlay.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.size == 0:
        return []

    height, width = gray.shape
    # Ink = darker than the page background (estimated as the 90th percentile).
    background = float(np.percentile(gray, 90))
    ink = (gray < background - 25).astype(np.float32)

    row_density = ink.mean(axis=1)
    threshold = max(0.005, float(row_density.mean()) * 0.5)
    active = row_density > threshold

    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    gap_tolerance = max(3, height // 150)
    gap = 0

    for y, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap > gap_tolerance:
                runs.append((start, y - gap))
                start = None
                gap = 0
    if start is not None:
        runs.append((start, height - 1))

    min_height = max(4, height // 200)
    blocks: List[_Block] = []
    for y1, y2 in runs:
        if y2 - y1 < min_height:
            continue
        band = ink[y1 : y2 + 1]
        col_density = band.mean(axis=0)
        columns = np.where(col_density > 0.01)[0]
        if columns.size == 0:
            continue
        blocks.append(_Block(int(columns[0]), int(y1), int(columns[-1]) + 1, int(y2) + 1))
        if len(blocks) >= max_blocks:
            break

    return blocks


def _classify(block: _Block, blocks: Sequence[_Block], page_height: int) -> str:
    """Heuristic label so the overlay shows a plausible mix of region types."""
    heights = [b.height for b in blocks if b.height > 0]
    median = float(np.median(heights)) if heights else 1.0

    if block.y2 < page_height * 0.08:
        return "header"
    if block.y1 > page_height * 0.94:
        return "footer"
    if block.height > median * 1.7:
        return "title"
    return "text"


def _render_grounded_markdown(
    blocks: Sequence[_Block],
    size: Tuple[int, int],
    *,
    source: str,
    prompt: str,
) -> str:
    """Emit output in the model's ``<|ref|>…<|det|>…`` format."""
    width, height = size
    wants_grounding = "<|grounding|>" in prompt

    parts: List[str] = [DEMO_BANNER]

    if not blocks:
        parts.append("_Es wurden keine Textregionen gefunden._")
        return "\n".join(parts)

    for block in blocks:
        label = _classify(block, blocks, height)
        body = block.text.strip()
        if not body:
            body = f"[{label}-Region · {block.x2 - block.x1}×{block.height}px · nur im Demo-Modus ohne Texterkennung]"
        if label == "title":
            body = f"# {body}"

        if wants_grounding:
            x1, y1, x2, y2 = block.normalized(width, height)
            parts.append(f"<|ref|>{label}<|/ref|><|det|>[[{x1}, {y1}, {x2}, {y2}]]<|/det|>{body}")
        else:
            parts.append(body)

    if source == "projection":
        parts.append(
            "\n_Kein Tesseract installiert — es wurden nur Layout-Regionen erkannt, kein Text gelesen._"
        )

    return "\n\n".join(parts)

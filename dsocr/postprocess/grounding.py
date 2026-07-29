"""Parse and render DeepSeek-OCR grounding output.

With ``<|grounding|>`` in the prompt the model interleaves layout annotations
into its markdown::

    <|ref|>title<|/ref|><|det|>[[12, 30, 940, 88]]<|/det|># Annual Report

Upstream ``run_dpsk_ocr_image.py`` regexes those out and immediately paints
rectangles. Here the parse step yields typed :class:`LayoutRegion` objects, so
the same result can drive an overlay, a JSON export, a region filter and a
figure-cropper without re-parsing.

Two hardening changes versus upstream: coordinates go through
``ast.literal_eval`` instead of ``eval``, and out-of-range boxes are clamped
rather than silently producing inverted rectangles.
"""

from __future__ import annotations

import ast
import colorsys
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

#: The model emits coordinates on a fixed 0..999 grid, independent of image size.
COORD_SCALE = 999

GROUNDING_PATTERN = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>",
    re.DOTALL,
)


@dataclass(frozen=True)
class BoundingBox:
    """A box in the model's normalised 0..999 space."""

    x1: int
    y1: int
    x2: int
    y2: int

    def clamped(self) -> "BoundingBox":
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        return BoundingBox(
            max(0, min(COORD_SCALE, x1)),
            max(0, min(COORD_SCALE, y1)),
            max(0, min(COORD_SCALE, x2)),
            max(0, min(COORD_SCALE, y2)),
        )

    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        box = self.clamped()
        return (
            int(box.x1 / COORD_SCALE * width),
            int(box.y1 / COORD_SCALE * height),
            int(box.x2 / COORD_SCALE * width),
            int(box.y2 / COORD_SCALE * height),
        )

    @property
    def area(self) -> float:
        box = self.clamped()
        return (box.x2 - box.x1) * (box.y2 - box.y1) / (COORD_SCALE**2)

    def as_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class LayoutRegion:
    """One ``<|ref|>…<|det|>…`` annotation."""

    label: str
    boxes: List[BoundingBox]
    #: The full matched substring, needed to strip it back out of the markdown.
    raw: str
    order: int = 0
    #: Text that followed this annotation in the model output, when we could
    #: attribute it (used for the region inspector).
    text: str = ""

    @property
    def is_figure(self) -> bool:
        return self.label.lower() in {"image", "figure"}

    @property
    def primary_box(self) -> Optional[BoundingBox]:
        return self.boxes[0] if self.boxes else None

    def to_dict(self, size: Optional[Tuple[int, int]] = None) -> Dict:
        payload: Dict = {
            "order": self.order,
            "label": self.label,
            "boxes_normalized": [b.as_list() for b in self.boxes],
        }
        if size is not None:
            width, height = size
            payload["boxes_pixels"] = [list(b.to_pixels(width, height)) for b in self.boxes]
        if self.text:
            payload["text"] = self.text
        return payload


def _parse_boxes(raw: str) -> List[BoundingBox]:
    """Parse the ``<|det|>`` payload, tolerating both nested and flat forms."""
    try:
        value = ast.literal_eval(raw.strip())
    except (ValueError, SyntaxError):
        return []

    if not isinstance(value, (list, tuple)) or not value:
        return []

    # Flat form: [x1, y1, x2, y2]
    if all(isinstance(v, (int, float)) for v in value):
        value = [value]

    boxes: List[BoundingBox] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            continue
        if not all(isinstance(v, (int, float)) for v in item):
            continue
        boxes.append(BoundingBox(*(int(round(v)) for v in item)).clamped())
    return boxes


def parse_grounding(text: str) -> List[LayoutRegion]:
    """Extract every layout annotation from a model response, in order."""
    regions: List[LayoutRegion] = []
    matches = list(GROUNDING_PATTERN.finditer(text))

    for order, match in enumerate(matches):
        boxes = _parse_boxes(match.group(2))
        if not boxes:
            continue
        # Everything up to the next annotation is this region's content.
        end = matches[order + 1].start() if order + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        regions.append(
            LayoutRegion(
                label=match.group(1).strip() or "text",
                boxes=boxes,
                raw=match.group(0),
                order=len(regions),
                text=body,
            )
        )
    return regions


def label_counts(regions: Sequence[LayoutRegion]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for region in regions:
        counts[region.label] = counts.get(region.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


#: Curated, well-separated colours for the labels the model actually emits.
#: All of them sit at a luminance that stays legible against white paper and
#: against a dark page scan.
LABEL_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "title": (214, 72, 15),  # deep orange
    "header": (134, 46, 156),  # purple
    "footer": (95, 99, 104),  # slate
    "page_number": (95, 99, 104),
    "text": (25, 113, 194),  # blue
    "paragraph": (25, 113, 194),
    "list": (12, 133, 153),  # teal
    "table": (47, 158, 68),  # green
    "image": (209, 63, 128),  # magenta
    "figure": (209, 63, 128),
    "caption": (176, 122, 12),  # ochre
    "formula": (99, 62, 199),  # indigo
    "equation": (99, 62, 199),
    "reference": (117, 87, 60),  # brown
}


def color_for_label(label: str) -> Tuple[int, int, int]:
    """Deterministic colour per label.

    Upstream picks a random colour per region, so the same document looks
    different on every run and the legend carries no meaning. Known labels get a
    fixed colour from :data:`LABEL_PALETTE`; anything unexpected is hashed into
    a hue band that the curated colours leave free.
    """
    key = label.lower().strip()
    if key in LABEL_PALETTE:
        return LABEL_PALETTE[key]

    digest = hashlib.sha1(key.encode("utf-8")).digest()
    # Restrict unknown labels to the 150°-210° (cyan) window, which the curated
    # palette does not use, so they never collide with a known region type.
    hue = (150 + (digest[0] / 255.0) * 60) / 360.0
    saturation = 0.55 + (digest[1] / 255.0) * 0.3
    value = 0.55 + (digest[2] / 255.0) * 0.2
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_layout_overlay(
    image: Image.Image,
    regions: Sequence[LayoutRegion],
    *,
    visible_labels: Optional[Iterable[str]] = None,
    show_labels: bool = True,
    fill_alpha: int = 38,
    line_width: Optional[int] = None,
) -> Image.Image:
    """Render layout boxes onto a copy of ``image``.

    ``visible_labels`` filters which region types are drawn — the UI wires this
    to a multiselect so a user can look at just the tables.
    """
    canvas = image.convert("RGB").copy()
    width, height = canvas.size
    if not regions:
        return canvas

    allowed = {label.lower() for label in visible_labels} if visible_labels is not None else None
    if line_width is None:
        line_width = max(2, round(min(width, height) / 400))

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shade = ImageDraw.Draw(overlay)
    stroke = ImageDraw.Draw(canvas)
    font = _load_font(max(12, round(min(width, height) / 60)))

    for region in regions:
        if allowed is not None and region.label.lower() not in allowed:
            continue
        color = color_for_label(region.label)
        for box in region.boxes:
            x1, y1, x2, y2 = box.to_pixels(width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            shade.rectangle([x1, y1, x2, y2], fill=color + (fill_alpha,))
            emphasis = line_width + 1 if region.label.lower() in {"title", "table"} else line_width
            stroke.rectangle([x1, y1, x2, y2], outline=color, width=emphasis)

            if not show_labels:
                continue
            caption = region.label
            text_box = stroke.textbbox((0, 0), caption, font=font)
            tw, th = text_box[2] - text_box[0], text_box[3] - text_box[1]
            pad = 3
            tx = x1
            ty = max(0, y1 - th - 2 * pad)
            stroke.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad], fill=color)
            stroke.text((tx + pad, ty + pad), caption, font=font, fill=(255, 255, 255))

    canvas.paste(overlay, (0, 0), overlay)
    return canvas


@dataclass
class CroppedFigure:
    """A figure region cut out of the source page."""

    index: int
    label: str
    image: Image.Image
    box_pixels: Tuple[int, int, int, int]
    filename: str = field(default="")

    def __post_init__(self) -> None:
        if not self.filename:
            self.filename = f"figure_{self.index:03d}.png"


def crop_figures(
    image: Image.Image,
    regions: Sequence[LayoutRegion],
    *,
    min_area: float = 0.0005,
    padding: int = 2,
) -> List[CroppedFigure]:
    """Cut out every ``image``/``figure`` region so it can be embedded in markdown."""
    width, height = image.size
    figures: List[CroppedFigure] = []

    for region in regions:
        if not region.is_figure:
            continue
        box = region.primary_box
        if box is None or box.area < min_area:
            continue
        x1, y1, x2, y2 = box.to_pixels(width, height)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        if x2 <= x1 or y2 <= y1:
            continue
        figures.append(
            CroppedFigure(
                index=len(figures),
                label=region.label,
                image=image.crop((x1, y1, x2, y2)),
                box_pixels=(x1, y1, x2, y2),
            )
        )
    return figures

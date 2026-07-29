"""Shared fixtures: synthetic pages and canned model output."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_page(width: int = 1240, height: int = 1754) -> Image.Image:
    """A synthetic A4-ish document page with a title and body text."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((90, 110), "Quartalsbericht 2026", font=_font(52), fill="black")

    y = 240
    body = _font(24)
    for block in range(4):
        for line in range(6):
            draw.text((90, y), f"Absatz {block + 1}, Zeile {line + 1} mit Fließtext.", font=body, fill="black")
            y += 38
        y += 46

    draw.rectangle([90, y + 20, width - 90, y + 260], outline="black", width=3)
    return image


@pytest.fixture
def page_image() -> Image.Image:
    return make_page()


@pytest.fixture
def small_image() -> Image.Image:
    return Image.new("RGB", (400, 300), "white")


@pytest.fixture
def grounded_output() -> str:
    """Model output in the exact shape DeepSeek-OCR emits with ``<|grounding|>``."""
    return (
        "<|ref|>title<|/ref|><|det|>[[40, 30, 950, 90]]<|/det|># Quartalsbericht 2026\n\n"
        "<|ref|>text<|/ref|><|det|>[[40, 120, 950, 300]]<|/det|>"
        "Der Umsatz stieg um 12 Prozent gegenüber dem Vorjahr.\n\n"
        "<|ref|>table<|/ref|><|det|>[[40, 320, 950, 520]]<|/det|>"
        "<table><tr><td>Region</td><td>Umsatz</td></tr>"
        "<tr><td>EMEA</td><td>4.2 Mio</td></tr>"
        "<tr><td>APAC</td><td>2.8 Mio</td></tr></table>\n\n"
        "<|ref|>image<|/ref|><|det|>[[100, 560, 880, 900]]<|/det|>\n\n"
        "<|ref|>footer<|/ref|><|det|>[[40, 950, 950, 985]]<|/det|>Seite 1 von 4"
        "<|end▁of▁sentence|>"
    )

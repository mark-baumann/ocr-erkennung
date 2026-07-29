"""Clean the raw model output into publishable Markdown / HTML / plain text.

Upstream does this inline in ``run_dpsk_ocr_image.py``: replace image
annotations with ``![](images/N.jpg)``, drop the remaining annotations, fix two
LaTeX macros. This module keeps that behaviour and adds special-token
stripping, HTML table extraction and a text-only projection.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .grounding import GROUNDING_PATTERN, CroppedFigure, LayoutRegion

#: Chat/control tokens that leak into the output when ``skip_special_tokens`` is off.
SPECIAL_TOKENS = (
    "<|end▁of▁sentence|>",
    "<|begin▁of▁sentence|>",
    "<|end_of_sentence|>",
    "<|begin_of_sentence|>",
    "<|User|>",
    "<|Assistant|>",
    "<image>",
    "<|grounding|>",
)

#: LaTeX macros DeepSeek emits that most renderers do not know.
LATEX_FIXUPS: Dict[str, str] = {
    r"\coloneqq": ":=",
    r"\eqqcolon": "=:",
    r"\eqcolon": "=:",
}

_TABLE_RE = re.compile(r"<table.*?>.*?</table>", re.DOTALL | re.IGNORECASE)
_TR_RE = re.compile(r"<tr.*?>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<t[dh].*?>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def strip_special_tokens(text: str) -> str:
    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


def apply_latex_fixups(text: str) -> str:
    for needle, replacement in LATEX_FIXUPS.items():
        text = text.replace(needle, replacement)
    return text


def strip_grounding(text: str) -> str:
    """Remove every ``<|ref|>…<|det|>…`` annotation, keeping the content."""
    return GROUNDING_PATTERN.sub("", text)


def tidy(text: str) -> str:
    """Collapse excessive blank lines and trailing whitespace."""
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return _BLANK_RUN_RE.sub("\n\n", text).strip()


@dataclass
class CleanedOutput:
    """The raw response projected into the formats the UI offers."""

    raw: str
    markdown: str
    plain_text: str
    figures: List[CroppedFigure]

    @property
    def word_count(self) -> int:
        return len(self.plain_text.split())

    @property
    def char_count(self) -> int:
        return len(self.plain_text)


def build_markdown(
    raw: str,
    regions: Sequence[LayoutRegion],
    figures: Sequence[CroppedFigure],
    *,
    figure_dir: str = "figures",
) -> str:
    """Replace figure annotations with image links and drop the rest.

    Figure regions are matched to crops in emission order, mirroring the
    ``img_idx`` counter upstream uses while drawing.
    """
    text = raw
    figure_regions = [region for region in regions if region.is_figure]

    for region, figure in zip(figure_regions, figures):
        text = text.replace(region.raw, f"\n\n![{figure.label}]({figure_dir}/{figure.filename})\n\n", 1)

    text = strip_grounding(text)
    text = strip_special_tokens(text)
    text = apply_latex_fixups(text)
    return tidy(text)


def to_plain_text(markdown_text: str) -> str:
    """Strip markdown/HTML markup down to readable prose."""
    text = _TAG_RE.sub(" ", markdown_text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__|\*|_|`)", "", text)
    text = re.sub(r"^\s*\|", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return tidy(text)


def clean_output(
    raw: str,
    regions: Sequence[LayoutRegion],
    figures: Sequence[CroppedFigure],
    *,
    figure_dir: str = "figures",
) -> CleanedOutput:
    markdown_text = build_markdown(raw, regions, figures, figure_dir=figure_dir)
    return CleanedOutput(
        raw=raw,
        markdown=markdown_text,
        plain_text=to_plain_text(markdown_text),
        figures=list(figures),
    )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


@dataclass
class ExtractedTable:
    """An HTML table found in the output, normalised to a rectangular grid."""

    index: int
    rows: List[List[str]]

    @property
    def header(self) -> List[str]:
        return self.rows[0] if self.rows else []

    @property
    def body(self) -> List[List[str]]:
        return self.rows[1:] if len(self.rows) > 1 else []

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.rows), max((len(r) for r in self.rows), default=0))

    def to_csv(self) -> str:
        import csv
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerows(self.rows)
        return buffer.getvalue()

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        width = self.shape[1]
        padded = [row + [""] * (width - len(row)) for row in self.rows]
        lines = ["| " + " | ".join(padded[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
        lines += ["| " + " | ".join(row) + " |" for row in padded[1:]]
        return "\n".join(lines)


def _cell_text(cell_html: str) -> str:
    text = _TAG_RE.sub(" ", cell_html)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_tables(text: str) -> List[ExtractedTable]:
    """Pull every ``<table>`` block out of the output as a normalised grid."""
    tables: List[ExtractedTable] = []
    for match in _TABLE_RE.finditer(text):
        rows: List[List[str]] = []
        for row_match in _TR_RE.finditer(match.group(0)):
            cells = [_cell_text(c) for c in _CELL_RE.findall(row_match.group(1))]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(ExtractedTable(index=len(tables), rows=rows))
    return tables


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_HTML_SHELL = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ max-width: 46rem; margin: 3rem auto; padding: 0 1.25rem;
         font: 16px/1.65 -apple-system, "Segoe UI", Roboto, sans-serif; color: #1c1f24; }}
  h1, h2, h3 {{ line-height: 1.25; margin-top: 2rem; }}
  img {{ max-width: 100%; height: auto; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1.25rem 0; }}
  th, td {{ border: 1px solid #d6dae0; padding: .45rem .6rem; text-align: left; }}
  th {{ background: #f4f6f8; }}
  pre {{ background: #f4f6f8; padding: .9rem; overflow-x: auto; border-radius: 6px; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #14171a; color: #e6e8ea; }}
    th, td {{ border-color: #333a42; }}
    th, pre {{ background: #1e2226; }}
  }}
</style>
{body}
"""


def to_html(markdown_text: str, *, title: str = "DeepSeek-OCR") -> str:
    """Render markdown to a standalone HTML document.

    Uses the ``markdown`` package when available and falls back to a small
    built-in converter so the export never hard-depends on it.
    """
    try:
        import markdown as md  # type: ignore

        body = md.markdown(markdown_text, extensions=["tables", "fenced_code"])
    except ImportError:
        body = _minimal_markdown_to_html(markdown_text)
    return _HTML_SHELL.format(title=html.escape(title), body=body)


def _minimal_markdown_to_html(text: str) -> str:
    """Good-enough markdown → HTML for headings, lists, images and paragraphs."""
    out: List[str] = []
    in_list = False

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            if in_list:
                out.append("</ul>")
                in_list = False
            continue

        # Pass raw HTML (the model's tables) straight through.
        if stripped.startswith("<"):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(stripped)
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            if in_list:
                out.append("</ul>")
                in_list = False
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            continue

        if re.match(r"^[-*+]\s+", stripped):
            if not in_list:
                out.append("<ul>")
                in_list = True
            item = re.sub(r"^[-*+]\s+", "", stripped)
            out.append(f"<li>{_inline(item)}</li>")
            continue

        if in_list:
            out.append("</ul>")
            in_list = False
        out.append(f"<p>{_inline(stripped)}</p>")

    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _inline(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)
    return escaped


def merge_pages(sections: Sequence[Tuple[str, str]], *, separator: Optional[str] = None) -> str:
    """Join per-page markdown into one document with page headings."""
    sep = separator if separator is not None else "\n\n---\n\n"
    blocks = [f"<!-- {label} -->\n\n{body}".strip() for label, body in sections if body.strip()]
    return sep.join(blocks)

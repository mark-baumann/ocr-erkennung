"""Export a pipeline run to downloadable artefacts."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from PIL import Image

from .markdown import to_html

if TYPE_CHECKING:  # pragma: no cover
    from ..pipeline import PageResult, RunResult


@dataclass
class Artifact:
    """One downloadable file."""

    filename: str
    data: bytes
    mime: str
    label: str

    @property
    def size_kb(self) -> float:
        return len(self.data) / 1024


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def markdown_artifact(run: "RunResult", *, stem: str = "ocr") -> Artifact:
    return Artifact(
        filename=f"{stem}.md",
        data=run.combined_markdown().encode("utf-8"),
        mime="text/markdown",
        label="Markdown",
    )


def text_artifact(run: "RunResult", *, stem: str = "ocr") -> Artifact:
    return Artifact(
        filename=f"{stem}.txt",
        data=run.combined_text().encode("utf-8"),
        mime="text/plain",
        label="Nur Text",
    )


def html_artifact(run: "RunResult", *, stem: str = "ocr", title: str = "DeepSeek-OCR") -> Artifact:
    return Artifact(
        filename=f"{stem}.html",
        data=to_html(run.combined_markdown(), title=title).encode("utf-8"),
        mime="text/html",
        label="HTML",
    )


def json_artifact(run: "RunResult", *, stem: str = "ocr") -> Artifact:
    payload = json.dumps(run.to_dict(), ensure_ascii=False, indent=2)
    return Artifact(
        filename=f"{stem}.json",
        data=payload.encode("utf-8"),
        mime="application/json",
        label="JSON (Layout + Text)",
    )


def bundle_artifact(
    run: "RunResult",
    *,
    stem: str = "deepseek-ocr-export",
    include_overlays: bool = True,
    include_pages: bool = False,
) -> Artifact:
    """Everything in one ZIP: markdown, text, HTML, JSON, figures, tables, overlays."""
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.md", _bundle_readme(run))
        archive.writestr("document.md", run.combined_markdown())
        archive.writestr("document.txt", run.combined_text())
        archive.writestr("document.html", to_html(run.combined_markdown(), title=stem))
        archive.writestr("layout.json", json.dumps(run.to_dict(), ensure_ascii=False, indent=2))

        for page in run.successful:
            prefix = f"pages/p{page.page.page_number:03d}"
            archive.writestr(f"{prefix}/page.md", page.markdown)
            archive.writestr(f"{prefix}/raw.mmd", page.result.text)

            for figure in page.figures:
                archive.writestr(f"{prefix}/figures/{figure.filename}", _png_bytes(figure.image))

            for table in page.tables:
                archive.writestr(f"{prefix}/tables/table_{table.index:02d}.csv", table.to_csv())

            if include_overlays and page.regions:
                archive.writestr(f"{prefix}/layout_overlay.png", _png_bytes(page.overlay()))

            if include_pages:
                archive.writestr(f"{prefix}/source.png", _png_bytes(page.page.image))

    return Artifact(
        filename=f"{stem}.zip",
        data=buffer.getvalue(),
        mime="application/zip",
        label="Komplettes Paket (ZIP)",
    )


def _bundle_readme(run: "RunResult") -> str:
    summary = run.summary
    lines = [
        "# DeepSeek-OCR Studio — Export",
        "",
        f"- Backend: `{summary.backend}`",
        f"- Resolution-Mode: `{run.mode_key}`",
        f"- Prompt: `{run.prompt.strip()}`",
        f"- Seiten: {summary.pages} (davon fehlgeschlagen: {summary.failed})",
        f"- Vision-Tokens: {summary.visual_tokens:,}".replace(",", "."),
        f"- Erkannte Layout-Regionen: {summary.regions}",
        f"- Gesamtlaufzeit: {summary.total_latency_s:.1f} s",
        "",
        "## Inhalt",
        "",
        "| Pfad | Beschreibung |",
        "| --- | --- |",
        "| `document.md` | Alle Seiten als ein Markdown-Dokument |",
        "| `document.txt` | Reiner Text ohne Markup |",
        "| `document.html` | Eigenständige HTML-Fassung |",
        "| `layout.json` | Strukturierte Regionen, Boxen und Metriken |",
        "| `pages/pNNN/page.md` | Markdown je Seite |",
        "| `pages/pNNN/raw.mmd` | Unbearbeitete Modellausgabe inkl. Grounding-Tags |",
        "| `pages/pNNN/figures/` | Ausgeschnittene Abbildungen |",
        "| `pages/pNNN/tables/` | Tabellen als CSV |",
        "| `pages/pNNN/layout_overlay.png` | Seite mit eingezeichneten Regionen |",
    ]

    if not summary.is_real:
        lines.insert(
            1,
            "\n> **Achtung:** Dieser Export stammt aus dem Demo-Backend (keine GPU), "
            "nicht aus DeepSeek-OCR.\n",
        )

    return "\n".join(lines) + "\n"


def page_artifacts(page: "PageResult") -> List[Artifact]:
    """Per-page downloads for the detail view."""
    stem = f"p{page.page.page_number:03d}"
    artifacts = [
        Artifact(f"{stem}.md", page.markdown.encode("utf-8"), "text/markdown", "Markdown"),
        Artifact(f"{stem}_raw.mmd", page.result.text.encode("utf-8"), "text/plain", "Rohausgabe"),
    ]
    if page.regions:
        artifacts.append(
            Artifact(
                f"{stem}_overlay.png",
                _png_bytes(page.overlay()),
                "image/png",
                "Layout-Overlay",
            )
        )
    return artifacts


def all_artifacts(run: "RunResult", *, stem: str = "ocr", title: Optional[str] = None) -> List[Artifact]:
    return [
        markdown_artifact(run, stem=stem),
        text_artifact(run, stem=stem),
        html_artifact(run, stem=stem, title=title or stem),
        json_artifact(run, stem=stem),
    ]

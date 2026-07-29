"""Headless batch OCR — the same pipeline the Streamlit app uses.

This replaces upstream's edit-the-config-then-run-the-script workflow::

    python -m dsocr.cli scans/*.pdf -o out/ --mode gundam --prompt markdown

Everything the UI can produce is available here too, which makes the app
scriptable for cron jobs and CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

from .config import MODES, PROMPTS, SETTINGS, Settings, get_mode
from .engines import EngineUnavailable, create_engine, probe_all
from .pipeline import RunResult, resolve_prompt
from .pipeline import run as run_pipeline
from .postprocess.export import bundle_artifact
from .preprocess.loader import Page, UnsupportedDocument, load_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsocr",
        description="DeepSeek-OCR im Batch — Bilder und PDFs zu Markdown, JSON und ZIP.",
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="Bild- oder PDF-Dateien.")
    parser.add_argument("-o", "--output", type=Path, default=Path("dsocr-out"), help="Ausgabeverzeichnis.")
    parser.add_argument("--mode", choices=list(MODES), default="gundam", help="Auflösungsmodus.")
    parser.add_argument("--prompt", choices=list(PROMPTS), default="markdown", help="Prompt-Vorlage.")
    parser.add_argument("--query", default="", help="Eingabe für Prompts mit Platzhalter (locate / custom).")
    parser.add_argument(
        "--backend", choices=["auto", "vllm", "transformers", "demo"], default="auto", help="Inferenz-Backend."
    )
    parser.add_argument("--model", default=SETTINGS.model_path, help="Modellpfad oder HF-Repo.")
    parser.add_argument("--dpi", type=int, default=SETTINGS.pdf_dpi, help="PDF-Rasterauflösung.")
    parser.add_argument("--max-pages", type=int, default=SETTINGS.max_pages, help="Seitenlimit pro PDF.")
    parser.add_argument("--min-crops", type=int, default=SETTINGS.min_crops, help="Minimale Kachelanzahl.")
    parser.add_argument("--max-crops", type=int, default=SETTINGS.max_crops, help="Maximale Kachelanzahl.")
    parser.add_argument("--no-batch", action="store_true", help="Seiten einzeln statt im Batch verarbeiten.")
    parser.add_argument("--zip", action="store_true", help="Zusätzlich ein ZIP-Komplettpaket schreiben.")
    parser.add_argument("--list-backends", action="store_true", help="Verfügbare Backends anzeigen und beenden.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Keine Fortschrittsausgabe.")
    return parser


def _load_pages(paths: Sequence[Path], settings: Settings, quiet: bool) -> List[Page]:
    pages: List[Page] = []
    for path in paths:
        if not path.is_file():
            print(f"übersprungen: {path} existiert nicht", file=sys.stderr)
            continue
        try:
            document = load_document(
                path.name,
                path.read_bytes(),
                dpi=settings.pdf_dpi,
                max_pages=settings.max_pages,
                start_index=len(pages),
            )
        except UnsupportedDocument as exc:
            print(f"übersprungen: {exc}", file=sys.stderr)
            continue
        pages.extend(document.pages)
        if not quiet:
            print(f"geladen: {path.name} → {len(document)} Seite(n)")
    return pages


def _write_outputs(run: RunResult, output: Path, *, write_zip: bool, quiet: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)

    (output / "document.md").write_text(run.combined_markdown(), encoding="utf-8")
    (output / "document.txt").write_text(run.combined_text(), encoding="utf-8")
    (output / "layout.json").write_text(
        json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for page in run.successful:
        page_dir = output / f"p{page.page.page_number:03d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        (page_dir / "page.md").write_text(page.markdown, encoding="utf-8")
        (page_dir / "raw.mmd").write_text(page.result.text, encoding="utf-8")

        for figure in page.figures:
            figure.image.save(page_dir / figure.filename)
        for table in page.tables:
            (page_dir / f"table_{table.index:02d}.csv").write_text(table.to_csv(), encoding="utf-8")
        if page.regions:
            page.overlay().save(page_dir / "layout_overlay.png")

    if write_zip:
        bundle = bundle_artifact(run)
        (output / bundle.filename).write_bytes(bundle.data)

    if not quiet:
        print(f"\ngeschrieben nach: {output.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_backends:
        for info in probe_all():
            mark = "✓" if info.available else "✗"
            synthetic = "  (synthetisch)" if not info.is_real else ""
            print(f" {mark} {info.key:14s} {info.detail}{synthetic}")
        return 0

    if not args.inputs:
        build_parser().error("Keine Eingabedateien angegeben.")

    settings = Settings()
    settings.model_path = args.model
    settings.pdf_dpi = args.dpi
    settings.max_pages = args.max_pages
    settings.min_crops = args.min_crops
    settings.max_crops = args.max_crops

    pages = _load_pages(args.inputs, settings, args.quiet)
    if not pages:
        print("Keine verarbeitbaren Seiten gefunden.", file=sys.stderr)
        return 1

    try:
        engine = create_engine(args.backend, settings)
    except EngineUnavailable as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2

    if not engine.is_real and not args.quiet:
        print("\n⚠  Demo-Backend aktiv — die Ergebnisse stammen nicht von DeepSeek-OCR.\n", file=sys.stderr)

    prompt = resolve_prompt(PROMPTS[args.prompt], args.query)

    def on_progress(done: int, total: int, label: str) -> None:
        if not args.quiet:
            print(f"  [{done}/{total}] {label}")

    run = run_pipeline(
        pages,
        engine,
        prompt,
        get_mode(args.mode),
        settings=settings,
        on_progress=None if args.quiet else on_progress,
        batch=not args.no_batch,
    )

    _write_outputs(run, args.output, write_zip=args.zip, quiet=args.quiet)

    summary = run.summary
    if not args.quiet:
        print(
            f"Seiten: {summary.pages} · Fehler: {summary.failed} · "
            f"Vision-Tokens: {summary.visual_tokens} · "
            f"Kompression: {summary.compression_ratio:.0f} px/Token · "
            f"Laufzeit: {summary.total_latency_s:.1f}s"
        )

    return 1 if summary.failed == summary.pages else 0


if __name__ == "__main__":
    raise SystemExit(main())

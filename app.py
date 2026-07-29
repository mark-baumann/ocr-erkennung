"""DeepSeek-OCR Studio — Streamlit entry point.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import streamlit as st

from dsocr import __version__
from dsocr.config import SETTINGS, Settings
from dsocr.engines import EngineUnavailable, create_engine
from dsocr.pipeline import RunResult, resolve_prompt, run as run_pipeline
from dsocr.preprocess.loader import (
    SUPPORTED_SUFFIXES,
    Page,
    UnsupportedDocument,
    load_document,
    pdf_page_count,
    suffix_of,
)
from dsocr.preprocess.tiling import plan_tiling, tile_preview
from dsocr.ui import results as results_ui
from dsocr.ui import sidebar as sidebar_ui
from dsocr.ui.theme import configure_page, format_int, hero, metric_row

UPLOAD_TYPES = sorted(suffix.lstrip(".") for suffix in SUPPORTED_SUFFIXES)


@st.cache_resource(show_spinner=False)
def _engine_for(backend: str, model_path: str, gpu_util: float):
    """One engine instance per (backend, model, vram budget) — weights load once."""
    settings = Settings()
    settings.model_path = model_path
    settings.gpu_memory_utilization = gpu_util
    return create_engine(backend, settings)


def _session_settings() -> Settings:
    if "settings" not in st.session_state:
        st.session_state["settings"] = Settings(
            model_path=SETTINGS.model_path,
            backend=SETTINGS.backend,
            min_crops=SETTINGS.min_crops,
            max_crops=SETTINGS.max_crops,
        )
    return st.session_state["settings"]


def _collect_pages(uploads, settings: Settings) -> Tuple[List[Page], List[str]]:
    """Expand uploads into pages, collecting per-file errors instead of raising."""
    pages: List[Page] = []
    problems: List[str] = []

    for upload in uploads:
        data = upload.getvalue()
        size_mb = len(data) / 1024**2
        if size_mb > settings.max_upload_mb:
            problems.append(f"`{upload.name}` ist {size_mb:.0f} MB groß (Limit: {settings.max_upload_mb} MB).")
            continue

        selection = None
        if suffix_of(upload.name) == ".pdf":
            selection = st.session_state.get(f"pdfsel_{upload.name}")

        try:
            document = load_document(
                upload.name,
                data,
                dpi=settings.pdf_dpi,
                max_pages=settings.max_pages,
                page_selection=selection,
                start_index=len(pages),
            )
        except UnsupportedDocument as exc:
            problems.append(str(exc))
            continue
        except Exception as exc:  # pragma: no cover - unexpected decoder failure
            problems.append(f"`{upload.name}` konnte nicht gelesen werden: {exc}")
            continue

        pages.extend(document.pages)

    return pages, problems


def _pdf_page_selectors(uploads, settings: Settings) -> None:
    """Let the user restrict which PDF pages get rasterised."""
    pdfs = [u for u in uploads if suffix_of(u.name) == ".pdf"]
    if not pdfs:
        return

    with st.expander("📑 Seitenauswahl für PDFs", expanded=False):
        for upload in pdfs:
            total = pdf_page_count(upload.getvalue())
            if total <= 1:
                st.caption(f"`{upload.name}` — {total} Seite")
                continue

            limit = min(total, settings.max_pages)
            first, last = st.slider(
                f"`{upload.name}` — {total} Seiten",
                min_value=1,
                max_value=total,
                value=(1, limit),
                key=f"pdfrange_{upload.name}",
            )
            st.session_state[f"pdfsel_{upload.name}"] = list(range(first - 1, last))
            if last - first + 1 > settings.max_pages:
                st.caption(f"Es werden die ersten {settings.max_pages} Seiten der Auswahl verarbeitet.")


def _budget_panel(pages: List[Page], controls) -> None:
    """Show the tiling plan and token cost before anything runs."""
    settings = controls.settings
    lo, hi = settings.crop_bounds()
    plans = [plan_tiling(page.size, controls.mode, min_num=lo, max_num=hi) for page in pages]

    visual = sum(p.visual_tokens for p in plans)
    sequence = sum(p.sequence_tokens for p in plans)
    pixels = sum(p.source_pixels for p in plans)
    ratio = pixels / visual if visual else 0

    metric_row(
        [
            ("Seiten", format_int(len(pages)), controls.mode.label.split(" · ")[0] + "-Modus"),
            ("Vision-Tokens", format_int(visual), f"{format_int(sequence)} inkl. Trenner"),
            ("Kompression", f"{ratio:,.0f}".replace(",", ".") if ratio else "—", "Quellpixel je Token"),
            (
                "Kontext",
                f"{sequence / max(1, settings.max_model_len) * 100:.0f} %",
                f"von {format_int(settings.max_model_len)} Tokens (gesamt)",
            ),
        ]
    )

    if sequence > settings.max_model_len and len(pages) == 1:
        st.warning(
            f"Diese Seite belegt {format_int(sequence)} Tokens und passt nicht in das "
            f"Kontextfenster von {format_int(settings.max_model_len)}. Wähle einen kleineren "
            "Auflösungsmodus oder reduziere das Kachel-Budget.",
            icon="⚠️",
        )

    preview_index = 0
    if len(pages) > 1:
        preview_index = st.selectbox(
            "Vorschau",
            range(len(pages)),
            format_func=lambda i: pages[i].label,
            key="budget_preview",
        )

    page = pages[preview_index]
    plan = plans[preview_index]

    left, right = st.columns([1, 1], gap="medium")
    with left:
        st.image(page.image, caption=f"Original · {page.size[0]}×{page.size[1]}px", use_container_width=True)
    with right:
        st.image(
            tile_preview(page.image, plan),
            caption=f"Encoder-Sicht · {plan.describe()} · {format_int(plan.visual_tokens)} Vision-Tokens",
            use_container_width=True,
        )


def _run(pages: List[Page], controls) -> Optional[RunResult]:
    try:
        engine = _engine_for(
            controls.backend, controls.settings.model_path, controls.settings.gpu_memory_utilization
        )
    except EngineUnavailable as exc:
        st.error(str(exc))
        return None

    prompt = resolve_prompt(controls.preset, controls.query)
    if controls.preset.needs_query and not controls.query.strip():
        st.error("Für diesen Prompt fehlt noch die Eingabe in der Seitenleiste.")
        return None

    progress = st.progress(0.0, text="Wird vorbereitet …")

    def on_progress(done: int, total: int, label: str) -> None:
        progress.progress(min(1.0, done / max(1, total)), text=f"{done}/{total} · {label}")

    try:
        with st.spinner("Modell arbeitet …"):
            result = run_pipeline(
                pages,
                engine,
                prompt,
                controls.mode,
                settings=controls.settings,
                on_progress=on_progress,
                batch=controls.batch,
            )
    except Exception as exc:
        progress.empty()
        st.error(f"Der Durchlauf ist fehlgeschlagen: {exc}")
        return None

    progress.empty()
    return result


def main() -> None:
    configure_page()

    settings = _session_settings()
    controls = sidebar_ui.render(settings)

    infos = sidebar_ui.backend_infos()
    resolved = controls.backend
    if resolved == "auto":
        resolved = next((i.key for i in infos if i.available), "")
    hero(sidebar_ui.status_chips(infos, active=resolved))

    uploads = st.file_uploader(
        "Dokumente hochladen",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        help="Bilder und PDFs. Mehrseitige PDFs werden automatisch in Seiten zerlegt.",
    )

    if not uploads:
        _empty_state()
        return

    _pdf_page_selectors(uploads, controls.settings)

    with st.spinner("Seiten werden vorbereitet …"):
        pages, problems = _collect_pages(uploads, controls.settings)

    for problem in problems:
        st.warning(problem, icon="📄")

    if not pages:
        st.error("Keine verarbeitbaren Seiten gefunden.")
        return

    st.markdown("### 🧮 Token-Budget & Encoder-Sicht")
    _budget_panel(pages, controls)

    if st.button("🚀 Analyse starten", type="primary", use_container_width=True):
        st.session_state.pop("bundle", None)
        result = _run(pages, controls)
        if result is not None:
            st.session_state["run"] = result

    result = st.session_state.get("run")
    if result is not None:
        st.markdown("### 📋 Ergebnis")
        results_ui.render(result, show_labels=controls.show_labels)


def _empty_state() -> None:
    st.info(
        "Lade ein Bild oder PDF hoch, um zu starten. Unterstützt werden "
        f"{', '.join('`.' + t + '`' for t in UPLOAD_TYPES)}.",
        icon="👆",
    )

    left, middle, right = st.columns(3, gap="medium")
    with left:
        st.markdown(
            "#### 🧩 Was passiert hier?\n"
            "Die Seite wird in wenige hundert *Vision-Tokens* komprimiert und vom Modell "
            "direkt als Markdown zurückgelesen — inklusive Überschriften, Tabellen und "
            "Formeln."
        )
    with middle:
        st.markdown(
            "#### 🗺️ Layout-Grounding\n"
            "Mit einem `<|grounding|>`-Prompt liefert das Modell für jede Region eine "
            "Bounding-Box. Daraus entstehen das Overlay, die Abbildungs-Ausschnitte und "
            "das Layout-JSON."
        )
    with right:
        st.markdown(
            "#### 🎚️ Auflösungsmodi\n"
            "Von *Tiny* (64 Tokens) bis *Gundam* (dynamisches Tiling). Das Token-Budget "
            "wird vor dem Start angezeigt, damit die Wahl nicht geraten werden muss."
        )

    st.caption(f"DeepSeek-OCR Studio v{__version__}")


if __name__ == "__main__":
    main()

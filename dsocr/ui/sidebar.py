"""Sidebar: backend, resolution mode, prompt and advanced knobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import streamlit as st

from ..config import DEFAULT_MODE, DEFAULT_PROMPT, MODES, PROMPTS, PromptPreset, ResolutionMode, Settings
from ..engines import AUTO_ORDER, BackendInfo, probe_all
from .theme import chip, sidebar_note


@dataclass
class Controls:
    """Everything the sidebar collected."""

    backend: str
    mode: ResolutionMode
    preset: PromptPreset
    query: str
    settings: Settings
    show_labels: bool
    batch: bool


@st.cache_data(show_spinner=False, ttl=60)
def _cached_probe() -> List[tuple]:
    """Probing touches CUDA, so cache it briefly instead of on every rerun."""
    return [(i.key, i.label, i.available, i.detail, i.is_real) for i in probe_all()]


def backend_infos() -> List[BackendInfo]:
    return [BackendInfo(*row) for row in _cached_probe()]


def status_chips(infos: List[BackendInfo], active: Optional[str] = None) -> str:
    chips = []
    for info in infos:
        if info.available:
            kind = "accent" if info.key == active else ("warn" if not info.is_real else "ok")
            chips.append(chip(f"{info.label} · bereit", kind))
        else:
            chips.append(chip(f"{info.label} · n/a", "err"))
    return "".join(chips)


def render(settings: Settings) -> Controls:
    st.sidebar.markdown("### ⚙️ Konfiguration")

    infos = backend_infos()
    available = [i for i in infos if i.available]

    # ---------------- Backend ----------------
    options = ["auto"] + [i.key for i in infos]
    labels = {"auto": "Automatisch"}
    labels.update({i.key: i.label + ("" if i.available else "  (nicht verfügbar)") for i in infos})

    backend = st.sidebar.selectbox(
        "Backend",
        options,
        format_func=lambda k: labels[k],
        help="`Automatisch` nimmt das schnellste verfügbare Backend: vLLM → Transformers → Demo.",
    )

    if backend != "auto":
        chosen = next((i for i in infos if i.key == backend), None)
        if chosen is not None and not chosen.available:
            st.sidebar.error(chosen.detail)

    resolved = backend if backend != "auto" else next(
        (k for k in AUTO_ORDER if any(i.key == k and i.available for i in infos)), None
    )
    active = next((i for i in infos if i.key == resolved), None)
    if active is not None:
        if active.is_real:
            st.sidebar.success(f"**{active.label}** — {active.detail}")
        else:
            st.sidebar.warning(
                f"**{active.label}** — {active.detail}\n\n"
                "DeepSeek-OCR braucht eine NVIDIA-GPU. Die Ergebnisse hier sind ein Platzhalter."
            )
    elif not available:
        st.sidebar.error("Kein Backend verfügbar.")

    st.sidebar.divider()

    # ---------------- Resolution mode ----------------
    st.sidebar.markdown("### 🎚️ Auflösung")
    mode_key = st.sidebar.selectbox(
        "Modus",
        list(MODES),
        index=list(MODES).index(DEFAULT_MODE),
        format_func=lambda k: MODES[k].label,
    )
    mode = MODES[mode_key]
    sidebar_note(mode.blurb)

    # ---------------- Prompt ----------------
    st.sidebar.markdown("### 💬 Aufgabe")
    preset_key = st.sidebar.selectbox(
        "Prompt",
        list(PROMPTS),
        index=list(PROMPTS).index(DEFAULT_PROMPT),
        format_func=lambda k: PROMPTS[k].label,
    )
    preset = PROMPTS[preset_key]
    sidebar_note(preset.blurb)

    query = ""
    if preset.needs_query:
        if preset.key == "locate":
            query = st.sidebar.text_input("Gesuchter Text", placeholder="z. B. Rechnungsnummer")
        else:
            query = st.sidebar.text_area(
                "Prompt-Text",
                value="<image>\nFree OCR.",
                height=96,
                help="Muss `<image>` enthalten, damit das Bild verarbeitet wird.",
            )

    show_labels = st.sidebar.toggle("Regionsnamen im Overlay", value=True)

    # ---------------- Advanced ----------------
    with st.sidebar.expander("Erweitert", expanded=False):
        settings.model_path = st.text_input("Modellpfad", value=settings.model_path)

        crops = st.slider(
            "Kachel-Budget (Gundam)",
            min_value=1,
            max_value=9,
            value=(settings.min_crops, settings.max_crops),
            help="Minimale und maximale Anzahl lokaler Kacheln. Weniger Kacheln = weniger "
            "VRAM und Tokens, aber gröbere Auflösung feiner Schrift.",
        )
        settings.min_crops, settings.max_crops = crops

        settings.max_new_tokens = st.slider(
            "Max. Ausgabe-Tokens", 512, 16384, value=settings.max_new_tokens, step=512
        )
        settings.pdf_dpi = st.select_slider(
            "PDF-Rasterung (DPI)", options=[96, 120, 144, 200, 300], value=settings.pdf_dpi
        )
        settings.max_pages = st.number_input(
            "Max. Seiten pro PDF", min_value=1, max_value=1000, value=settings.max_pages, step=1
        )
        batch = st.toggle(
            "Batch-Inferenz",
            value=True,
            help="Nur vLLM: alle Seiten in einem Durchlauf planen. Deutlich schneller bei PDFs.",
        )
        settings.gpu_memory_utilization = st.slider(
            "GPU-Speicheranteil (vLLM)", 0.3, 0.98, value=settings.gpu_memory_utilization, step=0.01
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "Modell: [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) · "
        "Code: [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) (MIT)"
    )

    return Controls(
        backend=backend,
        mode=mode,
        preset=preset,
        query=query,
        settings=settings,
        show_labels=show_labels,
        batch=batch,
    )

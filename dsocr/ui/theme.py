"""Page chrome: config, CSS and small presentational helpers.

Everything here is theme-aware — Streamlit's light and dark themes both have to
look deliberate, so colours are derived from CSS custom properties rather than
hard-coded hex values.
"""

from __future__ import annotations

import streamlit as st

PAGE_TITLE = "DeepSeek-OCR Studio"
PAGE_ICON = "🔍"

_CSS = """
<style>
:root {
  --dsocr-accent: #ff5c00;
  --dsocr-accent-soft: rgba(255, 92, 0, .12);
  --dsocr-border: rgba(128, 138, 152, .28);
  --dsocr-muted: rgba(128, 138, 152, 1);
  --dsocr-surface: rgba(128, 138, 152, .07);
}

/* Tighten the default vertical rhythm - the app is dense by nature. */
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1400px; }

/* ---------- Hero ---------- */
.dsocr-hero {
  border: 1px solid var(--dsocr-border);
  border-radius: 14px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1.4rem;
  background: linear-gradient(135deg, var(--dsocr-accent-soft), transparent 62%);
}
.dsocr-hero h1 {
  margin: 0 0 .35rem 0;
  font-size: 1.65rem;
  font-weight: 700;
  letter-spacing: -.02em;
}
.dsocr-hero p { margin: 0; color: var(--dsocr-muted); font-size: .94rem; line-height: 1.55; }
.dsocr-hero .dsocr-chips { margin-top: .85rem; }

/* ---------- Chips ---------- */
.dsocr-chip {
  display: inline-block;
  padding: .18rem .6rem;
  margin: 0 .35rem .35rem 0;
  border-radius: 999px;
  border: 1px solid var(--dsocr-border);
  font-size: .74rem;
  font-weight: 600;
  letter-spacing: .01em;
  white-space: nowrap;
}
.dsocr-chip.ok     { border-color: rgba(34, 168, 96, .45);  color: #22a860; background: rgba(34, 168, 96, .1); }
.dsocr-chip.warn   { border-color: rgba(214, 148, 0, .5);   color: #d69400; background: rgba(214, 148, 0, .1); }
.dsocr-chip.err    { border-color: rgba(214, 66, 66, .5);   color: #d64242; background: rgba(214, 66, 66, .1); }
.dsocr-chip.accent { border-color: rgba(255, 92, 0, .5);    color: var(--dsocr-accent); background: var(--dsocr-accent-soft); }

/* ---------- Metric cards ---------- */
.dsocr-metrics { display: flex; flex-wrap: wrap; gap: .7rem; margin: .2rem 0 1rem 0; }
.dsocr-metric {
  flex: 1 1 150px;
  border: 1px solid var(--dsocr-border);
  border-radius: 11px;
  padding: .7rem .85rem;
  background: var(--dsocr-surface);
}
.dsocr-metric .k {
  font-size: .68rem;
  text-transform: uppercase;
  letter-spacing: .07em;
  color: var(--dsocr-muted);
  font-weight: 700;
}
.dsocr-metric .v {
  font-size: 1.4rem;
  font-weight: 700;
  line-height: 1.25;
  margin-top: .15rem;
  font-variant-numeric: tabular-nums;
}
.dsocr-metric .h { font-size: .74rem; color: var(--dsocr-muted); margin-top: .1rem; }

/* ---------- Legend ---------- */
.dsocr-legend { display: flex; flex-wrap: wrap; gap: .45rem; margin: .5rem 0 .9rem 0; }
.dsocr-legend-item {
  display: inline-flex;
  align-items: center;
  gap: .38rem;
  font-size: .78rem;
  padding: .16rem .55rem .16rem .35rem;
  border: 1px solid var(--dsocr-border);
  border-radius: 999px;
}
.dsocr-swatch { width: .72rem; height: .72rem; border-radius: 3px; display: inline-block; }

/* ---------- Document preview ---------- */
.dsocr-doc {
  border: 1px solid var(--dsocr-border);
  border-radius: 11px;
  padding: 1.1rem 1.4rem;
  max-height: 68vh;
  overflow-y: auto;
}
.dsocr-doc table { border-collapse: collapse; width: 100%; margin: .8rem 0; }
.dsocr-doc th, .dsocr-doc td { border: 1px solid var(--dsocr-border); padding: .35rem .5rem; }

/* Wide content must scroll inside its own box, never the page. */
.dsocr-scroll-x { overflow-x: auto; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
.dsocr-sidebar-note {
  font-size: .78rem;
  color: var(--dsocr-muted);
  line-height: 1.5;
  border-left: 2px solid var(--dsocr-border);
  padding-left: .6rem;
  margin: .3rem 0 .7rem 0;
}
</style>
"""


def configure_page() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                "**DeepSeek-OCR Studio** — Streamlit-Oberfläche für "
                "[DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)."
            )
        },
    )
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(chips_html: str = "") -> None:
    st.markdown(
        f"""
        <div class="dsocr-hero">
          <h1>{PAGE_ICON} DeepSeek-OCR Studio</h1>
          <p>Dokumente, Scans und Fotos in strukturiertes Markdown verwandeln — mit
          Layout-Erkennung, Tabellen- und Abbildungsextraktion sowie einer
          Live-Analyse des optischen Kompressionsverhältnisses.</p>
          <div class="dsocr-chips">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chip(text: str, kind: str = "") -> str:
    return f'<span class="dsocr-chip {kind}">{text}</span>'


def metric_row(items) -> None:
    """Render ``(label, value, hint)`` triples as a row of cards."""
    cards = "".join(
        f'<div class="dsocr-metric"><div class="k">{label}</div>'
        f'<div class="v">{value}</div>'
        f'<div class="h">{hint}</div></div>'
        for label, value, hint in items
    )
    st.markdown(f'<div class="dsocr-metrics">{cards}</div>', unsafe_allow_html=True)


def legend(labels_to_colors) -> None:
    """Colour legend for the layout overlay."""
    if not labels_to_colors:
        return
    items = "".join(
        f'<span class="dsocr-legend-item">'
        f'<span class="dsocr-swatch" style="background: rgb({r},{g},{b})"></span>{label}</span>'
        for label, (r, g, b) in labels_to_colors
    )
    st.markdown(f'<div class="dsocr-legend">{items}</div>', unsafe_allow_html=True)


def sidebar_note(text: str) -> None:
    st.sidebar.markdown(f'<div class="dsocr-sidebar-note">{text}</div>', unsafe_allow_html=True)


def format_int(value: int) -> str:
    """German thousands separator."""
    return f"{value:,}".replace(",", ".")

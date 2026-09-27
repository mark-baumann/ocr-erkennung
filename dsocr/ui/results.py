"""Result rendering: document, layout, tables, figures, raw output, metrics."""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

from ..pipeline import PageResult, RunResult
from ..postprocess.export import all_artifacts, bundle_artifact, page_artifacts
from ..postprocess.grounding import color_for_label, label_counts
from .theme import format_int, legend, metric_row


def render(run: RunResult, *, show_labels: bool = True) -> None:
    if not run.pages:
        return

    _summary(run)

    failed = [p for p in run.pages if not p.ok]
    if failed:
        with st.expander(f"⚠️ {len(failed)} Seite(n) mit Fehlern", expanded=len(failed) == len(run.pages)):
            for page in failed:
                st.error(f"**{page.label}** — {page.error}")

    if not run.successful:
        return

    tabs = st.tabs(
        ["📄 Dokument", "🗺️ Layout", "📊 Tabellen", "🖼️ Abbildungen", "🧾 Rohausgabe", "📈 Metriken", "⬇️ Export"]
    )

    with tabs[0]:
        _document_tab(run)
    with tabs[1]:
        _layout_tab(run, show_labels=show_labels)
    with tabs[2]:
        _tables_tab(run)
    with tabs[3]:
        _figures_tab(run)
    with tabs[4]:
        _raw_tab(run)
    with tabs[5]:
        _metrics_tab(run)
    with tabs[6]:
        _export_tab(run)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _summary(run: RunResult) -> None:
    summary = run.summary

    if not summary.is_real:
        st.warning(
            "**Demo-Backend** — diese Ergebnisse stammen nicht von DeepSeek-OCR, sondern von der "
            "CPU-Ersatzerkennung. Für echte Ergebnisse wird eine NVIDIA-GPU mit installiertem "
            "`torch` + `transformers` (oder vLLM) benötigt.",
            icon="🧪",
        )

    words = sum(p.cleaned.word_count for p in run.successful if p.cleaned)
    ratio = summary.compression_ratio

    metric_row(
        [
            ("Seiten", format_int(summary.pages), f"{summary.failed} fehlgeschlagen"),
            ("Vision-Tokens", format_int(summary.visual_tokens), f"{format_int(summary.sequence_tokens)} inkl. Trenner"),
            ("Kompression", f"{ratio:,.0f}".replace(",", ".") if ratio else "—", "Quellpixel je Vision-Token"),
            ("Regionen", format_int(summary.regions), f"{len(run.region_counts())} Typen"),
            ("Wörter", format_int(words), "im bereinigten Text"),
            ("Laufzeit", f"{summary.total_latency_s:.1f} s", f"⌀ {summary.avg_latency_s:.2f} s/Seite"),
        ]
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def _page_picker(run: RunResult, key: str) -> Optional[PageResult]:
    pages = run.successful
    if not pages:
        return None
    if len(pages) == 1:
        return pages[0]
    labels = [p.label for p in pages]
    choice = st.selectbox("Seite", range(len(pages)), format_func=lambda i: labels[i], key=key)
    return pages[choice]


def _document_tab(run: RunResult) -> None:
    view = st.radio(
        "Ansicht",
        ["Gesamtes Dokument", "Einzelne Seite"],
        horizontal=True,
        label_visibility="collapsed",
        key="doc_view",
    )

    if view == "Gesamtes Dokument":
        body = run.combined_markdown()
        st.markdown(f'<div class="dsocr-doc">\n\n{body}\n\n</div>', unsafe_allow_html=True)
        return

    page = _page_picker(run, "doc_page")
    if page is None:
        return

    left, right = st.columns([1, 1.15], gap="medium")
    with left:
        st.image(page.page.image, caption=page.label, use_container_width=True)
    with right:
        st.markdown(f'<div class="dsocr-doc">\n\n{page.markdown}\n\n</div>', unsafe_allow_html=True)


def _layout_tab(run: RunResult, *, show_labels: bool) -> None:
    page = _page_picker(run, "layout_page")
    if page is None:
        return

    if not page.regions:
        st.info(
            "Für diese Seite gibt es keine Layout-Boxen. Wähle einen Prompt mit `<|grounding|>` "
            "(z. B. „Dokument → Markdown“), damit das Modell Regionen ausgibt."
        )
        st.image(page.page.image, use_container_width=True)
        return

    counts = label_counts(page.regions)
    chosen = st.multiselect(
        "Regionstypen",
        options=list(counts),
        default=list(counts),
        format_func=lambda label: f"{label} ({counts[label]})",
        key="layout_filter",
    )

    legend([(label, color_for_label(label)) for label in chosen])

    left, right = st.columns([1.5, 1], gap="medium")
    with left:
        st.image(
            page.overlay(visible_labels=chosen, show_labels=show_labels),
            caption=f"{page.label} · {len(page.regions)} Regionen",
            use_container_width=True,
        )
    with right:
        st.markdown("**Regionen**")
        rows = [
            {
                "#": region.order,
                "Typ": region.label,
                "Fläche %": round(region.primary_box.area * 100, 1) if region.primary_box else 0.0,
                "Text": (region.text[:70] + "…") if len(region.text) > 70 else region.text,
            }
            for region in page.regions
            if region.label in chosen
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True, height=440)


def _tables_tab(run: RunResult) -> None:
    tables = run.all_tables()
    if not tables:
        st.info(
            "Keine Tabellen gefunden. DeepSeek-OCR gibt Tabellen als HTML aus — sie erscheinen "
            "hier automatisch, sobald welche im Dokument sind."
        )
        return

    st.caption(f"{len(tables)} Tabelle(n) erkannt.")
    for page in run.successful:
        for table in page.tables:
            rows, cols = table.shape
            with st.expander(f"{page.label} · Tabelle {table.index + 1} — {rows}×{cols}", expanded=True):
                header = table.header
                body = table.body
                if body and header:
                    width = cols
                    padded = [row + [""] * (width - len(row)) for row in body]
                    st.dataframe(
                        {header[i] if i < len(header) else f"Spalte {i+1}": [r[i] for r in padded] for i in range(width)},
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.table(table.rows)
                st.download_button(
                    "CSV herunterladen",
                    data=table.to_csv().encode("utf-8"),
                    file_name=f"table_p{page.page.page_number:03d}_{table.index:02d}.csv",
                    mime="text/csv",
                    key=f"csv_{page.page.index}_{table.index}",
                )


def _figures_tab(run: RunResult) -> None:
    entries = [(page, figure) for page in run.successful for figure in page.figures]
    if not entries:
        st.info(
            "Keine Abbildungen ausgeschnitten. Das Modell markiert Bilder mit "
            "`<|ref|>image<|/ref|>` — dafür wird ein Grounding-Prompt benötigt."
        )
        return

    st.caption(f"{len(entries)} Abbildung(en) aus {len({p.label for p, _ in entries})} Seite(n).")
    columns = st.columns(3, gap="medium")
    for index, (page, figure) in enumerate(entries):
        with columns[index % 3]:
            st.image(figure.image, use_container_width=True)
            st.caption(f"{page.label} · {figure.label} · {figure.image.width}×{figure.image.height}px")


def _raw_tab(run: RunResult) -> None:
    page = _page_picker(run, "raw_page")
    if page is None:
        return
    st.caption(
        "Unbearbeitete Modellausgabe inklusive `<|ref|>`/`<|det|>`-Tags — die Grundlage für "
        "alle anderen Ansichten."
    )
    st.code(page.result.text or "(leer)", language="markdown")


def _metrics_tab(run: RunResult) -> None:
    summary = run.summary

    st.markdown("#### Optische Kompression")
    st.caption(
        "DeepSeek-OCR komprimiert eine Seite in wenige hundert Vision-Tokens. Die Kennzahl zeigt, "
        "wie viele Quellpixel auf einen Token entfallen — je höher, desto stärker die Kompression."
    )

    rows = [
        {
            "Seite": page.label,
            "Größe": f"{page.page.size[0]}×{page.page.size[1]}",
            "Kacheln": page.plan.describe(),
            "Vision-Tokens": page.plan.visual_tokens,
            "Sequenz-Tokens": page.plan.sequence_tokens,
            "Px/Token": round(page.plan.compression_ratio),
            "Regionen": len(page.regions),
            "Sekunden": round(page.result.latency_s, 2),
        }
        for page in run.pages
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    counts = run.region_counts()
    if counts:
        st.markdown("#### Regionsverteilung")
        st.bar_chart(counts, horizontal=True, color="#ff5c00", height=max(160, 34 * len(counts)))

    st.markdown("#### Lauf-Konfiguration")
    st.json(
        {
            "backend": summary.backend,
            "mode": run.mode_key,
            "prompt": run.prompt,
            "synthetic_results": not summary.is_real,
            "pages": summary.pages,
            "vision_tokens_total": summary.visual_tokens,
            "sequence_tokens_total": summary.sequence_tokens,
            "compression_px_per_token": round(summary.compression_ratio, 1),
        },
        expanded=False,
    )


def _export_tab(run: RunResult) -> None:
    st.markdown("#### Gesamtes Dokument")

    columns = st.columns(4, gap="small")
    for column, artifact in zip(columns, all_artifacts(run, stem="deepseek-ocr")):
        with column:
            st.download_button(
                artifact.label,
                data=artifact.data,
                file_name=artifact.filename,
                mime=artifact.mime,
                use_container_width=True,
                key=f"dl_{artifact.filename}",
            )
            st.caption(f"{artifact.size_kb:.1f} KB")

    st.divider()
    st.markdown("#### Komplettpaket")
    st.caption(
        "ZIP mit Markdown, Text, HTML, Layout-JSON sowie pro Seite: Rohausgabe, "
        "ausgeschnittene Abbildungen, Tabellen als CSV und das Layout-Overlay."
    )

    include_pages = st.checkbox("Quellseiten als PNG mitliefern", value=False)
    if st.button("ZIP erzeugen", type="primary"):
        with st.spinner("Paket wird gebaut …"):
            bundle = bundle_artifact(run, include_pages=include_pages)
        st.session_state["bundle"] = (bundle.filename, bundle.data, bundle.size_kb)

    bundle_state = st.session_state.get("bundle")
    if bundle_state:
        filename, data, size_kb = bundle_state
        st.download_button(
            f"⬇️ {filename} ({size_kb:.0f} KB)",
            data=data,
            file_name=filename,
            mime="application/zip",
            type="primary",
        )

    st.divider()
    st.markdown("#### Einzelne Seite")
    page = _page_picker(run, "export_page")
    if page is None:
        return
    columns = st.columns(3, gap="small")
    for column, artifact in zip(columns, page_artifacts(page)):
        with column:
            st.download_button(
                artifact.label,
                data=artifact.data,
                file_name=artifact.filename,
                mime=artifact.mime,
                use_container_width=True,
                key=f"pdl_{page.page.index}_{artifact.filename}",
            )

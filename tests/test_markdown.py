"""Markdown cleanup, table extraction and HTML rendering."""

from __future__ import annotations

from dsocr.postprocess.grounding import crop_figures, parse_grounding
from dsocr.postprocess.markdown import (
    apply_latex_fixups,
    build_markdown,
    clean_output,
    extract_tables,
    merge_pages,
    strip_grounding,
    strip_special_tokens,
    to_html,
    to_plain_text,
)


class TestStripping:
    def test_removes_grounding_tags(self, grounded_output):
        cleaned = strip_grounding(grounded_output)
        assert "<|ref|>" not in cleaned and "<|det|>" not in cleaned

    def test_keeps_content_between_tags(self, grounded_output):
        assert "Quartalsbericht 2026" in strip_grounding(grounded_output)

    def test_removes_special_tokens(self, grounded_output):
        assert "<|end▁of▁sentence|>" not in strip_special_tokens(grounded_output)

    def test_latex_fixups(self):
        assert apply_latex_fixups(r"a \coloneqq b") == "a := b"


class TestBuildMarkdown:
    def test_figures_become_image_links(self, page_image, grounded_output):
        regions = parse_grounding(grounded_output)
        figures = crop_figures(page_image, regions)
        markdown = build_markdown(grounded_output, regions, figures)
        assert f"figures/{figures[0].filename}" in markdown

    def test_no_tags_survive(self, page_image, grounded_output):
        regions = parse_grounding(grounded_output)
        markdown = build_markdown(grounded_output, regions, crop_figures(page_image, regions))
        for token in ("<|ref|>", "<|det|>", "<|grounding|>", "<image>"):
            assert token not in markdown

    def test_headings_preserved(self, page_image, grounded_output):
        regions = parse_grounding(grounded_output)
        markdown = build_markdown(grounded_output, regions, crop_figures(page_image, regions))
        assert "# Quartalsbericht 2026" in markdown

    def test_collapses_blank_line_runs(self):
        assert "\n\n\n" not in build_markdown("a\n\n\n\n\nb", [], [])


class TestPlainText:
    def test_strips_markup(self, page_image, grounded_output):
        regions = parse_grounding(grounded_output)
        cleaned = clean_output(grounded_output, regions, crop_figures(page_image, regions))
        assert "#" not in cleaned.plain_text
        assert "<td>" not in cleaned.plain_text

    def test_keeps_words(self):
        assert "Hallo Welt" in to_plain_text("## **Hallo** Welt")

    def test_drops_image_links(self):
        assert "figures/" not in to_plain_text("![x](figures/a.png)\n\nText")

    def test_keeps_link_labels(self):
        assert to_plain_text("[Anhang](http://x)") == "Anhang"

    def test_word_count(self, page_image, grounded_output):
        regions = parse_grounding(grounded_output)
        cleaned = clean_output(grounded_output, regions, crop_figures(page_image, regions))
        assert cleaned.word_count > 5


class TestTables:
    def test_extracts_table(self, grounded_output):
        tables = extract_tables(grounded_output)
        assert len(tables) == 1
        assert tables[0].shape == (3, 2)

    def test_header_and_body(self, grounded_output):
        table = extract_tables(grounded_output)[0]
        assert table.header == ["Region", "Umsatz"]
        assert table.body == [["EMEA", "4.2 Mio"], ["APAC", "2.8 Mio"]]

    def test_csv_roundtrip(self, grounded_output):
        csv = extract_tables(grounded_output)[0].to_csv()
        assert "Region,Umsatz" in csv
        assert "EMEA" in csv

    def test_markdown_rendering(self, grounded_output):
        markdown = extract_tables(grounded_output)[0].to_markdown()
        assert markdown.splitlines()[1].startswith("| ---")

    def test_ragged_rows_are_padded(self):
        table = extract_tables("<table><tr><td>a</td><td>b</td></tr><tr><td>c</td></tr></table>")[0]
        assert "| c |  |" in table.to_markdown()

    def test_entities_are_unescaped(self):
        table = extract_tables("<table><tr><td>A &amp; B</td></tr></table>")[0]
        assert table.rows[0][0] == "A & B"

    def test_no_tables_found(self):
        assert extract_tables("nur Text") == []


class TestHtml:
    def test_is_standalone_document(self):
        html = to_html("# Titel\n\nAbsatz")
        assert html.startswith("<!doctype html>")
        assert "<style>" in html

    def test_renders_headings_and_paragraphs(self):
        html = to_html("# Titel\n\nAbsatz")
        assert "Titel" in html and "Absatz" in html

    def test_supports_dark_mode(self):
        assert "prefers-color-scheme: dark" in to_html("x")

    def test_title_is_escaped(self):
        assert "<script>" not in to_html("x", title="<script>alert(1)</script>")


class TestMergePages:
    def test_joins_with_separator(self):
        merged = merge_pages([("S1", "eins"), ("S2", "zwei")])
        assert "eins" in merged and "zwei" in merged and "---" in merged

    def test_skips_empty_pages(self):
        assert "S2" not in merge_pages([("S1", "eins"), ("S2", "   ")])

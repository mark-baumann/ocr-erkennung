"""Grounding parser, colours and overlay rendering."""

from __future__ import annotations

import pytest
from PIL import Image

from dsocr.postprocess.grounding import (
    COORD_SCALE,
    LABEL_PALETTE,
    BoundingBox,
    color_for_label,
    crop_figures,
    draw_layout_overlay,
    label_counts,
    parse_grounding,
)


class TestBoundingBox:
    def test_clamps_out_of_range(self):
        box = BoundingBox(-50, -10, 2000, 1500).clamped()
        assert (box.x1, box.y1, box.x2, box.y2) == (0, 0, COORD_SCALE, COORD_SCALE)

    def test_clamps_inverted_coordinates(self):
        box = BoundingBox(800, 600, 200, 100).clamped()
        assert box.x1 < box.x2 and box.y1 < box.y2

    def test_to_pixels_scales_from_999_grid(self):
        box = BoundingBox(0, 0, 999, 999)
        assert box.to_pixels(1000, 500) == (0, 0, 1000, 500)

    def test_area_is_fraction_of_page(self):
        assert BoundingBox(0, 0, 999, 999).area == pytest.approx(1.0, rel=1e-3)


class TestParseGrounding:
    def test_parses_all_regions(self, grounded_output):
        regions = parse_grounding(grounded_output)
        assert [r.label for r in regions] == ["title", "text", "table", "image", "footer"]

    def test_boxes_are_parsed(self, grounded_output):
        title = parse_grounding(grounded_output)[0]
        assert title.primary_box.as_list() == [40, 30, 950, 90]

    def test_text_is_attributed_to_region(self, grounded_output):
        regions = parse_grounding(grounded_output)
        assert "Quartalsbericht 2026" in regions[0].text
        assert "12 Prozent" in regions[1].text

    def test_raw_is_preserved_for_stripping(self, grounded_output):
        for region in parse_grounding(grounded_output):
            assert region.raw in grounded_output

    def test_no_grounding_yields_nothing(self):
        assert parse_grounding("Plain OCR output without any tags.") == []

    def test_flat_box_form_is_accepted(self):
        regions = parse_grounding("<|ref|>text<|/ref|><|det|>[10, 20, 30, 40]<|/det|>hi")
        assert regions[0].primary_box.as_list() == [10, 20, 30, 40]

    def test_multiple_boxes_per_region(self):
        regions = parse_grounding("<|ref|>text<|/ref|><|det|>[[1,2,3,4],[5,6,7,8]]<|/det|>x")
        assert len(regions[0].boxes) == 2

    def test_malformed_payload_is_skipped_not_raised(self):
        assert parse_grounding("<|ref|>text<|/ref|><|det|>not-a-list<|/det|>x") == []

    def test_parser_does_not_evaluate_code(self):
        """Upstream uses eval() here; a literal parser must refuse expressions."""
        hostile = "<|ref|>text<|/ref|><|det|>__import__('os').system('true')<|/det|>x"
        assert parse_grounding(hostile) == []

    def test_regions_are_ordered(self, grounded_output):
        assert [r.order for r in parse_grounding(grounded_output)] == [0, 1, 2, 3, 4]

    def test_label_counts(self, grounded_output):
        counts = label_counts(parse_grounding(grounded_output))
        assert counts == {"footer": 1, "image": 1, "table": 1, "text": 1, "title": 1}


class TestColors:
    def test_known_labels_use_curated_palette(self):
        assert color_for_label("title") == LABEL_PALETTE["title"]

    def test_case_insensitive(self):
        assert color_for_label("TABLE") == color_for_label("table")

    def test_deterministic_for_unknown_labels(self):
        assert color_for_label("weird_thing") == color_for_label("weird_thing")

    def test_common_labels_are_visually_distinct(self):
        """text/title/table must not collapse into one colour on the overlay."""
        colors = [color_for_label(label) for label in ("text", "title", "table", "image")]
        for i, a in enumerate(colors):
            for b in colors[i + 1 :]:
                distance = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
                assert distance > 90, f"{a} and {b} are too close"

    def test_unknown_label_avoids_curated_hues(self):
        assert color_for_label("mystery") not in LABEL_PALETTE.values()


class TestOverlay:
    def test_returns_same_size(self, page_image, grounded_output):
        regions = parse_grounding(grounded_output)
        assert draw_layout_overlay(page_image, regions).size == page_image.size

    def test_overlay_modifies_pixels(self, page_image, grounded_output):
        regions = parse_grounding(grounded_output)
        overlay = draw_layout_overlay(page_image, regions)
        assert overlay.tobytes() != page_image.convert("RGB").tobytes()

    def test_filter_hides_regions(self, page_image, grounded_output):
        regions = parse_grounding(grounded_output)
        everything = draw_layout_overlay(page_image, regions)
        only_titles = draw_layout_overlay(page_image, regions, visible_labels=["title"])
        assert everything.tobytes() != only_titles.tobytes()

    def test_empty_regions_is_a_noop(self, page_image):
        overlay = draw_layout_overlay(page_image, [])
        assert overlay.tobytes() == page_image.convert("RGB").tobytes()

    def test_survives_degenerate_boxes(self, page_image):
        regions = parse_grounding("<|ref|>text<|/ref|><|det|>[[500,500,500,500]]<|/det|>x")
        draw_layout_overlay(page_image, regions)  # must not raise


class TestCropFigures:
    def test_crops_only_figure_regions(self, page_image, grounded_output):
        regions = parse_grounding(grounded_output)
        figures = crop_figures(page_image, regions)
        assert len(figures) == 1
        assert figures[0].label == "image"

    def test_crop_geometry(self, page_image, grounded_output):
        regions = parse_grounding(grounded_output)
        figure = crop_figures(page_image, regions)[0]
        x1, y1, x2, y2 = figure.box_pixels
        assert figure.image.size == (x2 - x1, y2 - y1)

    def test_filenames_are_unique(self, page_image):
        text = "".join(
            f"<|ref|>image<|/ref|><|det|>[[{i*100}, 10, {i*100+90}, 200]]<|/det|>" for i in range(1, 5)
        )
        figures = crop_figures(page_image, parse_grounding(text))
        assert len({f.filename for f in figures}) == len(figures)

    def test_tiny_regions_are_skipped(self, page_image):
        regions = parse_grounding("<|ref|>image<|/ref|><|det|>[[10,10,11,11]]<|/det|>x")
        assert crop_figures(page_image, regions, min_area=0.01) == []

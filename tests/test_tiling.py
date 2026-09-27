"""Tiling geometry and vision-token accounting."""

from __future__ import annotations

import pytest
from PIL import Image

from dsocr.config import MODES, get_mode, token_grid
from dsocr.preprocess.tiling import (
    candidate_ratios,
    count_tiles,
    dynamic_preprocess,
    plan_tiling,
    tile_preview,
)


class TestTokenGrid:
    @pytest.mark.parametrize(
        "size,expected",
        [(512, 8), (640, 10), (1024, 16), (1280, 20)],
    )
    def test_grid_matches_paper(self, size, expected):
        assert token_grid(size) == expected

    @pytest.mark.parametrize(
        "mode_key,expected",
        [("tiny", 64), ("small", 100), ("base", 256), ("large", 400)],
    )
    def test_nominal_tokens_match_readme(self, mode_key, expected):
        """The README quotes 64/100/256/400 vision tokens for these modes."""
        assert MODES[mode_key].nominal_vision_tokens == expected


class TestCandidateRatios:
    def test_respects_bounds(self):
        for cols, rows in candidate_ratios(2, 6):
            assert 2 <= cols * rows <= 6

    def test_sorted_by_area(self):
        areas = [c * r for c, r in candidate_ratios(1, 9)]
        assert areas == sorted(areas)

    def test_empty_when_bounds_impossible(self):
        assert candidate_ratios(10, 5) == []


class TestCountTiles:
    def test_portrait_page_prefers_taller_grid(self):
        cols, rows = count_tiles(1240, 1754, min_num=2, max_num=6)
        assert rows >= cols

    def test_landscape_page_prefers_wider_grid(self):
        cols, rows = count_tiles(1754, 1240, min_num=2, max_num=6)
        assert cols >= rows

    def test_degenerate_size(self):
        assert count_tiles(0, 0) == (1, 1)


class TestDynamicPreprocess:
    def test_tile_count_matches_grid(self):
        image = Image.new("RGB", (1240, 1754), "white")
        tiles, grid = dynamic_preprocess(image, min_num=2, max_num=6, image_size=640)
        assert len(tiles) == grid[0] * grid[1]

    def test_tiles_are_square_and_sized(self):
        image = Image.new("RGB", (1240, 1754), "white")
        tiles, _ = dynamic_preprocess(image, min_num=2, max_num=6, image_size=640)
        assert all(t.size == (640, 640) for t in tiles)

    def test_thumbnail_appended(self):
        image = Image.new("RGB", (1240, 1754), "white")
        tiles, grid = dynamic_preprocess(image, min_num=2, max_num=6, image_size=640, use_thumbnail=True)
        assert len(tiles) == grid[0] * grid[1] + 1


class TestPlanTiling:
    def test_non_crop_mode_never_tiles(self):
        plan = plan_tiling((2000, 3000), get_mode("base"))
        assert not plan.tiled
        assert plan.visual_tokens == 256

    def test_small_image_bypasses_tiling_even_in_gundam(self):
        """Upstream skips tiling for images that already fit in 640x640."""
        plan = plan_tiling((500, 400), get_mode("gundam"))
        assert not plan.tiled
        assert plan.visual_tokens == 256

    def test_gundam_adds_local_tiles(self):
        plan = plan_tiling((1240, 1754), get_mode("gundam"), min_num=2, max_num=6)
        cols, rows = plan.grid
        assert plan.tiled
        # 16x16 global + (10*cols)x(10*rows) local
        assert plan.visual_tokens == 256 + (10 * cols) * (10 * rows)

    def test_sequence_tokens_exceed_visual_tokens(self):
        """Row separators are real context cost and must be counted separately."""
        plan = plan_tiling((1240, 1754), get_mode("gundam"))
        assert plan.sequence_tokens > plan.visual_tokens

    def test_base_sequence_token_formula(self):
        plan = plan_tiling((2000, 2000), get_mode("base"))
        assert plan.sequence_tokens == (16 + 1) * 16 + 1

    def test_compression_ratio(self):
        plan = plan_tiling((1000, 1000), get_mode("base"))
        assert plan.compression_ratio == pytest.approx(1_000_000 / 256)

    def test_tighter_budget_lowers_token_cost(self):
        loose = plan_tiling((1240, 1754), get_mode("gundam"), min_num=2, max_num=9)
        tight = plan_tiling((1240, 1754), get_mode("gundam"), min_num=1, max_num=2)
        assert tight.visual_tokens <= loose.visual_tokens

    def test_describe(self):
        assert "Kacheln" in plan_tiling((1240, 1754), get_mode("gundam")).describe()
        assert "Kacheln" not in plan_tiling((800, 800), get_mode("base")).describe()


class TestTilePreview:
    def test_returns_same_size(self, page_image):
        plan = plan_tiling(page_image.size, get_mode("gundam"))
        assert tile_preview(page_image, plan).size == page_image.size

    def test_untiled_preview_is_unmodified(self, small_image):
        plan = plan_tiling(small_image.size, get_mode("base"))
        preview = tile_preview(small_image, plan)
        assert preview.tobytes() == small_image.convert("RGB").tobytes()

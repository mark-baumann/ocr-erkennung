"""Dynamic tiling ("Gundam" mode) and vision-token accounting.

The tiling maths is adopted from the upstream DeepSeek-OCR repository
(``DeepSeek-OCR-vllm/process/image_process.py``). Two things changed:

* the functions no longer read module level globals — bounds are arguments, so
  the UI can expose them as sliders;
* the token accounting from ``tokenize_with_images`` is factored out into
  :func:`plan_tiling`, which lets us show the compression ratio *before*
  spending a single GPU second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from PIL import Image

from ..config import ResolutionMode, token_grid

#: Images at or below this size skip tiling entirely (upstream behaviour).
TILING_BYPASS_SIZE = 640


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """Pick the (cols, rows) grid whose aspect ratio best matches the image.

    Adopted verbatim in behaviour from upstream: ties are broken in favour of
    the larger grid when the source image has enough area to justify it.
    """
    best_ratio_diff = float("inf")
    best_ratio: Tuple[int, int] = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def candidate_ratios(min_num: int, max_num: int) -> List[Tuple[int, int]]:
    """All (cols, rows) grids whose tile count falls within ``[min_num, max_num]``."""
    ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    return sorted(ratios, key=lambda x: (x[0] * x[1], x[0], x[1]))


def count_tiles(
    orig_width: int,
    orig_height: int,
    *,
    min_num: int = 2,
    max_num: int = 6,
    image_size: int = 640,
) -> Tuple[int, int]:
    """Return the (cols, rows) tile grid chosen for an image of this size."""
    if orig_height <= 0 or orig_width <= 0:
        return (1, 1)
    target_ratios = candidate_ratios(min_num, max_num)
    if not target_ratios:
        return (1, 1)
    return find_closest_aspect_ratio(
        orig_width / orig_height, target_ratios, orig_width, orig_height, image_size
    )


def dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 2,
    max_num: int = 6,
    image_size: int = 640,
    use_thumbnail: bool = False,
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """Split ``image`` into a grid of ``image_size`` tiles.

    Returns the tiles in reading order plus the chosen ``(cols, rows)`` grid.
    """
    orig_width, orig_height = image.size
    target_aspect_ratio = count_tiles(
        orig_width, orig_height, min_num=min_num, max_num=max_num, image_size=image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    cols = target_width // image_size
    processed_images = []
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images, target_aspect_ratio


# ---------------------------------------------------------------------------
# Token accounting (extension)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TilingPlan:
    """What the encoder will do with one page, and what it costs in tokens.

    ``visual_tokens`` counts only latent image queries — this is the number
    quoted in the paper (64 / 100 / 256 / 400). ``sequence_tokens`` additionally
    counts the row separators that ``tokenize_with_images`` interleaves, i.e.
    the real context footprint.
    """

    mode_key: str
    source_size: Tuple[int, int]
    grid: Tuple[int, int]
    tiled: bool
    global_grid: int
    local_grid: int
    visual_tokens: int
    sequence_tokens: int

    @property
    def num_tiles(self) -> int:
        return self.grid[0] * self.grid[1] if self.tiled else 0

    @property
    def source_pixels(self) -> int:
        return self.source_size[0] * self.source_size[1]

    @property
    def compression_ratio(self) -> float:
        """Source pixels per vision token — the paper's optical compression lens."""
        if self.visual_tokens == 0:
            return 0.0
        return self.source_pixels / self.visual_tokens

    def describe(self) -> str:
        if self.tiled:
            cols, rows = self.grid
            return f"1 globale Ansicht + {cols}×{rows} Kacheln"
        return "1 globale Ansicht"


def plan_tiling(
    size: Tuple[int, int],
    mode: ResolutionMode,
    *,
    min_num: int = 2,
    max_num: int = 6,
) -> TilingPlan:
    """Compute the tiling grid and token cost for an image of ``size``.

    Mirrors the branch logic of ``DeepseekOCRProcessor.tokenize_with_images``:
    images that already fit in 640x640 are never tiled, and tiling only happens
    when the mode enables ``crop_mode``.
    """
    width, height = size
    tiled = False
    grid = (1, 1)

    if mode.crop_mode and not (width <= TILING_BYPASS_SIZE and height <= TILING_BYPASS_SIZE):
        grid = count_tiles(width, height, min_num=min_num, max_num=max_num, image_size=mode.image_size)
        tiled = grid[0] > 1 or grid[1] > 1

    nq_base = token_grid(mode.base_size)
    nq_local = token_grid(mode.image_size)

    # Global view: nq_base rows of (nq_base queries + 1 newline), plus a trailing newline.
    visual = nq_base**2
    sequence = (nq_base + 1) * nq_base + 1

    if tiled:
        cols, rows = grid
        visual += (nq_local * cols) * (nq_local * rows)
        sequence += (nq_local * cols + 1) * (nq_local * rows)

    return TilingPlan(
        mode_key=mode.key,
        source_size=(width, height),
        grid=grid,
        tiled=tiled,
        global_grid=nq_base,
        local_grid=nq_local,
        visual_tokens=visual,
        sequence_tokens=sequence,
    )


def tile_preview(
    image: Image.Image,
    plan: TilingPlan,
    *,
    line_width: int = 3,
    color: Tuple[int, int, int] = (255, 92, 0),
) -> Image.Image:
    """Draw the tile grid onto a copy of ``image`` for the UI preview."""
    from PIL import ImageDraw

    preview = image.convert("RGB").copy()
    if not plan.tiled:
        return preview

    cols, rows = plan.grid
    width, height = preview.size
    draw = ImageDraw.Draw(preview)
    for c in range(1, cols):
        x = int(width * c / cols)
        draw.line([(x, 0), (x, height)], fill=color, width=line_width)
    for r in range(1, rows):
        y = int(height * r / rows)
        draw.line([(0, y), (width, y)], fill=color, width=line_width)
    draw.rectangle([0, 0, width - 1, height - 1], outline=color, width=line_width)
    return preview

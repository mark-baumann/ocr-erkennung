"""Turning raw model output into structured, exportable results."""

from .export import Artifact, all_artifacts, bundle_artifact, page_artifacts
from .grounding import (
    BoundingBox,
    CroppedFigure,
    LayoutRegion,
    color_for_label,
    crop_figures,
    draw_layout_overlay,
    label_counts,
    parse_grounding,
)
from .markdown import CleanedOutput, ExtractedTable, clean_output, extract_tables, to_html, to_plain_text

__all__ = [
    "Artifact",
    "BoundingBox",
    "CleanedOutput",
    "CroppedFigure",
    "ExtractedTable",
    "LayoutRegion",
    "all_artifacts",
    "bundle_artifact",
    "clean_output",
    "color_for_label",
    "crop_figures",
    "draw_layout_overlay",
    "extract_tables",
    "label_counts",
    "page_artifacts",
    "parse_grounding",
    "to_html",
    "to_plain_text",
]

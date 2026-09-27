"""The orchestration layer: pages in, structured results out.

Upstream ships three near-duplicate scripts (image / pdf / eval-batch) that each
re-implement load → preprocess → generate → postprocess. This module has that
flow exactly once, so the Streamlit UI, the notebook, CLI and tests all take the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from PIL import Image

from .config import SETTINGS, PromptPreset, ResolutionMode, Settings
from .engines import OcrEngine, OcrRequest, OcrResult
from .postprocess.grounding import CroppedFigure, LayoutRegion, crop_figures, label_counts, parse_grounding
from .postprocess.markdown import CleanedOutput, ExtractedTable, clean_output, extract_tables
from .preprocess.loader import Page
from .preprocess.tiling import TilingPlan, plan_tiling

#: ``(page_index, total_pages, label)`` — used by interactive callers for progress.
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class PageResult:
    """Everything produced for one page."""

    page: Page
    plan: TilingPlan
    result: OcrResult
    regions: List[LayoutRegion] = field(default_factory=list)
    figures: List[CroppedFigure] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    cleaned: Optional[CleanedOutput] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def markdown(self) -> str:
        return self.cleaned.markdown if self.cleaned else ""

    @property
    def plain_text(self) -> str:
        return self.cleaned.plain_text if self.cleaned else ""

    @property
    def label(self) -> str:
        return self.page.label

    def overlay(self, **kwargs) -> Image.Image:
        from .postprocess.grounding import draw_layout_overlay

        return draw_layout_overlay(self.page.image, self.regions, **kwargs)

    def to_dict(self) -> Dict[str, object]:
        return {
            "source": self.page.source_name,
            "page_number": self.page.page_number,
            "size": list(self.page.size),
            "backend": self.result.backend,
            "latency_s": round(self.result.latency_s, 3),
            "tiling": {
                "mode": self.plan.mode_key,
                "grid": list(self.plan.grid),
                "tiled": self.plan.tiled,
                "visual_tokens": self.plan.visual_tokens,
                "sequence_tokens": self.plan.sequence_tokens,
                "compression_ratio": round(self.plan.compression_ratio, 1),
            },
            "regions": [r.to_dict(self.page.size) for r in self.regions],
            "tables": [{"index": t.index, "shape": list(t.shape), "rows": t.rows} for t in self.tables],
            "markdown": self.markdown,
            "error": self.error,
        }


@dataclass
class RunSummary:
    """Aggregate telemetry over a whole run."""

    pages: int = 0
    failed: int = 0
    total_latency_s: float = 0.0
    visual_tokens: int = 0
    sequence_tokens: int = 0
    source_pixels: int = 0
    regions: int = 0
    backend: str = ""
    is_real: bool = True

    @property
    def avg_latency_s(self) -> float:
        return self.total_latency_s / self.pages if self.pages else 0.0

    @property
    def compression_ratio(self) -> float:
        """Source pixels per vision token across the whole run."""
        return self.source_pixels / self.visual_tokens if self.visual_tokens else 0.0


@dataclass
class RunResult:
    """The full output of one pipeline run."""

    pages: List[PageResult] = field(default_factory=list)
    summary: RunSummary = field(default_factory=RunSummary)
    prompt: str = ""
    mode_key: str = ""

    @property
    def successful(self) -> List[PageResult]:
        return [p for p in self.pages if p.ok]

    def combined_markdown(self) -> str:
        from .postprocess.markdown import merge_pages

        return merge_pages([(p.label, p.markdown) for p in self.successful])

    def combined_text(self) -> str:
        return "\n\n".join(p.plain_text for p in self.successful if p.plain_text)

    def all_regions(self) -> List[LayoutRegion]:
        return [region for page in self.successful for region in page.regions]

    def region_counts(self) -> Dict[str, int]:
        return label_counts(self.all_regions())

    def all_tables(self) -> List[ExtractedTable]:
        return [table for page in self.successful for table in page.tables]

    def to_dict(self) -> Dict[str, object]:
        return {
            "prompt": self.prompt,
            "mode": self.mode_key,
            "backend": self.summary.backend,
            "synthetic": not self.summary.is_real,
            "summary": {
                "pages": self.summary.pages,
                "failed": self.summary.failed,
                "total_latency_s": round(self.summary.total_latency_s, 3),
                "avg_latency_s": round(self.summary.avg_latency_s, 3),
                "visual_tokens": self.summary.visual_tokens,
                "sequence_tokens": self.summary.sequence_tokens,
                "compression_ratio": round(self.summary.compression_ratio, 1),
                "regions": self.summary.regions,
            },
            "pages": [p.to_dict() for p in self.pages],
        }


def build_requests(
    pages: Sequence[Page],
    prompt: str,
    mode: ResolutionMode,
    settings: Optional[Settings] = None,
) -> List[OcrRequest]:
    settings = settings or SETTINGS
    lo, hi = settings.crop_bounds()
    return [
        OcrRequest(
            image=page.image,
            prompt=prompt,
            mode=mode,
            max_new_tokens=settings.max_new_tokens,
            min_crops=lo,
            max_crops=hi,
        )
        for page in pages
    ]


def postprocess_page(
    page: Page,
    plan: TilingPlan,
    result: OcrResult,
    *,
    figure_dir: str = "figures",
) -> PageResult:
    """Parse one raw model response into structured output."""
    page_result = PageResult(page=page, plan=plan, result=result)

    if not result.text.strip():
        # A backend that raised records the reason in metadata — surface that
        # instead of the generic "no text" message, so failures stay debuggable.
        page_result.error = str(result.metadata.get("error") or "Das Modell hat keinen Text zurückgegeben.")
        page_result.cleaned = clean_output("", [], [], figure_dir=figure_dir)
        return page_result

    regions = parse_grounding(result.text)
    figures = crop_figures(page.image, regions)
    page_result.regions = regions
    page_result.figures = figures
    page_result.cleaned = clean_output(result.text, regions, figures, figure_dir=figure_dir)
    page_result.tables = extract_tables(result.text)
    return page_result


def run(
    pages: Sequence[Page],
    engine: OcrEngine,
    prompt: str,
    mode: ResolutionMode,
    *,
    settings: Optional[Settings] = None,
    on_progress: Optional[ProgressCallback] = None,
    batch: bool = True,
) -> RunResult:
    """Run the full pipeline over ``pages``.

    ``batch=True`` lets a backend that supports real batching (vLLM) schedule
    every page at once; otherwise pages run one by one so progress stays live.
    """
    settings = settings or SETTINGS
    pages = list(pages)
    lo, hi = settings.crop_bounds()

    plans = [plan_tiling(page.size, mode, min_num=lo, max_num=hi) for page in pages]
    requests = build_requests(pages, prompt, mode, settings)

    run_result = RunResult(prompt=prompt, mode_key=mode.key)
    run_result.summary.backend = engine.key
    run_result.summary.is_real = engine.is_real

    if not pages:
        return run_result

    use_batch = batch and type(engine).infer_batch is not OcrEngine.infer_batch and len(pages) > 1

    if use_batch:
        if on_progress is not None:
            on_progress(0, len(pages), f"{len(pages)} Seiten im Batch …")

        def _page_done(index: int, _result: OcrResult) -> None:
            if on_progress is not None:
                on_progress(index + 1, len(pages), pages[index].label)

        try:
            results = engine.infer_batch(requests, on_page=_page_done)
        except Exception as exc:
            results = [OcrResult(text="", latency_s=0.0, backend=engine.key) for _ in pages]
            for page, plan, result in zip(pages, plans, results):
                failed = PageResult(page=page, plan=plan, result=result, error=str(exc))
                failed.cleaned = clean_output("", [], [])
                run_result.pages.append(failed)
            _finalize(run_result)
            return run_result
    else:
        results = []
        for index, request in enumerate(requests):
            if on_progress is not None:
                on_progress(index, len(pages), pages[index].label)
            try:
                results.append(engine.infer(request))
            except Exception as exc:
                results.append(
                    OcrResult(text="", latency_s=0.0, backend=engine.key, metadata={"error": str(exc)})
                )
        if on_progress is not None:
            on_progress(len(pages), len(pages), "fertig")

    for page, plan, result in zip(pages, plans, results):
        page_result = postprocess_page(page, plan, result)
        if page_result.ok and result.metadata.get("error"):
            page_result.error = str(result.metadata["error"])
        run_result.pages.append(page_result)

    _finalize(run_result)
    return run_result


def _finalize(run_result: RunResult) -> None:
    summary = run_result.summary
    summary.pages = len(run_result.pages)
    summary.failed = sum(1 for p in run_result.pages if not p.ok)
    summary.total_latency_s = sum(p.result.latency_s for p in run_result.pages)
    summary.visual_tokens = sum(p.plan.visual_tokens for p in run_result.pages)
    summary.sequence_tokens = sum(p.plan.sequence_tokens for p in run_result.pages)
    summary.source_pixels = sum(p.plan.source_pixels for p in run_result.pages)
    summary.regions = sum(len(p.regions) for p in run_result.pages)


def resolve_prompt(preset: PromptPreset, query: str = "") -> str:
    """Render a preset, guarding against a custom prompt without ``<image>``."""
    prompt = preset.render(query)
    if preset.key == "custom" and "<image>" not in prompt:
        prompt = f"<image>\n{prompt.lstrip()}"
    return prompt

"""Pipeline orchestration, engine registry, loader and exports."""

from __future__ import annotations

import io
import json
import zipfile

import pytest
from PIL import Image

from dsocr import pipeline
from dsocr.config import PROMPTS, Settings, get_mode
from dsocr.engines import AUTO_ORDER, EngineUnavailable, create_engine, probe_all, resolve_backend
from dsocr.engines.base import OcrEngine, OcrRequest, OcrResult
from dsocr.engines.demo_engine import DemoEngine
from dsocr.postprocess.export import all_artifacts, bundle_artifact, page_artifacts
from dsocr.preprocess.loader import (
    Page,
    UnsupportedDocument,
    downscale,
    load_document,
    load_image_bytes,
    suffix_of,
)


class FakeEngine(OcrEngine):
    """Returns canned output so the pipeline can be tested without a model."""

    key = "fake"
    label = "Fake"

    def __init__(self, text: str = "", fail: bool = False):
        super().__init__(Settings())
        self._text = text
        self._fail = fail
        self.calls = 0

    @classmethod
    def probe(cls, settings):  # pragma: no cover - unused
        from dsocr.engines.base import BackendInfo

        return BackendInfo(cls.key, cls.label, True, "test")

    def infer(self, request, on_token=None):
        self.calls += 1
        if self._fail:
            raise RuntimeError("boom")
        return OcrResult(text=self._text, latency_s=0.01, backend=self.key)


class BatchEngine(FakeEngine):
    key = "batch"

    def __init__(self, text: str = ""):
        super().__init__(text)
        self.batch_calls = 0

    def infer_batch(self, requests, on_page=None):
        self.batch_calls += 1
        results = [OcrResult(text=self._text, latency_s=0.01, backend=self.key) for _ in requests]
        for index, result in enumerate(results):
            if on_page is not None:
                on_page(index, result)
        return results


@pytest.fixture
def pages(page_image):
    return [Page(index=0, image=page_image, source_name="test.png")]


@pytest.fixture
def run_result(pages, grounded_output):
    return pipeline.run(pages, FakeEngine(grounded_output), "<image>\ntest", get_mode("gundam"))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestLoader:
    def test_suffix_of(self):
        assert suffix_of("a/b/Scan.PDF") == ".pdf"
        assert suffix_of("noext") == ""

    def test_loads_png(self, page_image):
        buffer = io.BytesIO()
        page_image.save(buffer, format="PNG")
        document = load_document("x.png", buffer.getvalue())
        assert len(document) == 1
        assert document.pages[0].image.mode == "RGB"

    def test_rejects_unknown_format(self):
        with pytest.raises(UnsupportedDocument):
            load_document("x.docx", b"data")

    def test_rejects_corrupt_image(self):
        with pytest.raises(UnsupportedDocument):
            load_document("x.png", b"not-an-image")

    def test_page_labels(self, page_image):
        buffer = io.BytesIO()
        page_image.save(buffer, format="PNG")
        page = load_document("scan.png", buffer.getvalue()).pages[0]
        assert page.label == "scan.png · Seite 1"

    def test_converts_to_rgb(self):
        buffer = io.BytesIO()
        Image.new("L", (50, 50), 128).save(buffer, format="PNG")
        assert load_image_bytes(buffer.getvalue()).mode == "RGB"

    def test_downscale_caps_long_edge(self, page_image):
        assert max(downscale(page_image, 800).size) == 800

    def test_downscale_leaves_small_images(self, small_image):
        assert downscale(small_image, 5000).size == small_image.size


class TestPdfLoader:
    @staticmethod
    def _pdf(pages: int = 3) -> bytes:
        fitz = pytest.importorskip("fitz")
        doc = fitz.open()
        for i in range(pages):
            page = doc.new_page()
            page.insert_text((72, 144), f"Seite {i + 1}")
        data = doc.tobytes()
        doc.close()
        return data

    def test_expands_all_pages(self):
        document = load_document("d.pdf", self._pdf(3))
        assert len(document) == 3
        assert document.is_pdf

    def test_page_numbers_are_one_based(self):
        document = load_document("d.pdf", self._pdf(3))
        assert [p.page_number for p in document.pages] == [1, 2, 3]

    def test_max_pages_is_enforced(self):
        assert len(load_document("d.pdf", self._pdf(5), max_pages=2)) == 2

    def test_page_selection(self):
        document = load_document("d.pdf", self._pdf(5), page_selection=[1, 3])
        assert [p.page_number for p in document.pages] == [2, 4]

    def test_dpi_changes_raster_size(self):
        low = load_document("d.pdf", self._pdf(1), dpi=96).pages[0]
        high = load_document("d.pdf", self._pdf(1), dpi=200).pages[0]
        assert high.size[0] > low.size[0]


# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_probe_covers_all_backends(self):
        assert {i.key for i in probe_all()} == set(AUTO_ORDER)

    def test_probe_never_raises(self):
        for info in probe_all():
            assert isinstance(info.detail, str) and info.detail

    def test_auto_resolves_to_something(self):
        assert resolve_backend("auto") in AUTO_ORDER

    def test_demo_is_always_last_resort(self):
        assert AUTO_ORDER[-1] == DemoEngine.key

    def test_unknown_backend_raises(self):
        with pytest.raises(EngineUnavailable):
            resolve_backend("gpt-ocr")

    def test_auto_raises_when_everything_disabled(self):
        settings = Settings()
        settings.allow_demo_engine = False
        # vLLM/transformers are unavailable in CI, so disabling demo leaves nothing.
        if any(i.available for i in probe_all(settings)):
            pytest.skip("Ein echtes Backend ist verfügbar")
        with pytest.raises(EngineUnavailable):
            resolve_backend("auto", settings)

    def test_create_engine_returns_instance(self):
        assert isinstance(create_engine("auto"), OcrEngine)

    def test_demo_engine_is_flagged_synthetic(self):
        assert DemoEngine.is_real is False


class TestDemoEngine:
    def test_produces_grounded_output(self, page_image):
        result = DemoEngine(Settings()).infer(
            OcrRequest(image=page_image, prompt="<image>\n<|grounding|>x", mode=get_mode("gundam"))
        )
        assert "<|ref|>" in result.text
        assert "<|det|>" in result.text

    def test_omits_grounding_when_not_requested(self, page_image):
        result = DemoEngine(Settings()).infer(
            OcrRequest(image=page_image, prompt="<image>\nFree OCR.", mode=get_mode("base"))
        )
        assert "<|ref|>" not in result.text

    def test_output_is_parseable(self, page_image):
        from dsocr.postprocess.grounding import parse_grounding

        result = DemoEngine(Settings()).infer(
            OcrRequest(image=page_image, prompt="<image>\n<|grounding|>x", mode=get_mode("gundam"))
        )
        assert parse_grounding(result.text)

    def test_flags_itself_as_demo(self, page_image):
        result = DemoEngine(Settings()).infer(
            OcrRequest(image=page_image, prompt="<image>\nFree OCR.", mode=get_mode("base"))
        )
        assert "Demo-Modus" in result.text

    def test_handles_blank_page(self):
        blank = Image.new("RGB", (600, 800), "white")
        result = DemoEngine(Settings()).infer(
            OcrRequest(image=blank, prompt="<image>\nFree OCR.", mode=get_mode("base"))
        )
        assert isinstance(result.text, str)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_every_preset_renders(self):
        for preset in PROMPTS.values():
            assert pipeline.resolve_prompt(preset, "Rechnungsnummer")

    def test_grounding_presets_carry_the_token(self):
        for preset in PROMPTS.values():
            if preset.grounding and not preset.needs_query:
                assert "<|grounding|>" in preset.template

    def test_locate_substitutes_query(self):
        prompt = pipeline.resolve_prompt(PROMPTS["locate"], "Gesamtsumme")
        assert "<|ref|>Gesamtsumme<|/ref|>" in prompt

    def test_custom_prompt_gets_image_token(self):
        prompt = pipeline.resolve_prompt(PROMPTS["custom"], "Lies das vor.")
        assert prompt.startswith("<image>")

    def test_custom_prompt_keeps_existing_image_token(self):
        prompt = pipeline.resolve_prompt(PROMPTS["custom"], "<image>\nFree OCR.")
        assert prompt.count("<image>") == 1


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestRun:
    def test_produces_one_result_per_page(self, pages, grounded_output):
        result = pipeline.run(pages, FakeEngine(grounded_output), "<image>\nx", get_mode("base"))
        assert len(result.pages) == len(pages)

    def test_parses_regions(self, run_result):
        assert run_result.summary.regions == 5

    def test_summary_aggregates_tokens(self, run_result):
        assert run_result.summary.visual_tokens == run_result.pages[0].plan.visual_tokens

    def test_compression_ratio_is_positive(self, run_result):
        assert run_result.summary.compression_ratio > 0

    def test_extracts_tables(self, run_result):
        assert len(run_result.all_tables()) == 1

    def test_crops_figures(self, run_result):
        assert len(run_result.pages[0].figures) == 1

    def test_progress_callback_reaches_completion(self, pages, grounded_output):
        seen = []
        pipeline.run(
            pages,
            FakeEngine(grounded_output),
            "<image>\nx",
            get_mode("base"),
            on_progress=lambda done, total, label: seen.append((done, total)),
        )
        assert seen[-1] == (len(pages), len(pages))

    def test_empty_page_list(self):
        result = pipeline.run([], FakeEngine("x"), "<image>\nx", get_mode("base"))
        assert result.pages == [] and result.summary.pages == 0

    def test_empty_model_output_is_an_error(self, pages):
        result = pipeline.run(pages, FakeEngine(""), "<image>\nx", get_mode("base"))
        assert result.pages[0].error is not None
        assert result.successful == []

    def test_engine_exception_is_captured_per_page(self, pages):
        result = pipeline.run(pages, FakeEngine(fail=True), "<image>\nx", get_mode("base"))
        assert result.summary.failed == 1
        assert "boom" in result.pages[0].error

    def test_one_failure_does_not_sink_the_run(self, page_image, grounded_output):
        class Flaky(FakeEngine):
            def infer(self, request, on_token=None):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("nope")
                return OcrResult(text=grounded_output, latency_s=0.01, backend=self.key)

        two_pages = [
            Page(index=0, image=page_image, source_name="a.png"),
            Page(index=1, image=page_image, source_name="b.png"),
        ]
        result = pipeline.run(two_pages, Flaky(), "<image>\nx", get_mode("base"))
        assert result.summary.failed == 1 and len(result.successful) == 1

    def test_records_backend_authenticity(self, pages, grounded_output):
        result = pipeline.run(pages, DemoEngine(Settings()), "<image>\nx", get_mode("base"))
        assert result.summary.is_real is False
        real = pipeline.run(pages, FakeEngine(grounded_output), "<image>\nx", get_mode("base"))
        assert real.summary.is_real is True


class TestBatching:
    def test_batch_engine_gets_one_call(self, page_image, grounded_output):
        engine = BatchEngine(grounded_output)
        four = [Page(index=i, image=page_image, source_name=f"p{i}.png") for i in range(4)]
        pipeline.run(four, engine, "<image>\nx", get_mode("base"), batch=True)
        assert engine.batch_calls == 1 and engine.calls == 0

    def test_batch_disabled_falls_back_to_loop(self, page_image, grounded_output):
        engine = BatchEngine(grounded_output)
        four = [Page(index=i, image=page_image, source_name=f"p{i}.png") for i in range(4)]
        pipeline.run(four, engine, "<image>\nx", get_mode("base"), batch=False)
        assert engine.batch_calls == 0 and engine.calls == 4

    def test_single_page_never_batches(self, pages, grounded_output):
        engine = BatchEngine(grounded_output)
        pipeline.run(pages, engine, "<image>\nx", get_mode("base"), batch=True)
        assert engine.batch_calls == 0

    def test_batch_failure_marks_every_page(self, page_image):
        class BrokenBatch(BatchEngine):
            def infer_batch(self, requests, on_page=None):
                raise RuntimeError("engine died")

        four = [Page(index=i, image=page_image, source_name=f"p{i}.png") for i in range(4)]
        result = pipeline.run(four, BrokenBatch(), "<image>\nx", get_mode("base"), batch=True)
        assert result.summary.failed == 4
        assert all("engine died" in p.error for p in result.pages)


class TestSerialization:
    def test_to_dict_is_json_serializable(self, run_result):
        assert json.loads(json.dumps(run_result.to_dict()))

    def test_page_dict_has_pixel_boxes(self, run_result):
        region = run_result.to_dict()["pages"][0]["regions"][0]
        assert "boxes_pixels" in region and "boxes_normalized" in region

    def test_synthetic_flag_in_export(self, pages):
        result = pipeline.run(pages, DemoEngine(Settings()), "<image>\nx", get_mode("base"))
        assert result.to_dict()["synthetic"] is True


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_artifact_set(self, run_result):
        names = {a.filename for a in all_artifacts(run_result, stem="doc")}
        assert names == {"doc.md", "doc.txt", "doc.html", "doc.json"}

    def test_artifacts_are_non_empty(self, run_result):
        for artifact in all_artifacts(run_result):
            assert artifact.data

    def test_json_artifact_parses(self, run_result):
        payload = next(a for a in all_artifacts(run_result) if a.filename.endswith(".json"))
        assert json.loads(payload.data.decode("utf-8"))["summary"]["pages"] == 1

    def test_bundle_contains_expected_layout(self, run_result):
        bundle = bundle_artifact(run_result)
        with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
            names = archive.namelist()
        for expected in ("README.md", "document.md", "document.txt", "document.html", "layout.json"):
            assert expected in names
        assert any(n.endswith("raw.mmd") for n in names)
        assert any("figures/" in n for n in names)
        assert any("tables/" in n for n in names)
        assert any(n.endswith("layout_overlay.png") for n in names)

    def test_bundle_can_include_source_pages(self, run_result):
        bundle = bundle_artifact(run_result, include_pages=True)
        with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
            assert any(n.endswith("source.png") for n in archive.namelist())

    def test_bundle_readme_warns_about_demo(self, pages):
        result = pipeline.run(pages, DemoEngine(Settings()), "<image>\nx", get_mode("base"))
        bundle = bundle_artifact(result)
        with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
            readme = archive.read("README.md").decode("utf-8")
        assert "Demo-Backend" in readme

    def test_page_artifacts(self, run_result):
        names = {a.filename for a in page_artifacts(run_result.pages[0])}
        assert names == {"p001.md", "p001_raw.mmd", "p001_overlay.png"}


class TestSourceCapping:
    def test_oversized_image_is_capped(self):
        from dsocr.preprocess.loader import MAX_SOURCE_EDGE

        buffer = io.BytesIO()
        Image.new("RGB", (MAX_SOURCE_EDGE + 2000, 1000), "white").save(buffer, format="PNG")
        page = load_document("huge.png", buffer.getvalue()).pages[0]
        assert max(page.size) == MAX_SOURCE_EDGE

    def test_normal_scan_is_untouched(self, page_image):
        buffer = io.BytesIO()
        page_image.save(buffer, format="PNG")
        page = load_document("a4.png", buffer.getvalue()).pages[0]
        assert page.size == page_image.size

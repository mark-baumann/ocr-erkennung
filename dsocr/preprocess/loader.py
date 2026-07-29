"""Turn whatever the user uploaded into a list of RGB pages."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | PDF_SUFFIXES

#: Longest edge we keep. The encoder resizes to at most 1280px for the global
#: view and 640px per tile (so ~1920px for the largest 3x3 grid), which makes
#: anything beyond this pure preprocessing cost. Set generously so no realistic
#: scan loses detail — this only guards against absurd inputs like 10000px TIFFs.
MAX_SOURCE_EDGE = 4000


class UnsupportedDocument(ValueError):
    """Raised when a file cannot be turned into pages."""


@dataclass
class Page:
    """A single rasterised page ready for the encoder."""

    index: int
    image: Image.Image
    source_name: str
    #: 1-based page number within the source document (1 for standalone images).
    page_number: int = 1

    @property
    def label(self) -> str:
        return f"{self.source_name} · Seite {self.page_number}"

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


@dataclass
class Document:
    """An uploaded file expanded into pages."""

    name: str
    pages: List[Page] = field(default_factory=list)
    is_pdf: bool = False

    def __len__(self) -> int:
        return len(self.pages)


def suffix_of(name: str) -> str:
    """Lower-cased file extension of ``name``, or ``""`` when it has none.

    Only the basename is inspected, so a dot in a directory component
    (``scans.2026/report``) is not mistaken for an extension.
    """
    basename = name.replace("\\", "/").rsplit("/", 1)[-1]
    stem, dot, ext = basename.rpartition(".")
    if not dot or not stem or not ext:
        return ""
    return f".{ext.lower()}"


def load_image_bytes(data: bytes) -> Image.Image:
    """Decode image bytes, honouring the EXIF orientation flag.

    Phone photos of documents are frequently stored rotated with an EXIF hint;
    upstream handles this in ``load_image`` and we keep that behaviour.
    """
    image = Image.open(io.BytesIO(data))
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:  # pragma: no cover - corrupt EXIF, keep the raw image
        pass
    return image.convert("RGB")


def rasterize_pdf(
    data: bytes,
    *,
    dpi: int = 144,
    max_pages: int = 100,
    pages: Optional[Sequence[int]] = None,
) -> List[Image.Image]:
    """Render a PDF to RGB images with PyMuPDF.

    ``pages`` is an optional sequence of 0-based page indices; when omitted the
    first ``max_pages`` pages are rendered.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise UnsupportedDocument(
            "PDF-Unterstützung benötigt PyMuPDF. Installiere es mit `pip install pymupdf`."
        ) from exc

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    rendered: List[Image.Image] = []

    with fitz.open(stream=data, filetype="pdf") as doc:
        wanted = list(pages) if pages is not None else list(range(len(doc)))
        wanted = [p for p in wanted if 0 <= p < len(doc)][:max_pages]
        for page_index in wanted:
            pixmap = doc.load_page(page_index).get_pixmap(matrix=matrix, alpha=False)
            rendered.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))

    return rendered


def load_document(
    name: str,
    data: bytes,
    *,
    dpi: int = 144,
    max_pages: int = 100,
    page_selection: Optional[Sequence[int]] = None,
    start_index: int = 0,
) -> Document:
    """Expand an uploaded file into a :class:`Document`.

    ``start_index`` lets a caller keep a global page counter across several
    uploads so every page has a stable id in the batch view.
    """
    ext = suffix_of(name)
    if ext not in SUPPORTED_SUFFIXES:
        raise UnsupportedDocument(
            f"'{name}' hat das nicht unterstützte Format '{ext or 'unbekannt'}'. "
            f"Erlaubt: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if ext in PDF_SUFFIXES:
        images = rasterize_pdf(data, dpi=dpi, max_pages=max_pages, pages=page_selection)
        if not images:
            raise UnsupportedDocument(f"'{name}' enthält keine renderbaren Seiten.")
        numbers = list(page_selection)[: len(images)] if page_selection is not None else range(len(images))
        pages = [
            Page(
                index=start_index + i,
                image=downscale(img, MAX_SOURCE_EDGE),
                source_name=name,
                page_number=int(n) + 1,
            )
            for i, (img, n) in enumerate(zip(images, numbers))
        ]
        return Document(name=name, pages=pages, is_pdf=True)

    try:
        image = load_image_bytes(data)
    except Exception as exc:
        raise UnsupportedDocument(f"'{name}' konnte nicht als Bild gelesen werden: {exc}") from exc

    page = Page(index=start_index, image=downscale(image, MAX_SOURCE_EDGE), source_name=name)
    return Document(name=name, pages=[page], is_pdf=False)


def pdf_page_count(data: bytes) -> int:
    """Cheap page count without rasterising anything."""
    try:
        import fitz
    except ImportError:  # pragma: no cover
        return 0
    with fitz.open(stream=data, filetype="pdf") as doc:
        return len(doc)


def downscale(image: Image.Image, max_edge: int) -> Image.Image:
    """Shrink an image so its longest edge is at most ``max_edge`` pixels.

    Very large scans (600 dpi A3) cost preprocessing time without helping the
    encoder, which resizes to at most 1280px anyway.
    """
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)

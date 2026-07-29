"""DeepSeek-OCR Studio — a Streamlit front end around DeepSeek-OCR.

The heavy lifting lives in three layers:

``dsocr.preprocess``
    Loading documents and the dynamic tiling / token accounting adopted from
    the upstream DeepSeek-OCR repository.
``dsocr.engines``
    Interchangeable inference backends (vLLM, transformers, CPU demo).
``dsocr.postprocess``
    Grounding parser, markdown cleanup, table extraction and exports.

:mod:`dsocr.pipeline` wires them together.
"""

from .config import DEFAULT_MODE, DEFAULT_PROMPT, MODES, PROMPTS, SETTINGS, Settings, get_mode

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_MODE",
    "DEFAULT_PROMPT",
    "MODES",
    "PROMPTS",
    "SETTINGS",
    "Settings",
    "__version__",
    "get_mode",
]

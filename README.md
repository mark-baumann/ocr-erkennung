# 🔍 OCR Recognition NN — DeepSeek-OCR Studio

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red?logo=streamlit)](https://streamlit.io)
[![DeepSeek](https://img.shields.io/badge/Modell-DeepSeek--OCR-purple)](https://github.com/deepseek-ai/DeepSeek-OCR)
[![Tests](https://img.shields.io/badge/Tests-139%20passed-green?logo=pytest)](https://pytest.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Streamlit-Studio für DeepSeek-OCR — Layout-Grounding, Tabellen, Token-Analyse und Exporte.** Das Vision-Language-Modell komprimiert Dokumentseiten in wenige hundert Vision-Tokens und liest sie direkt als Markdown zurück.

> Basiert auf [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR). Erweitert um eine vollständige Streamlit-UI, CLI, CPU-Fallback und 139 Tests.

---

## ✨ Features

- **📄 Multi-Format:** Bilder (PNG, JPG) und PDFs (mehrseitig)
- **🎯 Layout-Grounding:** Bounding-Boxen für Titel, Text, Tabellen, Abbildungen
- **📊 Tabellen-Export:** Erkannte Tabellen als DataFrame + CSV
- **🔢 Token-Budget:** Vision-Tokens und Kompressionsrate **vor** dem Lauf berechnet
- **🖼️ Encoder-Sicht:** Kachelraster-Overlay zur Überprüfung des Auflösungsmodus
- **💾 Multi-Export:** Markdown, Text, HTML, Layout-JSON, CSVs, Abbildungen, ZIP
- **⚡ Drei Backends:** vLLM (Batch), Transformers (Einzelseiten), Demo (CPU)
- **🧪 139 Tests:** Tiling, Grounding-Parser, Markdown, Exporte, Fehlerbehandlung
- **🐳 Docker:** Mit GPU-Passthrough und Modell-Caching

---

## 🚀 Installation

### Ohne GPU — UI, Tests, Entwicklung

```bash
git clone https://github.com/mark-baumann/ocr_recognition_nn.git
cd ocr_recognition_nn
pip install -r requirements.txt
streamlit run app.py
```

### Mit GPU — echtes DeepSeek-OCR

```bash
pip install -r requirements.txt
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-gpu.txt
pip install flash-attn==2.7.3 --no-build-isolation   # optional, ~2x schneller
pip install "vllm>=0.11.1"                            # optional, Batch-Backend
streamlit run app.py
```

### Docker

```bash
docker build -t deepseek-ocr-studio .
docker run --gpus all -p 8501:8501 -v deepseek-models:/models deepseek-ocr-studio
```

---

## 🖥️ Nutzung

### Streamlit-App

```bash
streamlit run app.py
```

1. **Dokument hochladen** (Bild oder PDF)
2. **Auflösungsmodus wählen:** Tiny, Small, Base, Large, Gundam
3. **Prompt auswählen:** Markdown, OCR mit Layout, Free OCR, Abbildung, etc.
4. **Ergebnisse:** Markdown, Tabellen, Layout-Overlay, Exporte

### CLI

```bash
# Backends prüfen
python -m dsocr.cli --list-backends

# Verzeichnis verarbeiten
python -m dsocr.cli scans/*.pdf -o out/ --mode gundam --prompt markdown --zip

# Textstelle lokalisieren
python -m dsocr.cli rechnung.png --prompt locate --query "Gesamtsumme" -o out/
```

### Tests

```bash
python -m pytest tests/ -q     # 139 Tests, ~4 s, keine GPU nötig
```

---

## 🎛️ Auflösungsmodi

| Modus | Global | Kacheln | Vision-Tokens | Wofür |
|---|---|---|---|---|
| Tiny | 512 px | — | 64 | Screenshots |
| Small | 640 px | — | 100 | einspaltige Dokumente |
| Base | 1024 px | — | 256 | gescannte A4-Seiten |
| Large | 1280 px | — | 400 | kleine Schrift |
| **Gundam** | 1024 px | n × 640 px | 256 + n·100 | Zeitungen, Formulare |

---

## 🧱 Tech-Stack

| Komponente | Technologie |
|---|---|
| **Frontend** | Streamlit |
| **OCR-Modell** | DeepSeek-OCR (Vision-Language) |
| **Backends** | vLLM, HuggingFace Transformers, Demo (CPU) |
| **PDF** | PyMuPDF |
| **Sprache** | Python 3.10+ |

---

## 📁 Projektstruktur

```
ocr_recognition_nn/
├── app.py                          # Streamlit-Einstiegspunkt
├── dsocr/
│   ├── config.py                   # Auflösungsmodi, Prompts, Settings
│   ├── pipeline.py                 # Orchestrierung
│   ├── cli.py                      # Headless-Batch
│   ├── preprocess/                 # Loader, Tiling, Token-Rechnung
│   ├── engines/                    # vLLM, Transformers, Demo
│   ├── postprocess/                # Grounding, Markdown, Export
│   └── ui/                         # Theme, Sidebar, Ergebnisse
├── tests/                          # 139 Tests
├── assets/                         # Screenshots
├── Dockerfile
└── requirements.txt
```

---

## 👤 Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann) · [markb.de](https://markb.de)

---

*Modell und Referenzcode von [DeepSeek](https://github.com/deepseek-ai/DeepSeek-OCR) (MIT). Dieses Projekt steht ebenfalls unter MIT-Lizenz.*

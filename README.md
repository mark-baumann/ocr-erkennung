# 🔍 OCR Recognition NN — DeepSeek-OCR

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mark-baumann/ocr-erkennung/blob/claude/deepseek-ocr-streamlit-46tyl3/ocr_colab.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![DeepSeek](https://img.shields.io/badge/Modell-DeepSeek--OCR-purple)](https://github.com/deepseek-ai/DeepSeek-OCR)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Handschrift- und Dokument-Erkennung (OCR) ohne App — Google-Colab-fähig und headless.** Das Vision-Language-Modell DeepSeek-OCR komprimiert Dokumentseiten in wenige hundert Vision-Tokens und liest sie direkt als Markdown zurück — auch Handschrift.

> Basiert auf [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR). Erweitert um ein Google-Colab-Notebook, eine CLI und CPU-Fallback. Keine Streamlit-App, keine GUI.

---

## ✨ Features

- **📄 Multi-Format:** Bilder (PNG, JPG) und PDFs (mehrseitig)
- **🎯 Layout-Grounding:** Bounding-Boxen für Titel, Text, Tabellen, Abbildungen
- **📊 Tabellen-Export:** Erkannte Tabellen als DataFrame + CSV
- **🔢 Token-Budget:** Vision-Tokens und Kompressionsrate **vor** dem Lauf berechnet
- **🖼️ Encoder-Sicht:** Kachelraster-Overlay zur Überprüfung des Auflösungsmodus
- **💾 Multi-Export:** Markdown, Text, HTML, Layout-JSON, CSVs, Abbildungen, ZIP
- **⚡ Drei Backends:** vLLM (Batch), Transformers (Einzelseiten), Demo (CPU)
- **🐳 Docker:** Mit GPU-Passthrough und Modell-Caching

---

## 🚀 Installation

### Ohne GPU — Notebook und CLI

```bash
git clone https://github.com/mark-baumann/ocr-erkennung.git
cd ocr-erkennung
pip install -e .
jupyter notebook ocr_colab.ipynb
```

### Mit GPU — echtes DeepSeek-OCR

```bash
pip install -e .
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-gpu.txt
pip install flash-attn==2.7.3 --no-build-isolation   # optional, ~2x schneller
pip install "vllm>=0.11.1"                            # optional, Batch-Backend
jupyter notebook ocr_colab.ipynb
```

### Docker

```bash
docker build -t deepseek-ocr .
docker run --rm -v $(pwd)/scans:/in -v $(pwd)/out:/out deepseek-ocr /in/rechnung.png -o /out --mode gundam --prompt markdown --zip
```

> Das Image startet die headless CLI (`python -m dsocr.cli`), nicht mehr eine Streamlit-App.

---

## 🖥️ Nutzung

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mark-baumann/ocr-erkennung/blob/claude/deepseek-ocr-streamlit-46tyl3/ocr_colab.ipynb)

Direkt im Browser öffnen:

```bash
https://colab.research.google.com/github/mark-baumann/ocr-erkennung/blob/claude/deepseek-ocr-streamlit-46tyl3/ocr_colab.ipynb
```

In Colab zuerst eine GPU-Laufzeit aktivieren (Laufzeit > Laufzeittyp ändern → GPU). Das Notebook installiert das Projekt und die GPU-Abhängigkeiten, fragt anschließend Bild- oder PDF-Dateien ab und lädt ein ZIP mit Markdown, Text, Layout-JSON und Seiten-Ergebnissen herunter.

### CLI

```bash
# Backends prüfen
python -m dsocr.cli --list-backends

# Verzeichnis verarbeiten
python -m dsocr.cli scans/*.pdf -o out/ --mode gundam --prompt markdown --zip

# Textstelle lokalisieren
python -m dsocr.cli rechnung.png --prompt locate --query "Gesamtsumme" -o out/
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
| **Ausführung** | Jupyter Notebook / CLI |
| **OCR-Modell** | DeepSeek-OCR (Vision-Language) |
| **Backends** | vLLM, HuggingFace Transformers, Demo (CPU) |
| **PDF** | PyMuPDF |
| **Sprache** | Python 3.10+ |

---

## 📁 Projektstruktur

```
ocr-erkennung/
├── ocr_colab.ipynb                  # Google-Colab-Einstiegspunkt
├── dsocr/
│   ├── config.py                   # Auflösungsmodi, Prompts, Settings
│   ├── pipeline.py                 # Orchestrierung
│   ├── cli.py                      # Headless-Batch
│   ├── preprocess/                 # Loader, Tiling, Token-Rechnung
│   ├── engines/                    # vLLM, Transformers, Demo
│   ├── postprocess/                # Grounding, Markdown, Export
├── assets/                         # Screenshots
├── Dockerfile
└── requirements-gpu.txt
```

---

## 👤 Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann) · [markb.de](https://markb.de)

---

*Modell und Referenzcode von [DeepSeek](https://github.com/deepseek-ai/DeepSeek-OCR) (MIT). Dieses Projekt steht ebenfalls unter MIT-Lizenz.*

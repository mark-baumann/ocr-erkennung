# 🔍 OCR Recognition NN — DeepSeek-OCR Studio

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mark-baumann/ocr-erkennung/blob/claude/deepseek-ocr-streamlit-46tyl3/ocr_colab.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![DeepSeek](https://img.shields.io/badge/Modell-DeepSeek--OCR-purple)](https://github.com/deepseek-ai/DeepSeek-OCR)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Handschrift- und Dokument-Erkennung (OCR) mit Streamlit-GUI — zusätzlich Google-Colab-fähig und headless.** Das Vision-Language-Modell DeepSeek-OCR komprimiert Dokumentseiten in wenige hundert Vision-Tokens und liest sie direkt als Markdown zurück — auch Handschrift.

> Basiert auf [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR). Erweitert um eine Streamlit-GUI (AUG-241), ein Google-Colab-Notebook, eine CLI und CPU-Fallback.

---

## ✨ Features

- **🖥️ Streamlit-GUI:** Upload, Token-Budget, Encoder-Sicht, Layout-Overlay und Exporte direkt im Browser
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

### Ohne GPU — GUI, Notebook und CLI

```bash
git clone https://github.com/mark-baumann/ocr-erkennung.git
cd ocr-erkennung
pip install -e ".[app]"
streamlit run app/app.py
```

### Mit GPU — echtes DeepSeek-OCR

```bash
pip install -e ".[app]"
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-gpu.txt
pip install flash-attn==2.7.3 --no-build-isolation   # optional, ~2x schneller
pip install "vllm>=0.11.1"                            # optional, Batch-Backend
streamlit run app/app.py                              # oder: jupyter notebook ocr_colab.ipynb
```

### Docker

```bash
docker build -t deepseek-ocr .
# GUI (Default, Port 8513 — Live: dokumenten-ocr.markb.de):
docker run --rm -p 8513:8513 deepseek-ocr
# Headless-CLI:
docker run --rm -v $(pwd)/scans:/in -v $(pwd)/out:/out --entrypoint python deepseek-ocr -m dsocr.cli /in/rechnung.png -o /out --mode gundam --prompt markdown --zip
```

> Das Image startet die Streamlit-GUI (`app/app.py`); die headless CLI bleibt über `--entrypoint python -m dsocr.cli` erreichbar.

---

## 🖥️ Nutzung

### Streamlit-GUI (Live)

Die GUI läuft live auf **https://dokumenten-ocr.markb.de** (Service `deepseek-ocr-studio`, Port 8513 im Infrastruktur-Deploy).

Lokal:

```bash
streamlit run app/app.py
```

Im Browser: Dokumente hochladen → Token-Budget und Kachelraster prüfen → „Analyse starten" → Ergebnis als Markdown/Tabellen/Layout-Export herunterladen.

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
| **GUI** | Streamlit (Port 8513) |
| **Ausführung** | Streamlit-GUI / Jupyter Notebook / CLI |
| **OCR-Modell** | DeepSeek-OCR (Vision-Language) |
| **Backends** | vLLM, HuggingFace Transformers, Demo (CPU) |
| **PDF** | PyMuPDF |
| **Sprache** | Python 3.10+ |

---

## 📁 Projektstruktur

```
ocr-erkennung/
├── app/app.py                       # Streamlit-GUI (Einstiegspunkt)
├── ocr_colab.ipynb                  # Google-Colab-Einstiegspunkt
├── dsocr/
│   ├── config.py                   # Auflösungsmodi, Prompts, Settings
│   ├── pipeline.py                 # Orchestrierung
│   ├── cli.py                      # Headless-Batch
│   ├── preprocess/                 # Loader, Tiling, Token-Rechnung
│   ├── engines/                    # vLLM, Transformers, Demo
│   ├── postprocess/                # Grounding, Markdown, Export
│   └── ui/                         # Sidebar, Ergebnis-Tabs, Theme
├── assets/                         # Screenshots
├── Dockerfile
└── requirements-gpu.txt
```

---

## 👤 Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann) · [markb.de](https://markb.de)

---

*Modell und Referenzcode von [DeepSeek](https://github.com/deepseek-ai/DeepSeek-OCR) (MIT). Dieses Projekt steht ebenfalls unter MIT-Lizenz.*

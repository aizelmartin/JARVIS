# JARVIS – Universal Sign Language Translator

An ML-powered sign language recognition system that converts hand gestures captured via webcam into text and speech.

> **College Project** | Python · OpenCV · MediaPipe · scikit-learn

---

## Features

- 🖐️ **Sign Language → Text** — Real-time hand gesture recognition using webcam
- 🔊 **Sign Language → Speech** — Recognized text converted to speech via TTS
- 🎤 **Speech → Text** (Supporting feature) — Basic speech input support
- 📊 **Syllabus ML Coverage** — PCA, Clustering, KNN, Decision Trees, Neural Networks

## Tech Stack

| Component | Library |
|---|---|
| Webcam capture | OpenCV |
| Hand landmark detection | MediaPipe |
| Feature extraction | NumPy |
| Dimensionality reduction | PCA (scikit-learn) |
| Classifiers | KNN, Decision Tree, Random Forest, MLP (scikit-learn) |
| Text-to-Speech | pyttsx3 |

## Project Structure

```
JARVIS/
├── src/
│   ├── camera/          # Webcam stream management
│   ├── hand_tracking/   # MediaPipe landmark extraction & normalization
│   ├── dataset/         # Dataset ingestion & preprocessing
│   ├── ml/              # ML training, evaluation, and inference
│   ├── speech/          # Text-to-Speech (TTS)
│   └── ui/              # Overlay / HUD utilities
├── data/
│   ├── raw/             # Source images (not committed to Git)
│   └── processed/       # Extracted landmark CSVs
├── models/              # Saved trained models (.pkl)
├── docs/                # Project documentation
├── tests/               # Unit tests
├── assets/              # Screenshots, demo images
├── main.py              # Entry point
└── requirements.txt
```

## Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd JARVIS

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python main.py
```

## Controls

| Key | Action |
|---|---|
| `Q` | Quit |
| `Space` | Insert space between words |

## Team

| Name | Role |
|---|---|
| Member 1 | ML Pipeline |
| Member 2 | Hand Tracking |
| Member 3 | UI & Integration |
| Member 4 | Dataset & Documentation |

---

*Built as a college ML project — 2026*

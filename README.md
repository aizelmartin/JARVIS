# JARVIS – Universal Sign Language Translator 🦻🤖

**JARVIS** is an advanced, two-way communication system designed to bridge the gap between deaf/mute individuals and hearing individuals. Built as an academic Machine Learning project, it leverages real-time computer vision, custom classification models, and native Speech-to-Text / Text-to-Speech engines.

---

## 🌟 Key Features

### 1. Sign Language to Speech (For the Deaf/Mute User)
* **Real-Time Hand Tracking:** Uses MediaPipe to track 126 skeletal landmarks across both hands simultaneously at 30+ FPS.
* **Custom Machine Learning Model:** A highly optimized Random Forest Classifier trained exclusively on custom signs collected in the deployment environment, achieving **99.27% accuracy**.
* **Smart Sentence Builder:** Instead of just flashing words, JARVIS uses a "Hold-to-Confirm" dwell stabilization to construct full sentences dynamically without any word limits.
* **Native Text-to-Speech (TTS):** Uses asynchronous Windows SAPI (SpVoice) to speak the constructed sentences out loud in real time.

### 2. Speech to Text (For the Hearing Person)
* **Microphone STT:** Hearing individuals can press a button to speak into the microphone. JARVIS transcribes their speech and displays it on the HUD in real-time so the deaf user can read it.

### 3. High-Tech Glassmorphism HUD
* Features a sleek, non-intrusive Heads-Up Display (HUD) showing ML model confidence, system telemetry, current sentence, and real-time STT banners.

---

## 🛠️ Technology Stack
* **Language:** Python 3.x
* **Computer Vision:** OpenCV (cv2), MediaPipe (Hand Tracking)
* **Machine Learning:** Scikit-Learn (Random Forest, SVM, KNN), Pandas, NumPy
* **Speech Integration:** `pyttsx3` / `win32com` (TTS), `SpeechRecognition`, `sounddevice` (STT)

---

## 📊 Model Architecture & Performance
During Phase 4 benchmarking, 5 different algorithms were evaluated using 5-Fold Cross Validation on 3,400 samples across 17 distinct custom signs.

| Model | Test Accuracy | Inference Latency |
|-------|---------------|-------------------|
| **Random Forest (Winner)** | **99.26%** | **12.87 ms** |
| SVM (RBF Kernel) | 99.12% | 0.34 ms |
| KNN (k=5) | 98.82% | 4.80 ms |
| Logistic Regression | 98.24% | 0.25 ms |
| Decision Tree | 97.94% | 0.23 ms |

*The model was trained exclusively on the deployment environment background to ensure maximum real-world reliability.*

---

## 🚀 How to Run

### Option A: Web Interface (New!)
The newest version of JARVIS includes a professional, real-time web interface. This mode uses your browser's webcam and offers language translation features.
1. **Activate the virtual environment:**
   ```powershell
   .venv\Scripts\activate
   ```
2. **Start the Flask server:**
   ```bash
   python web_app.py
   ```
3. **Open your browser** and navigate to: `http://127.0.0.1:5000`

### Option B: Terminal Interface
The classic HUD overlay mode:
1. **Launch JARVIS:**
   ```bash
   python main.py
   ```
2. **Controls:**
   * `ENTER`: Speak the entire constructed sentence aloud
   * `BACKSPACE`: Delete the last added word
   * `C`: Clear the current sentence
   * `M`: Activate Microphone (Speech-to-Text for the hearing person)
   * `T`: Toggle Auto-Voice ON/OFF
   * `Q`: Quit

---

## 📝 Custom Data Collection
JARVIS includes a built-in Data Collector. To train your own words:
1. Run `python src/dataset/data_collector.py --signs word1 word2`
2. Follow on-screen prompts to record 200 samples per word.
3. Run `python src/dataset/merge_datasets.py`
4. Run `python src/ml/train_models.py --data "data/processed/landmarks_combined.csv"`
5. Launch `main.py`!

---
*Developed for academic purposes to explore accessible AI integration.*

# J.A.R.V.I.S. – Joint AI Recognition and Voice Interpretation System 🦻🤖

**J.A.R.V.I.S.** is an advanced, two-way communication system designed to bridge the gap between deaf/mute individuals and hearing individuals. Originally an academic Machine Learning project, it has evolved into a complete, professional desktop application offering real-time sign recognition, translation, multilingual speech output, and an integrated sign training workspace.

---

## 🌟 Key Features

### 1. Sign Language to Speech (For the Deaf/Mute User)
* **Real-Time Hand Tracking:** Uses MediaPipe to track skeletal landmarks across hands via live camera feed.
* **Custom Machine Learning Model:** Highly optimized classification trained on custom signs for high accuracy.
* **Smart Sentence Builder:** Uses "Hold-to-Confirm" temporal stabilization to construct full sentences dynamically without rigid word limits.
* **Multilingual Translation & Speech:** Automatically translates the constructed sentence and speaks it aloud in supported languages using `gTTS` and `pygame`.

### 2. Speech to Text (For the Hearing Person)
* **Multilingual Microphone STT:** Hearing individuals can press a button to speak in multiple supported languages. JARVIS transcribes their speech, displays it in real-time, and can instantly provide an English translation for the deaf user.

### 3. Integrated Sign Training Workspace
* **Dynamic Learning:** A full-window training mode built directly into the UI.
* **Live Collection:** Record new hand-landmark samples easily with live visual feedback.
* **Safe Hot-Reloading:** Automatically trains the machine learning model in the background and hot-reloads it without restarting the app, preserving a safe backup of the previous model.

### 4. Cross-Platform Availability
* Includes source code for the **Android App** and **Web Interface** variants.

---

## 🛠️ Technology Stack
* **Language:** Python 3.x
* **UI Framework:** CustomTkinter (Professional Dark Theme)
* **Computer Vision:** OpenCV (`cv2`), MediaPipe (Hand Tracking)
* **Machine Learning:** Scikit-Learn (SVM/Random Forest), Pandas, NumPy
* **Speech & Translation:** `googletrans`, `gTTS`, `SpeechRecognition`, `sounddevice`, `pygame`

---

## 🚀 Setup & Installation (Windows)

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/aizelmartin/JARVIS.git
   cd JARVIS
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Hardware & Network Requirements:**
   * **Webcam** (required for sign recognition)
   * **Microphone & Speakers** (required for speech-to-text and text-to-speech)
   * **Active Internet Connection** (required for `googletrans`, `gTTS`, and Google STT APIs)

---

## 🎮 How to Run

**Launch the Desktop Application:**
```powershell
python main.py
```

### Supported Languages
English, Malayalam, Hindi, Tamil, Kannada, and Telugu.
*(Note: Requires an internet connection for translation and voice synthesis.)*

---

## 📝 Training New Signs

You can easily add new signs directly through the desktop app:
1. Turn the Camera **ON**.
2. Click **➕ Open Training Workspace** on the right panel.
3. Enter the label for your new sign.
4. Select a target number of samples (e.g., 200).
5. Click **▶ Start Collection** and perform the sign, slightly varying distance and angle.
6. Once completed, click **💾 Save & Train**. The model will re-train in the background and hot-reload automatically.

*(Any large model backups or datasets generated during training are kept locally and ignored by Git to save space.)*

---
*“Connecting people, beyond words.”*

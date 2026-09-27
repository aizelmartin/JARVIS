"""
web_app.py — JARVIS Web Interface
===================================
Flask + Flask-SocketIO (gevent + gevent-websocket) backend.

Architecture:
  Browser webcam → base64 JPEG frames via WebSocket
  → existing HandDetector + SignClassifier pipeline
  → telemetry + annotated frame → browser
"""

# ── MUST be first: gevent monkey-patch stdlib before any other imports ──────── #
from gevent import monkey
monkey.patch_all(thread=False, socket=True, select=True, time=True)

import os
import sys
import time
import cv2
import numpy as np
import base64
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from googletrans import Translator

# ── Import existing JARVIS modules (do NOT modify these) ──────────────────── #
sys.path.insert(0, os.path.dirname(__file__))
from src.hand_tracking.hand_detector import HandDetector
from src.ml.classifier import SignClassifier
from src.ml.sentence_builder import SentenceBuilder
from src.speech.speech_engine import SpeechEngine
from src.speech.speech_listener import SpeechListener

# ── Flask App ─────────────────────────────────────────────────────────────── #
app = Flask(__name__)
app.config['SECRET_KEY'] = 'jarvis_secret_key'

# Explicitly use gevent + gevent-websocket for robust WebSocket support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent',
                    logger=False, engineio_logger=False)

# ── Initialize JARVIS ML & Speech Components ──────────────────────────────── #
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "saved_models", "sign_classifier.pkl")

detector = HandDetector(max_hands=2, min_detection_conf=0.7)

try:
    classifier = SignClassifier(MODEL_PATH, smooth_window=5, confidence_threshold=0.65)
except FileNotFoundError:
    print(f"[JARVIS] WARNING: Model not found at {MODEL_PATH}")
    print("[JARVIS] Run: python src/ml/train_models.py")
    classifier = None

sentence_builder = SentenceBuilder(max_words=100)
speech_engine    = SpeechEngine(rate=2, volume=100)
speech_listener  = SpeechListener()
translator       = Translator()

# ── Dwell-stabilization state ─────────────────────────────────────────────── #
REQUIRED_STABLE_FRAMES = 9
current_candidate  = None
hold_count         = 0
last_spoken_sign   = None
spoken_flash_timer = 0.0


import threading
frame_counter = 0

# ── Routes ────────────────────────────────────────────────────────────────── #
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': classifier is not None,
        'classes': list(classifier.classes) if classifier else []
    })


# ── Socket Events ─────────────────────────────────────────────────────────── #
@socketio.on('connect')
def on_connect():
    print("[JARVIS Web] Client connected.")
    emit('status', {'data': 'Connected to JARVIS Backend'})


@socketio.on('disconnect')
def on_disconnect():
    print("[JARVIS Web] Client disconnected.")


@socketio.on('process_frame')
def handle_process_frame(data):
    """
    Receives base64 JPEG frame from browser, runs it through:
      HandDetector → SignClassifier → SentenceBuilder
    Returns annotated frame + telemetry to browser.
    """
    global current_candidate, hold_count, last_spoken_sign, spoken_flash_timer, frame_counter

    if classifier is None:
        print("[JARVIS Web] ERROR: Classifier is None!")
        emit('processed_frame', {'error': 'ML model not loaded. Run train_models.py first.'})
        return

    try:
        frame_counter += 1
        # ── Decode base64 image from browser canvas ────────────────────────── #
        img_data = data.get('image', '')
        if not img_data or ',' not in img_data:
            print(f"[JARVIS Web] Frame #{frame_counter}: Invalid image data")
            return

        _, encoded = img_data.split(',', 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print(f"[JARVIS Web] Frame #{frame_counter}: cv2.imdecode returned None")
            return

        # ── Mirror flip horizontally (matches CameraStream in main.py) ───── #
        frame = cv2.flip(frame, 1)

        # ── Hand detection & landmark extraction (existing pipeline) ──────── #
        t0 = time.time()
        detector.process_frame(frame)
        dt = (time.time() - t0) * 1000

        n_hands      = detector.num_hands_detected()
        features_126 = detector.extract_features_both()

        pred_sign  = "..."
        confidence = 0.0

        # ── ML Prediction with dwell stabilization (same logic as main.py) ── #
        if n_hands > 0:
            pred_sign, confidence = classifier.predict_smooth(features_126)

            if pred_sign != "..." and confidence >= 0.70:
                if pred_sign == current_candidate:
                    hold_count += 1
                else:
                    current_candidate = pred_sign
                    hold_count = 1

                # Confirmed after holding stable for REQUIRED_STABLE_FRAMES
                if hold_count >= REQUIRED_STABLE_FRAMES:
                    if pred_sign != last_spoken_sign or (time.time() - spoken_flash_timer) > 2.5:
                        sentence_builder.add_word(pred_sign)
                        last_spoken_sign   = pred_sign
                        spoken_flash_timer = time.time()
            else:
                current_candidate = None
                hold_count        = 0
        else:
            classifier.reset_history()
            current_candidate = None
            hold_count        = 0
            last_spoken_sign  = None

        hold_ratio = (
            min(1.0, hold_count / REQUIRED_STABLE_FRAMES)
            if (current_candidate and hold_count > 0) else 0.0
        )

        if frame_counter % 15 == 0 or n_hands > 0:
            print(f"[JARVIS Web] Frame #{frame_counter}: shape={frame.shape}, detect={dt:.1f}ms, hands={n_hands}, sign={pred_sign} ({confidence*100:.0f}%), hold={hold_ratio*100:.0f}%")

        # ── Extract landmark JSON to send to frontend ────────────────────── #
        hands_data = []
        if detector._results and detector._results.multi_hand_landmarks:
            for i, hand_lm in enumerate(detector._results.multi_hand_landmarks):
                raw_label = detector._results.multi_handedness[i].classification[0].label
                corrected  = 'Left' if raw_label == 'Right' else 'Right'
                landmarks  = [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]
                hands_data.append({'label': corrected, 'landmarks': landmarks})

        emit('processed_frame', {
            'hands':      hands_data,
            'sign':       pred_sign,
            'confidence': confidence,
            'hold_ratio': hold_ratio,
            'sentence':   sentence_builder.get_sentence()
        })
    except Exception as e:
        print(f"[JARVIS Web] Exception in handle_process_frame: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pass  # No lock to release



@socketio.on('command')
def handle_command(data):
    """Delete last word or clear entire sentence."""
    cmd = data.get('action')
    if cmd == 'clear':
        sentence_builder.clear()
    elif cmd == 'delete':
        sentence_builder.remove_last()
    emit('sentence_updated', {'sentence': sentence_builder.get_sentence()})


@socketio.on('translate')
def handle_translate(data):
    """Translate the current sentence to the target language."""
    sentence    = sentence_builder.get_sentence()
    target_lang = data.get('lang', 'en')

    if not sentence or sentence == '...':
        emit('translation_result', {'translated': '...', 'lang': target_lang})
        return

    try:
        res = translator.translate(sentence, dest=target_lang)
        emit('translation_result', {'translated': res.text, 'lang': target_lang})
    except Exception as e:
        print(f"[JARVIS Web] Translation error: {e}")
        emit('translation_result', {'translated': f'[Translation error: {e}]', 'lang': target_lang})


@socketio.on('speak')
def handle_speak(data):
    """Speak text using existing Windows SAPI engine."""
    text = data.get('text', '').strip()
    if text and text != '...':
        speech_engine.speak(text, force=True)


@socketio.on('start_stt')
def handle_stt(data):
    """Start Speech-to-Text in background thread and stream back result."""
    if speech_listener.is_listening:
        return

    def stt_worker():
        speech_listener.listen_async(duration_sec=3.5)
        while speech_listener.is_listening:
            time.sleep(0.4)
            socketio.emit('stt_status', {'status': speech_listener.status_msg})
        socketio.emit('stt_result', {
            'text':   speech_listener.last_text,
            'status': speech_listener.status_msg
        })

    import gevent
    gevent.spawn(stt_worker)


# ── Entry Point ───────────────────────────────────────────────────────────── #
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  JARVIS Web Interface")
    print("="*55)
    print(f"  Model loaded : {classifier is not None}")
    if classifier:
        print(f"  Model name   : {classifier.model_name}")
        print(f"  Signs        : {list(classifier.classes)}")
    print("="*55)
    print("  Open browser at: http://localhost:5000")
    print("  NOTE: Use 'localhost', NOT '127.0.0.1' (camera security)")
    print("="*55 + "\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)

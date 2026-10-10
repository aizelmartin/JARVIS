import cv2
import sys
import os
import time
import threading
from typing import Optional

import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

# Translation & Speech
from googletrans import Translator

sys.path.insert(0, os.path.dirname(__file__))

from src.camera.camera_stream import CameraStream
from src.hand_tracking.hand_detector import HandDetector
from src.ml.classifier import SignClassifier
from src.ml.sentence_builder import SentenceBuilder
from src.speech.speech_engine import SpeechEngine
from src.speech.speech_listener import SpeechListener

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "saved_models", "sign_classifier.pkl")

# UI Settings
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class JarvisDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("J.A.R.V.I.S. — Universal Sign Language Translator")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # State Variables
        self.tts_enabled = ctk.BooleanVar(value=True)
        self.is_fullscreen = False
        
        self.sentence = ""
        self.translated_text = ""
        self.current_sign = "..."
        self.current_conf = 0.0
        self.target_lang = ctk.StringVar(value="Malayalam")
        
        self.lang_map = {
            "Malayalam": "ml",
            "Hindi": "hi",
            "Tamil": "ta",
            "Kannada": "kn",
            "Telugu": "te"
        }

        # Pipeline instances
        self.cam = None
        self.detector = None
        self.classifier = None
        self.sentence_builder = None
        self.speech = None
        self.listener = None
        self.translator = Translator()

        self.running = True
        
        self.REQUIRED_STABLE_FRAMES = 9
        self.current_candidate = None
        self.hold_count = 0
        self.last_spoken_sign = None
        self.spoken_flash_timer = 0.0

        self.setup_ui()
        self.bind_shortcuts()
        self.init_pipeline()

    def setup_ui(self):
        # Header
        self.header_frame = ctk.CTkFrame(self, height=60, corner_radius=0)
        self.header_frame.pack(side="top", fill="x")
        
        self.title_label = ctk.CTkLabel(self.header_frame, text="J.A.R.V.I.S.", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(side="left", padx=20, pady=10)
        
        self.subtitle_label = ctk.CTkLabel(self.header_frame, text="AI-Powered Sign Language Translator", font=ctk.CTkFont(size=14))
        self.subtitle_label.pack(side="left", padx=10, pady=10)
        
        self.status_label = ctk.CTkLabel(self.header_frame, text="Initializing...", text_color="orange")
        self.status_label.pack(side="right", padx=20, pady=10)
        
        self.fs_btn = ctk.CTkButton(self.header_frame, text="Fullscreen (F11)", width=120, command=self.toggle_fullscreen)
        self.fs_btn.pack(side="right", padx=10, pady=10)

        # Main Workspace
        self.main_paned = ctk.CTkFrame(self, fg_color="transparent")
        self.main_paned.pack(side="top", fill="both", expand=True, padx=20, pady=10)
        
        self.main_paned.columnconfigure(0, weight=6)
        self.main_paned.columnconfigure(1, weight=4)
        self.main_paned.rowconfigure(0, weight=1)

        # Left Column: Camera
        self.camera_frame = ctk.CTkFrame(self.main_paned)
        self.camera_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        self.video_label = ctk.CTkLabel(self.camera_frame, text="")
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.hud_frame = ctk.CTkFrame(self.camera_frame, height=50)
        self.hud_frame.pack(fill="x", padx=10, pady=10)
        
        self.sign_display = ctk.CTkLabel(self.hud_frame, text="Sign: ...", font=ctk.CTkFont(size=20, weight="bold"))
        self.sign_display.pack(side="left", padx=20, pady=10)
        
        self.conf_progress = ctk.CTkProgressBar(self.hud_frame, width=150)
        self.conf_progress.set(0)
        self.conf_progress.pack(side="left", padx=20, pady=10)
        
        # Right Column: Text & Controls
        self.right_frame = ctk.CTkFrame(self.main_paned)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # Sentence section
        ctk.CTkLabel(self.right_frame, text="Constructed Sentence", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        self.sentence_textbox = ctk.CTkTextbox(self.right_frame, height=100, font=ctk.CTkFont(size=18))
        self.sentence_textbox.pack(fill="x", padx=10, pady=5)
        self.sentence_textbox.configure(state="disabled")

        # Translation section
        lang_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        lang_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(lang_frame, text="Translation Language:").pack(side="left")
        self.lang_option = ctk.CTkOptionMenu(lang_frame, variable=self.target_lang, values=list(self.lang_map.keys()))
        self.lang_option.pack(side="left", padx=10)
        
        self.translate_btn = ctk.CTkButton(lang_frame, text="Translate", width=100, command=self.do_translation)
        self.translate_btn.pack(side="right")

        self.translation_textbox = ctk.CTkTextbox(self.right_frame, height=100, font=ctk.CTkFont(size=18))
        self.translation_textbox.pack(fill="x", padx=10, pady=5)
        
        # STT Section
        stt_frame = ctk.CTkFrame(self.right_frame)
        stt_frame.pack(fill="x", padx=10, pady=10)
        self.stt_label = ctk.CTkLabel(stt_frame, text="Mic: Idle", font=ctk.CTkFont(size=14))
        self.stt_label.pack(side="left", padx=10, pady=10)
        self.stt_btn = ctk.CTkButton(stt_frame, text="🎙 Listen (M)", width=100, command=self.listen_mic)
        self.stt_btn.pack(side="right", padx=10, pady=10)

        # Controls Section
        ctrl_frame = ctk.CTkFrame(self.right_frame)
        ctrl_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Grid of controls
        ctrl_frame.columnconfigure((0,1), weight=1)
        
        self.btn_speak_sign = ctk.CTkButton(ctrl_frame, text="Speak Current Sign (S)", command=self.speak_current_sign)
        self.btn_speak_sign.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.btn_speak_sent = ctk.CTkButton(ctrl_frame, text="Speak Sentence (Enter)", command=self.speak_sentence)
        self.btn_speak_sent.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.btn_del_word = ctk.CTkButton(ctrl_frame, text="Delete Last Word (Back)", command=self.delete_word)
        self.btn_del_word.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        self.btn_clear = ctk.CTkButton(ctrl_frame, text="Clear All (C)", command=self.clear_all)
        self.btn_clear.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        self.btn_speak_trans = ctk.CTkButton(ctrl_frame, text="Speak Translation", command=self.speak_translation)
        self.btn_speak_trans.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        self.toggle_tts = ctk.CTkSwitch(ctrl_frame, text="Auto-Speak", variable=self.tts_enabled)
        self.toggle_tts.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

    def bind_shortcuts(self):
        self.bind("<Return>", lambda e: self.speak_sentence() if str(self.focus_get()) != str(self.translation_textbox) else None)
        self.bind("<BackSpace>", lambda e: self.delete_word() if str(self.focus_get()) != str(self.translation_textbox) else None)
        self.bind("<c>", lambda e: self.clear_all() if str(self.focus_get()) != str(self.translation_textbox) else None)
        self.bind("<C>", lambda e: self.clear_all() if str(self.focus_get()) != str(self.translation_textbox) else None)
        self.bind("<m>", lambda e: self.listen_mic() if str(self.focus_get()) != str(self.translation_textbox) else None)
        self.bind("<M>", lambda e: self.listen_mic() if str(self.focus_get()) != str(self.translation_textbox) else None)
        self.bind("<s>", lambda e: self.speak_current_sign() if str(self.focus_get()) != str(self.translation_textbox) else None)
        self.bind("<S>", lambda e: self.speak_current_sign() if str(self.focus_get()) != str(self.translation_textbox) else None)
        self.bind("<t>", lambda e: self.toggle_tts.toggle() if str(self.focus_get()) != str(self.translation_textbox) else None)
        self.bind("<T>", lambda e: self.toggle_tts.toggle() if str(self.focus_get()) != str(self.translation_textbox) else None)
        self.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.bind("<Escape>", lambda e: self.exit_fullscreen())

    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.attributes("-fullscreen", self.is_fullscreen)

    def exit_fullscreen(self, event=None):
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.attributes("-fullscreen", False)

    def init_pipeline(self):
        self.status_label.configure(text="Starting camera...", text_color="orange")
        self.cam = CameraStream(camera_index=0, width=640, height=480)
        self.detector = HandDetector(max_hands=2, min_detection_conf=0.7)
        
        if not self.cam.start():
            self.status_label.configure(text="Camera Error!", text_color="red")
            return

        if not os.path.exists(MODEL_PATH):
            self.status_label.configure(text="Model Missing!", text_color="red")
            return

        self.classifier = SignClassifier(MODEL_PATH, smooth_window=5, confidence_threshold=0.65)
        self.speech = SpeechEngine(rate=2, volume=100)
        self.listener = SpeechListener()
        self.sentence_builder = SentenceBuilder(max_words=100)
        
        self.status_label.configure(text="Connected", text_color="green")
        
        # Start processing loop
        self.process_thread = threading.Thread(target=self.camera_worker, daemon=True)
        self.process_thread.start()
        
        self.update_ui_loop()

    def camera_worker(self):
        while self.running:
            frame = self.cam.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Copy frame for OpenCV drawing
            display_frame = frame.copy()
            
            self.detector.process_frame(display_frame)
            self.detector.draw_landmarks(display_frame)
            
            n_hands = self.detector.num_hands_detected()
            features_126 = self.detector.extract_features_both()
            
            pred_sign = "..."
            confidence = 0.0

            if n_hands > 0:
                pred_sign, confidence = self.classifier.predict_smooth(features_126)
                
                if pred_sign != "..." and confidence >= 0.70:
                    if pred_sign == self.current_candidate:
                        self.hold_count += 1
                    else:
                        self.current_candidate = pred_sign
                        self.hold_count = 1
                        
                    if self.hold_count >= self.REQUIRED_STABLE_FRAMES:
                        if pred_sign != self.last_spoken_sign or (time.time() - self.spoken_flash_timer) > 2.5:
                            if self.tts_enabled.get():
                                self.speech.speak(pred_sign)
                            self.sentence_builder.add_word(pred_sign)
                            self.last_spoken_sign = pred_sign
                            self.spoken_flash_timer = time.time()
                else:
                    self.current_candidate = None
                    self.hold_count = 0
            else:
                self.classifier.reset_history()
                self.current_candidate = None
                self.hold_count = 0
                self.last_spoken_sign = None

            self.current_sign = pred_sign
            self.current_conf = confidence
            
            # Convert BGR to RGB for Tkinter
            cv2_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            self.latest_image = Image.fromarray(cv2_rgb)

    def update_ui_loop(self):
        if not self.running:
            return
            
        # Update Camera Feed
        if hasattr(self, 'latest_image') and self.latest_image:
            # Resize proportionally to fit the label
            label_w = self.video_label.winfo_width()
            label_h = self.video_label.winfo_height()
            
            if label_w > 10 and label_h > 10:
                img_w, img_h = self.latest_image.size
                ratio = min(label_w/img_w, label_h/img_h)
                new_w = int(img_w * ratio)
                new_h = int(img_h * ratio)
                
                resized = self.latest_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                photo = ctk.CTkImage(light_image=resized, dark_image=resized, size=(new_w, new_h))
                
                self.video_label.configure(image=photo)
                self.video_label.image = photo

        # Update HUD
        if self.current_sign != "...":
            self.sign_display.configure(text=f"Sign: {self.current_sign.upper()}", text_color="#00FF96")
        else:
            self.sign_display.configure(text="Sign: ...", text_color="gray")
            
        self.conf_progress.set(self.current_conf)

        # Update Sentence
        current_sentence = self.sentence_builder.get_sentence()
        if current_sentence == "...":
            current_sentence = ""
            
        if self.sentence != current_sentence:
            self.sentence = current_sentence
            self.sentence_textbox.configure(state="normal")
            self.sentence_textbox.delete("0.0", "end")
            self.sentence_textbox.insert("0.0", self.sentence)
            self.sentence_textbox.configure(state="disabled")

        # Update Mic Status
        if self.listener:
            self.stt_label.configure(text=self.listener.status_msg)
            if not self.listener.is_listening and self.listener.last_text:
                pass # The STT text can be appended or just shown. 
                # For this design, let's keep it simple: STT status shows what was heard.
        
        self.after(30, self.update_ui_loop)

    def speak_current_sign(self):
        if self.current_sign != "...":
            self.speech.speak(self.current_sign, force=True)

    def speak_sentence(self):
        sent = self.sentence_builder.get_sentence()
        if sent != "...":
            self.speech.speak(sent, force=True)

    def delete_word(self):
        self.sentence_builder.remove_last()

    def clear_all(self):
        self.sentence_builder.clear()
        if self.listener:
            self.listener.clear()
        self.translation_textbox.delete("0.0", "end")

    def listen_mic(self):
        if self.listener and not self.listener.is_listening:
            self.listener.listen_async(duration_sec=4.0)

    def do_translation(self):
        sent = self.sentence_builder.get_sentence()
        if not sent or sent == "...":
            self.translation_textbox.delete("0.0", "end")
            self.translation_textbox.insert("0.0", "[No sentence to translate]")
            return
            
        lang = self.lang_map.get(self.target_lang.get(), "en")
        self.translation_textbox.delete("0.0", "end")
        self.translation_textbox.insert("0.0", "Translating...")
        
        def translate_worker():
            try:
                res = self.translator.translate(sent, dest=lang)
                self.after(0, lambda: self._update_translation(res.text))
            except Exception as e:
                self.after(0, lambda: self._update_translation(f"[Translation Error: {e}]"))
                
        threading.Thread(target=translate_worker, daemon=True).start()

    def _update_translation(self, text):
        self.translation_textbox.delete("0.0", "end")
        self.translation_textbox.insert("0.0", text)

    def speak_translation(self):
        text = self.translation_textbox.get("0.0", "end").strip()
        if text and not text.startswith("["):
            # SAPI might struggle with non-English, but we can pass it
            self.speech.speak(text, force=True)

    def on_closing(self):
        self.running = False
        if self.cam:
            self.cam.release()
        if self.detector:
            self.detector.close()
        if self.speech:
            self.speech.stop()
        self.destroy()


if __name__ == "__main__":
    app = JarvisDesktopApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

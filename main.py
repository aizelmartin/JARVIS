"""
main.py — J.A.R.V.I.S. | Joint AI Recognition and Voice Interpretation System
================================================================================
Application states: MAIN  <->  TRAINING  (SHUTTING_DOWN on exit)

Features
--------
- Sign recognition (MediaPipe + SVM, temporal smoothing)
- Sentence construction, googletrans translation, multilingual gTTS/pygame speech
- Hearing Person Reply: multilingual STT + auto English translation
- Integrated Sign Training: full-window workspace, safe model hot-reload
- Recognition is fully suspended (camera frames still shared) during training

Keyboard shortcuts (when no text field is focused — MAIN state only)
----------------------------------------------------------------------
  Enter      Speak sentence        Backspace  Delete last word
  C          Clear all             S          Speak current sign
  M          Microphone listen     T          Toggle Auto-Speak
"""

import os, sys, time, threading, queue, tempfile
from enum import Enum, auto

import cv2, numpy as np
from PIL import Image

import customtkinter as ctk
from customtkinter import CTkFont
from googletrans import Translator
from gtts import gTTS
import pygame

sys.path.insert(0, os.path.dirname(__file__))

from src.camera.camera_stream  import CameraStream
from src.hand_tracking.hand_detector import HandDetector
from src.ml.classifier          import SignClassifier
from src.ml.sentence_builder    import SentenceBuilder
from src.speech.speech_engine   import SpeechEngine
from src.speech.speech_listener import SpeechListener

# ── App State ─────────────────────────────────────────────────────────────── #
class AppState(Enum):
    MAIN          = auto()
    TRAINING      = auto()
    SHUTTING_DOWN = auto()

# ── Constants ─────────────────────────────────────────────────────────────── #
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "saved_models", "sign_classifier.pkl")
REQUIRED_STABLE_FRAMES = 9

BG_DARK     = "#051628"
SURF        = "#0B2B50"
SURF_VAR    = "#134275"
INPUT_BG    = "#1E5A96"
PRIMARY     = "#00E5FF"
SECONDARY   = "#9D4EDD"
ON_BG       = "#FFFFFF"
ON_BG_MUTED = "#94A3B8"
DANGER      = "#EF4444"
SUCCESS     = "#10B981"
WARN        = "#F59E0B"

LANGUAGES = [
    ("English",   "en"),
    ("Malayalam", "ml"),
    ("Hindi",     "hi"),
    ("Tamil",     "ta"),
    ("Kannada",   "kn"),
    ("Telugu",    "te"),
]
LANG_MAP    = {n: c for n, c in LANGUAGES}
GTTS_LOCALE = {"en": "en", "ml": "ml", "hi": "hi", "ta": "ta", "kn": "kn", "te": "te"}

# Google STT BCP-47 codes
STT_LANG_MAP = {
    "English":   "en-US",
    "Malayalam": "ml-IN",
    "Hindi":     "hi-IN",
    "Tamil":     "ta-IN",
    "Kannada":   "kn-IN",
    "Telugu":    "te-IN",
}

# ── gTTS serialised worker ─────────────────────────────────────────────────── #
pygame.mixer.init()
_gtts_queue: queue.Queue = queue.Queue(maxsize=2)
_gtts_busy = False

def _gtts_worker():
    global _gtts_busy
    while True:
        text, lang, on_done, on_error = _gtts_queue.get()
        _gtts_busy = True
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = f.name
            tts.save(tmp)
            pygame.mixer.music.load(tmp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.unload()
            try: os.remove(tmp)
            except: pass
            if on_done: on_done()
        except Exception as e:
            if on_error: on_error(str(e))
        finally:
            _gtts_busy = False
            _gtts_queue.task_done()

threading.Thread(target=_gtts_worker, daemon=True, name="gTTS-Worker").start()

def speak_gtts(text: str, lang: str = "en", on_done=None, on_error=None, force: bool = False):
    text = text.strip()
    if not text or text == "...": return
    try:
        if force and not _gtts_queue.empty():
            try: _gtts_queue.get_nowait()
            except queue.Empty: pass
        _gtts_queue.put_nowait((text, lang, on_done, on_error))
    except queue.Full: pass

# ── Theme ─────────────────────────────────────────────────────────────────── #
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ─────────────────────────────────────────────────────────────────────────────
class JarvisApp(ctk.CTk):

    # ─── Init ─────────────────────────────────────────────────────────── #
    def __init__(self):
        super().__init__()
        self.title("J.A.R.V.I.S.")
        self.geometry("1280x820")
        self.minsize(1060, 700)
        self.configure(fg_color=BG_DARK)

        self._state = AppState.MAIN

        # Camera
        self._running    = True
        self._cam_active = False

        # Recognition
        self._current_sign  = "..."
        self._current_conf  = 0.0
        self._hold_ratio    = 0.0
        self._candidate     = None
        self._hold_count    = 0
        self._last_added    = None
        self._last_added_t  = 0.0

        # Shared frame (thread-safe)
        self._frame_lock      = threading.Lock()
        self._latest_frame    = None
        self._latest_features = None
        self._latest_n_hands  = 0

        # UI dirty-check cache (avoids redundant configure calls)
        self._dc_sign   = None
        self._dc_conf   = -1.0
        self._dc_hold   = -1.0
        self._dc_sent   = ""
        self._dc_stt    = ""
        self._dc_stt_st = ""
        self._dc_tts    = ""
        self._dc_trans  = ""

        # Flags
        self._auto_speak_saved = True
        self._auto_speak    = ctk.BooleanVar(value=True)
        self._stt_locked    = False
        self._trans_locked  = False
        self._tts_status    = ""
        self._trans_status  = ""

        # Pipeline objects
        self.cam              = None
        self.detector         = None
        self.classifier       = None
        self.sentence_builder = None
        self.sapi_speech      = None
        self.listener         = None
        self.translator       = Translator()

        # Translation
        self._trans_source  = ctk.StringVar(value="Sign Sentence")
        self._target_lang   = ctk.StringVar(value="Malayalam")
        self._stt_lang      = ctk.StringVar(value="English")
        self._stt_raw_text  = ""

        # Training
        self._train_samples      = []
        self._train_target       = 200
        self._train_recording    = False
        self._train_last_capture = 0.0

        self._build_ui()
        self._bind_shortcuts()
        self._start_pipeline()
        self._ui_loop()

    # ─── UI BUILD ─────────────────────────────────────────────────────── #
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=6, minsize=640)
        self.grid_columnconfigure(1, weight=4, minsize=400)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_main_content()

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=SURF, height=68, corner_radius=0)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        hdr.grid_propagate(False)
        hdr.grid_columnconfigure(1, weight=1)

        logo = ctk.CTkFrame(hdr, fg_color="transparent")
        logo.grid(row=0, column=0, padx=20, pady=8, sticky="w")
        ctk.CTkLabel(logo, text="J.A.R.V.I.S.", font=CTkFont("Arial", 24, "bold"),
                     text_color=PRIMARY).pack(side="left")
        ctk.CTkLabel(logo, text=" | Joint AI Recognition and Voice Interpretation System",
                     font=CTkFont("Arial", 12), text_color=SECONDARY).pack(side="left", padx=(4,0))

        ctk.CTkLabel(hdr, text="“Connecting people, beyond words.”",
                     font=CTkFont("Arial", 13, slant="italic"),
                     text_color=ON_BG_MUTED).grid(row=0, column=1, padx=8, sticky="w")

        sf = ctk.CTkFrame(hdr, fg_color="transparent")
        sf.grid(row=0, column=2, padx=20, pady=8, sticky="e")
        self._cam_dot   = ctk.CTkLabel(sf, text="●", font=CTkFont(size=14), text_color="gray")
        self._cam_dot.pack(side="left", padx=(0,4))
        self._cam_label = ctk.CTkLabel(sf, text="Camera OFF", text_color=ON_BG_MUTED)
        self._cam_label.pack(side="left", padx=(0,20))
        self._rec_dot   = ctk.CTkLabel(sf, text="●", font=CTkFont(size=14), text_color="gray")
        self._rec_dot.pack(side="left", padx=(0,4))
        self._rec_label = ctk.CTkLabel(sf, text="Idle", text_color=ON_BG_MUTED)
        self._rec_label.pack(side="left")

    def _build_main_content(self):
        self._left_col = ctk.CTkFrame(self, fg_color="transparent")
        self._left_col.grid(row=1, column=0, sticky="nsew", padx=(16,8), pady=(0,16))
        self._left_col.grid_rowconfigure(0, weight=1)
        self._left_col.grid_rowconfigure(1, weight=0)
        self._left_col.grid_columnconfigure(0, weight=1)
        self._build_camera_card(self._left_col)
        self._build_recognition_card(self._left_col)

        self._right_col = ctk.CTkScrollableFrame(
            self, fg_color=BG_DARK,
            scrollbar_button_color=SURF_VAR,
            scrollbar_button_hover_color=PRIMARY,
        )
        self._right_col.grid(row=1, column=1, sticky="nsew", padx=(8,16), pady=(0,16))
        self._right_col.grid_columnconfigure(0, weight=1)
        self._build_sentence_card(self._right_col)
        self._build_translation_card(self._right_col)
        self._build_stt_card(self._right_col)
        self._build_train_entry_card(self._right_col)
        self._build_help_card(self._right_col)

    # ── Camera card ── #
    def _build_camera_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=SURF, corner_radius=12)
        card.grid(row=0, column=0, sticky="nsew", pady=(0,8))
        card.grid_rowconfigure(1, weight=1)
        card.grid_columnconfigure(0, weight=1)

        tr = ctk.CTkFrame(card, fg_color="transparent")
        tr.grid(row=0, column=0, sticky="ew", padx=16, pady=(12,6))
        tr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(tr, text="Live Camera", font=CTkFont("Arial",16,"bold"),
                     text_color=ON_BG).grid(row=0, column=0, sticky="w")
        tf = ctk.CTkFrame(tr, fg_color="transparent")
        tf.grid(row=0, column=1, sticky="e")
        ctk.CTkLabel(tf, text="Camera:", text_color=ON_BG_MUTED).pack(side="left", padx=(0,6))
        self._cam_switch = ctk.CTkSwitch(tf, text="", variable=ctk.BooleanVar(value=False),
                                         onvalue=True, offvalue=False,
                                         progress_color=PRIMARY, command=self._toggle_camera, width=46)
        self._cam_switch.pack(side="left")

        self._video_label = ctk.CTkLabel(card, text="", fg_color="black", corner_radius=8)
        self._video_label.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0,16))

        self._cam_off_label = ctk.CTkLabel(card,
            text="📷\nCamera is OFF\nToggle the switch above to start",
            font=CTkFont("Arial",14), text_color=ON_BG_MUTED,
            fg_color="#001830", corner_radius=8)
        self._cam_off_label.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0,16))

    # ── Recognition card ── #
    def _build_recognition_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=SURF, corner_radius=12)
        card.grid(row=1, column=0, sticky="ew")
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(card, text="Sign Recognition",
                     font=CTkFont("Arial",15,"bold"), text_color=ON_BG
                     ).grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(10,2))

        self._sign_label = ctk.CTkLabel(card, text="...", font=CTkFont("Arial",30,"bold"),
                                         text_color=ON_BG_MUTED)
        self._sign_label.grid(row=1, column=0, sticky="w", padx=16, pady=(2,0))
        self._conf_pct_label = ctk.CTkLabel(card, text="", text_color=ON_BG_MUTED)
        self._conf_pct_label.grid(row=1, column=1, sticky="e", padx=16)

        cr = ctk.CTkFrame(card, fg_color="transparent")
        cr.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(2,0))
        cr.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(cr, text="Confidence", text_color=ON_BG_MUTED, font=CTkFont(size=11)).grid(row=0, column=0, sticky="w")
        self._conf_bar = ctk.CTkProgressBar(cr, height=7, progress_color=SECONDARY, fg_color=BG_DARK)
        self._conf_bar.set(0)
        self._conf_bar.grid(row=0, column=1, sticky="ew", padx=(8,0))

        hr = ctk.CTkFrame(card, fg_color="transparent")
        hr.grid(row=3, column=0, columnspan=2, sticky="ew", padx=16, pady=(4,0))
        hr.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(hr, text="Hold to Confirm", text_color=ON_BG_MUTED, font=CTkFont(size=11)).grid(row=0, column=0, sticky="w")
        self._hold_bar = ctk.CTkProgressBar(hr, height=7, progress_color=WARN, fg_color=BG_DARK)
        self._hold_bar.set(0)
        self._hold_bar.grid(row=0, column=1, sticky="ew", padx=(8,0))

        bot = ctk.CTkFrame(card, fg_color="transparent")
        bot.grid(row=4, column=0, columnspan=2, sticky="ew", padx=16, pady=(8,10))
        bot.grid_columnconfigure(0, weight=1)
        ctk.CTkCheckBox(bot, text="Auto Speak", variable=self._auto_speak,
                        text_color=ON_BG, checkmark_color=BG_DARK, fg_color=PRIMARY
                        ).grid(row=0, column=0, sticky="w")
        self._speak_sign_btn = ctk.CTkButton(bot, text="▶  Speak Sign",
            fg_color=PRIMARY, hover_color=SECONDARY, text_color=BG_DARK,
            font=CTkFont(weight="bold"), command=self._speak_current_sign, width=130, height=32)
        self._speak_sign_btn.grid(row=0, column=1, sticky="e")

    # ── Sentence card ── #
    def _build_sentence_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=SURF, corner_radius=12)
        card.pack(fill="x", pady=(0,10))
        ctk.CTkLabel(card, text="Constructed Sentence",
                     font=CTkFont("Arial",15,"bold"), text_color=ON_BG
                     ).pack(anchor="w", padx=16, pady=(12,6))
        self._sentence_box = ctk.CTkTextbox(card, height=90, font=CTkFont("Arial",17),
            fg_color=INPUT_BG, text_color=ON_BG, border_color=PRIMARY, border_width=1,
            state="disabled", corner_radius=8)
        self._sentence_box.pack(fill="x", padx=16, pady=(0,10))

        br = ctk.CTkFrame(card, fg_color="transparent")
        br.pack(fill="x", padx=16, pady=(0,12))
        br.grid_columnconfigure((0,1,2), weight=1)
        ctk.CTkButton(br, text="⌫  Delete Word", fg_color=SURF_VAR, hover_color=SURF,
            text_color=ON_BG, border_color=PRIMARY, border_width=1,
            command=self._delete_word, height=34).grid(row=0, column=0, sticky="ew", padx=(0,4))
        ctk.CTkButton(br, text="✕  Clear All", fg_color=DANGER, hover_color="#B71C1C",
            text_color=ON_BG, command=self._clear_all, height=34
            ).grid(row=0, column=1, sticky="ew", padx=4)
        ctk.CTkButton(br, text="▶  Speak Sentence", fg_color=PRIMARY, hover_color=SECONDARY,
            text_color=BG_DARK, font=CTkFont(weight="bold"), command=self._speak_sentence, height=34
            ).grid(row=0, column=2, sticky="ew", padx=(4,0))

    # ── Translation card ── #
    def _build_translation_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=SURF, corner_radius=12)
        card.pack(fill="x", pady=(0,10))
        ctk.CTkLabel(card, text="Translation & Speech",
                     font=CTkFont("Arial",15,"bold"), text_color=ON_BG
                     ).pack(anchor="w", padx=16, pady=(12,6))

        sr2 = ctk.CTkFrame(card, fg_color="transparent")
        sr2.pack(fill="x", padx=16, pady=(0,8))
        ctk.CTkLabel(sr2, text="Source:", text_color=ON_BG_MUTED).pack(side="left")
        for lbl in ("Sign Sentence", "Speech Text"):
            ctk.CTkRadioButton(sr2, text=lbl, variable=self._trans_source, value=lbl,
                               text_color=ON_BG, fg_color=PRIMARY).pack(side="left", padx=(10,0))

        lr = ctk.CTkFrame(card, fg_color="transparent")
        lr.pack(fill="x", padx=16, pady=(0,8))
        lr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(lr, text="Translate to:", text_color=ON_BG_MUTED).grid(row=0, column=0, sticky="w")
        self._lang_menu = ctk.CTkOptionMenu(lr, variable=self._target_lang,
            values=[n for n,_ in LANGUAGES], fg_color=INPUT_BG, button_color=INPUT_BG,
            button_hover_color=PRIMARY, dropdown_fg_color=SURF_VAR, text_color=ON_BG)
        self._lang_menu.grid(row=1, column=0, sticky="ew", pady=(4,0))
        self._translate_btn = ctk.CTkButton(lr, text="Translate", width=110,
            fg_color="#0277BD", hover_color=PRIMARY, text_color=ON_BG,
            command=self._do_translate, height=34)
        self._translate_btn.grid(row=1, column=1, sticky="e", padx=(8,0), pady=(4,0))

        self._trans_status_label = ctk.CTkLabel(card, text="", text_color=ON_BG_MUTED, font=CTkFont(size=12))
        self._trans_status_label.pack(anchor="w", padx=16)
        ctk.CTkLabel(card, text="Translated Text:", text_color=ON_BG_MUTED, font=CTkFont(size=12)
                     ).pack(anchor="w", padx=16, pady=(6,2))
        self._trans_box = ctk.CTkTextbox(card, height=75, font=CTkFont("Arial",16),
            fg_color=INPUT_BG, text_color=ON_BG, border_color=PRIMARY, border_width=1, corner_radius=8)
        self._trans_box.pack(fill="x", padx=16, pady=(0,8))

        self._speak_trans_btn = ctk.CTkButton(card, text="▶  Speak Translation",
            fg_color=PRIMARY, hover_color=SECONDARY, text_color=BG_DARK,
            font=CTkFont(weight="bold"), command=self._speak_translation, height=36)
        self._speak_trans_btn.pack(fill="x", padx=16, pady=(0,6))
        self._tts_status_label = ctk.CTkLabel(card, text="", text_color=ON_BG_MUTED, font=CTkFont(size=12))
        self._tts_status_label.pack(anchor="w", padx=16, pady=(0,10))

    # ── Hearing Person card ── #
    def _build_stt_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=SURF, corner_radius=12)
        card.pack(fill="x", pady=(0,10))
        ctk.CTkLabel(card, text="Hearing Person Reply",
                     font=CTkFont("Arial",15,"bold"), text_color=ON_BG
                     ).pack(anchor="w", padx=16, pady=(12,4))

        # Language selector
        lgr = ctk.CTkFrame(card, fg_color="transparent")
        lgr.pack(fill="x", padx=16, pady=(0,6))
        lgr.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(lgr, text="Spoken language:", text_color=ON_BG_MUTED,
                     font=CTkFont(size=12)).grid(row=0, column=0, sticky="w", padx=(0,8))
        self._stt_lang_menu = ctk.CTkOptionMenu(lgr, variable=self._stt_lang,
            values=[n for n,_ in LANGUAGES], fg_color=INPUT_BG, button_color=INPUT_BG,
            button_hover_color=PRIMARY, dropdown_fg_color=SURF_VAR, text_color=ON_BG, width=170)
        self._stt_lang_menu.grid(row=0, column=1, sticky="ew")

        # Listen row
        mr = ctk.CTkFrame(card, fg_color="transparent")
        mr.pack(fill="x", padx=16, pady=(0,8))
        self._mic_btn = ctk.CTkButton(mr, text="🎙  Listen",
            fg_color=SURF_VAR, hover_color=PRIMARY, text_color=ON_BG,
            border_color=PRIMARY, border_width=1, command=self._listen_mic, width=110, height=34)
        self._mic_btn.pack(side="left")
        self._stt_status_label = ctk.CTkLabel(mr, text="Press Listen or M", text_color=ON_BG_MUTED)
        self._stt_status_label.pack(side="left", padx=12)

        # Recognized text (raw)
        ctk.CTkLabel(card, text="Recognized text:", text_color=ON_BG_MUTED, font=CTkFont(size=12)
                     ).pack(anchor="w", padx=16, pady=(0,2))
        self._stt_box = ctk.CTkTextbox(card, height=50, font=CTkFont("Arial",14),
            fg_color=INPUT_BG, text_color=ON_BG, border_color=INPUT_BG, border_width=1,
            corner_radius=8, state="disabled")
        self._stt_box.pack(fill="x", padx=16, pady=(0,6))

        # English translation of heard speech
        ctk.CTkLabel(card, text="English translation:", text_color=ON_BG_MUTED, font=CTkFont(size=12)
                     ).pack(anchor="w", padx=16, pady=(0,2))
        self._stt_trans_box = ctk.CTkTextbox(card, height=50, font=CTkFont("Arial",14),
            fg_color=INPUT_BG, text_color=ON_BG, border_color=INPUT_BG, border_width=1, corner_radius=8)
        self._stt_trans_box.pack(fill="x", padx=16, pady=(0,6))

        # Action buttons
        ar = ctk.CTkFrame(card, fg_color="transparent")
        ar.pack(fill="x", padx=16, pady=(0,12))
        ar.grid_columnconfigure((0,1), weight=1)
        ctk.CTkButton(ar, text="Use as Translation Source",
            fg_color=SURF_VAR, hover_color=SURF, text_color=PRIMARY,
            border_color=PRIMARY, border_width=1,
            command=self._use_stt_as_source, height=32
            ).grid(row=0, column=0, sticky="ew", padx=(0,4))
        self._speak_stt_trans_btn = ctk.CTkButton(ar, text="▶  Speak English",
            fg_color=PRIMARY, hover_color=SECONDARY, text_color=BG_DARK,
            font=CTkFont(weight="bold"), command=self._speak_stt_translation, height=32)
        self._speak_stt_trans_btn.grid(row=0, column=1, sticky="ew", padx=(4,0))

    # ── Train entry card ── #
    def _build_train_entry_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=SURF, corner_radius=12)
        card.pack(fill="x", pady=(0,10))
        ctk.CTkLabel(card, text="Train New Sign",
                     font=CTkFont("Arial",15,"bold"), text_color=ON_BG
                     ).pack(anchor="w", padx=16, pady=(12,4))
        ctk.CTkLabel(card,
            text="Collect hand-landmark samples for a new sign and retrain the classifier.\n"
                 "Camera must be ON. Turn on the camera first.",
            text_color=ON_BG_MUTED, font=CTkFont(size=12), justify="left"
            ).pack(anchor="w", padx=16, pady=(0,8))
        ctk.CTkButton(card, text="➕  Open Training Workspace",
            fg_color=SUCCESS, hover_color="#059669",
            text_color=BG_DARK, font=CTkFont(weight="bold"),
            command=self._open_training_workspace, height=38
            ).pack(fill="x", padx=16, pady=(0,14))

    # ── Keyboard shortcuts (compact 2-column grid) ── #
    def _build_help_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=SURF, corner_radius=12)
        card.pack(fill="x", pady=(0,10))
        ctk.CTkLabel(card, text="Keyboard Shortcuts",
                     font=CTkFont("Arial",14,"bold"), text_color=ON_BG
                     ).pack(anchor="w", padx=16, pady=(10,4))
        ctk.CTkLabel(card, text="Active on main screen when no text field is focused:",
                     text_color=ON_BG_MUTED, font=CTkFont(size=11)
                     ).pack(anchor="w", padx=16, pady=(0,6))

        pairs = [
            ("Enter",     "Speak sentence",    "Backspace", "Delete last word"),
            ("C",         "Clear all",         "S",         "Speak current sign"),
            ("M",         "Microphone listen", "T",         "Toggle Auto-Speak"),
        ]
        gf = ctk.CTkFrame(card, fg_color="transparent")
        gf.pack(fill="x", padx=16, pady=(0,12))
        gf.grid_columnconfigure((0,1,2,3), weight=1)
        for r,(k1,d1,k2,d2) in enumerate(pairs):
            ctk.CTkLabel(gf, text=k1, font=CTkFont(size=12, weight="bold"),
                         fg_color=SURF_VAR, corner_radius=4, text_color=PRIMARY, width=80
                         ).grid(row=r, column=0, sticky="ew", padx=(0,2), pady=2)
            ctk.CTkLabel(gf, text=d1, text_color=ON_BG_MUTED, font=CTkFont(size=12), anchor="w"
                         ).grid(row=r, column=1, sticky="w", padx=(4,12), pady=2)
            ctk.CTkLabel(gf, text=k2, font=CTkFont(size=12, weight="bold"),
                         fg_color=SURF_VAR, corner_radius=4, text_color=PRIMARY, width=80
                         ).grid(row=r, column=2, sticky="ew", padx=(0,2), pady=2)
            ctk.CTkLabel(gf, text=d2, text_color=ON_BG_MUTED, font=CTkFont(size=12), anchor="w"
                         ).grid(row=r, column=3, sticky="w", padx=(4,0), pady=2)
        ctk.CTkFrame(card, height=4, fg_color="transparent").pack()

    # ─── Keyboard shortcuts ────────────────────────────────────────────── #
    def _bind_shortcuts(self):
        self.bind("<Return>",    self._sh_enter)
        self.bind("<BackSpace>", self._sh_backspace)
        self.bind("<c>",         self._sh_c)
        self.bind("<C>",         self._sh_c)
        self.bind("<m>",         self._sh_m)
        self.bind("<M>",         self._sh_m)
        self.bind("<s>",         self._sh_s)
        self.bind("<S>",         self._sh_s)
        self.bind("<t>",         self._sh_t)
        self.bind("<T>",         self._sh_t)

    def _in_text_field(self) -> bool:
        try:
            w = str(self.focus_get())
            return any(x in w for x in ("entry", "text", "textbox"))
        except: return False

    def _sh_enter(self, _=None):
        if self._state == AppState.MAIN and not self._in_text_field(): self._speak_sentence()
    def _sh_backspace(self, _=None):
        if self._state == AppState.MAIN and not self._in_text_field(): self._delete_word()
    def _sh_c(self, _=None):
        if self._state == AppState.MAIN and not self._in_text_field(): self._clear_all()
    def _sh_m(self, _=None):
        if self._state == AppState.MAIN and not self._in_text_field(): self._listen_mic()
    def _sh_s(self, _=None):
        if self._state == AppState.MAIN and not self._in_text_field(): self._speak_current_sign()
    def _sh_t(self, _=None):
        if self._state == AppState.MAIN and not self._in_text_field():
            self._auto_speak.set(not self._auto_speak.get())

    # ─── Pipeline ─────────────────────────────────────────────────────── #
    def _start_pipeline(self):
        self._update_status(cam=False)
        try:
            if not os.path.exists(MODEL_PATH):
                self._show_error(f"Model not found:\n{MODEL_PATH}\n\nRun: python src/ml/train_models.py")
                return
            self.detector         = HandDetector(max_hands=2, min_detection_conf=0.7)
            self.classifier       = SignClassifier(MODEL_PATH, smooth_window=5, confidence_threshold=0.65)
            self.sentence_builder = SentenceBuilder(max_words=100)
            self.sapi_speech      = SpeechEngine(rate=2, volume=100)
            self.listener         = SpeechListener()
        except Exception as e:
            self._show_error(f"Pipeline init error: {e}")

    # ─── Camera ───────────────────────────────────────────────────────── #
    def _toggle_camera(self):
        if self._cam_active: self._stop_camera()
        else: self._open_camera()

    def _open_camera(self):
        if self._cam_active: return
        try:
            self.cam = CameraStream(camera_index=0, width=640, height=480)
            if not self.cam.start():
                self._cam_off_label.configure(text="⚠  Camera failed to open.")
                try: self._train_cam_off_label.configure(text="⚠  Camera failed to open.")
                except: pass
                return
            self._cam_active = True
            self._cam_off_label.lower(self._video_label)
            try: self._train_cam_off_label.lower(self._train_video_label)
            except: pass
            try: self._cam_switch.select()
            except: pass
            try: self._train_cam_switch.select()
            except: pass
            self._update_status(cam=True)
            threading.Thread(target=self._camera_worker, daemon=True, name="CamWorker").start()
        except Exception as e:
            self._show_error(f"Camera error: {e}")

    def _stop_camera(self):
        self._cam_active = False
        if self.cam: self.cam.release(); self.cam = None
        if self.classifier: self.classifier.reset_history()
        self._current_sign = "..."; self._current_conf = 0.0; self._hold_ratio = 0.0
        self._candidate = None; self._hold_count = 0
        self._dc_sign = None  # Force UI update
        try: self._cam_off_label.lift()
        except: pass
        try: self._train_cam_off_label.lift()
        except: pass
        try: self._cam_switch.deselect()
        except: pass
        try: self._train_cam_switch.deselect()
        except: pass
        self._update_status(cam=False)

    def _update_status(self, cam: bool):
        if cam:
            self._cam_dot.configure(text_color=SUCCESS)
            self._cam_label.configure(text="Camera ON", text_color=SUCCESS)
        else:
            self._cam_dot.configure(text_color="gray")
            self._cam_label.configure(text="Camera OFF", text_color=ON_BG_MUTED)

    # ─── Camera worker ────────────────────────────────────────────────── #
    def _camera_worker(self):
        """Background thread. Recognition runs only in MAIN state."""
        while self._cam_active and self._state != AppState.SHUTTING_DOWN:
            if self.cam is None: break
            frame = self.cam.read_frame()
            if frame is None: time.sleep(0.02); continue

            display = frame.copy()
            self.detector.process_frame(display)
            self.detector.draw_landmarks(display)
            n_hands      = self.detector.num_hands_detected()
            features_126 = self.detector.extract_features_both()
            pred_sign    = "..."
            confidence   = 0.0

            if self._state == AppState.MAIN:
                if n_hands > 0:
                    pred_sign, confidence = self.classifier.predict_smooth(features_126)
                    if pred_sign != "..." and confidence >= 0.70:
                        if pred_sign == self._candidate:
                            self._hold_count += 1
                        else:
                            self._candidate  = pred_sign
                            self._hold_count = 1
                        if self._hold_count >= REQUIRED_STABLE_FRAMES:
                            now = time.time()
                            if (pred_sign != self._last_added or (now - self._last_added_t) > 2.5):
                                self.sentence_builder.add_word(pred_sign)
                                if self._auto_speak.get():
                                    self.sapi_speech.speak(pred_sign)
                                self._last_added   = pred_sign
                                self._last_added_t = now
                    else:
                        self._candidate  = None
                        self._hold_count = 0
                else:
                    self.classifier.reset_history()
                    self._candidate  = None
                    self._hold_count = 0
                    self._last_added = None

                self._current_sign = pred_sign
                self._current_conf = confidence
                self._hold_ratio   = (
                    min(1.0, self._hold_count / REQUIRED_STABLE_FRAMES)
                    if (self._candidate and self._hold_count > 0) else 0.0)

            # Always store frame for UI
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            with self._frame_lock:
                self._latest_frame    = img
                self._latest_features = features_126
                self._latest_n_hands  = n_hands

    # ─── UI refresh loop ──────────────────────────────────────────────── #
    def _ui_loop(self):
        if self._state == AppState.SHUTTING_DOWN: return

        if self._cam_active and self._state == AppState.MAIN:
            with self._frame_lock:
                img = self._latest_frame
            if img:
                lw = self._video_label.winfo_width()
                lh = self._video_label.winfo_height()
                if lw > 10 and lh > 10:
                    iw, ih  = img.size
                    ratio   = min(lw / iw, lh / ih)
                    nw, nh  = int(iw * ratio), int(ih * ratio)
                    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
                    photo   = ctk.CTkImage(light_image=resized, dark_image=resized, size=(nw, nh))
                    self._video_label.configure(image=photo)
                    self._video_label.image = photo

        # Recognition (dirty-check)
        sign = self._current_sign
        conf = round(self._current_conf, 3)
        hold = round(self._hold_ratio, 3)

        if sign != self._dc_sign:
            self._dc_sign = sign
            if sign != "...":
                self._sign_label.configure(text=sign.upper(), text_color=SECONDARY)
                self._rec_dot.configure(text_color=SUCCESS)
                self._rec_label.configure(text="Detecting", text_color=SUCCESS)
            else:
                self._sign_label.configure(text="...", text_color=ON_BG_MUTED)
                if self._cam_active and self._state == AppState.MAIN:
                    self._rec_dot.configure(text_color=WARN)
                    self._rec_label.configure(text="Watching...", text_color=WARN)
                elif self._state == AppState.TRAINING and self._cam_active:
                    self._rec_dot.configure(text_color=PRIMARY)
                    self._rec_label.configure(text="Training", text_color=PRIMARY)
                else:
                    self._rec_dot.configure(text_color="gray")
                    self._rec_label.configure(text="Idle", text_color=ON_BG_MUTED)

        if conf != self._dc_conf:
            self._dc_conf = conf
            self._conf_pct_label.configure(text=f"{conf*100:.0f}%" if sign != "..." else "")
            self._conf_bar.set(conf)

        if hold != self._dc_hold:
            self._dc_hold = hold
            self._hold_bar.set(hold)

        if self.sentence_builder:
            sent = self.sentence_builder.get_sentence()
            disp = sent if sent != "..." else ""
            if disp != self._dc_sent:
                self._dc_sent = disp
                self._sentence_box.configure(state="normal")
                self._sentence_box.delete("0.0", "end")
                if disp: self._sentence_box.insert("0.0", disp)
                self._sentence_box.configure(state="disabled")

        if self.listener:
            st = self.listener.status_msg
            tx = self.listener.last_text
            if st != self._dc_stt_st:
                self._dc_stt_st = st
                self._stt_status_label.configure(text=st)
            if tx != self._dc_stt:
                self._dc_stt = tx
                self._stt_box.configure(state="normal")
                self._stt_box.delete("0.0", "end")
                if tx: self._stt_box.insert("0.0", tx)
                self._stt_box.configure(state="disabled")

        if self._tts_status != self._dc_tts:
            self._dc_tts = self._tts_status
            self._tts_status_label.configure(text=self._tts_status)

        if self._trans_status != self._dc_trans:
            self._dc_trans = self._trans_status
            self._trans_status_label.configure(text=self._trans_status)

        self.after(33, self._ui_loop)

    # ─── Actions ──────────────────────────────────────────────────────── #
    def _speak_current_sign(self):
        if self._current_sign and self._current_sign != "...":
            self.sapi_speech.speak(self._current_sign, force=True)

    def _speak_sentence(self):
        if self.sentence_builder:
            s = self.sentence_builder.get_sentence()
            if s and s != "...": self.sapi_speech.speak(s, force=True)

    def _delete_word(self):
        if self.sentence_builder: self.sentence_builder.remove_last()

    def _clear_all(self):
        if self.sentence_builder: self.sentence_builder.clear()
        if self.listener: self.listener.clear()
        self._stt_raw_text = ""
        try:
            self._stt_status_label.configure(text="Press Listen or M")
            self._stt_trans_box.delete("0.0", "end")
        except: pass
        self._trans_box.delete("0.0", "end")
        self._trans_status = ""
        self._tts_status   = ""

    # ─── Multilingual STT ─────────────────────────────────────────────── #
    def _listen_mic(self):
        if self._stt_locked or (self.listener and self.listener.is_listening): return
        if not self.listener: return

        lang_name = self._stt_lang.get()
        bcp47     = STT_LANG_MAP.get(lang_name, "en-US")
        self._stt_locked = True

        def _run():
            import sounddevice as sd
            import speech_recognition as sr
            self.listener.is_listening = True
            self.listener.status_msg   = f"🎙 Listening ({lang_name})..."
            try:
                dur         = 5.0
                sr_rate     = self.listener.sample_rate
                buf         = sd.rec(int(dur * sr_rate), samplerate=sr_rate, channels=1, dtype="int16")
                sd.wait()
                self.listener.status_msg = "Transcribing..."
                aud = sr.AudioData(buf.tobytes(), sr_rate, 2)
                try:
                    text = self.listener.recognizer.recognize_google(aud, language=bcp47)
                    self.listener.last_text = text
                    self._stt_raw_text      = text
                    self.listener.status_msg = f"Heard: '{text}'"
                    if lang_name != "English" and text:
                        self.after(0, lambda t=text: self._auto_translate_to_english(t))
                    else:
                        self.after(0, lambda t=text: self._set_stt_english_box(t))
                except sr.UnknownValueError:
                    e2 = np.sqrt(np.mean(buf.astype(np.float32) ** 2))
                    self.listener.status_msg = (
                        "No speech detected (mic too quiet)" if e2 < 50 else "Could not understand audio")
                except sr.RequestError as e:
                    self.listener.status_msg = f"Speech service error: {e}"
            except Exception as ex:
                self.listener.status_msg = f"Mic error: {ex}"
            finally:
                self.listener.is_listening = False
                self._stt_locked = False

        threading.Thread(target=_run, daemon=True, name="STTWorker").start()

    def _auto_translate_to_english(self, text: str):
        def _w():
            try:
                r = self.translator.translate(text, dest="en")
                self.after(0, lambda t=r.text: self._set_stt_english_box(t))
            except Exception as e:
                self.after(0, lambda: self._set_stt_english_box(f"[Translation error: {e}]"))
        threading.Thread(target=_w, daemon=True, name="STT-TransWorker").start()

    def _set_stt_english_box(self, text: str):
        try:
            self._stt_trans_box.delete("0.0", "end")
            if text: self._stt_trans_box.insert("0.0", text)
        except: pass

    def _speak_stt_translation(self):
        text = self._stt_trans_box.get("0.0", "end").strip()
        if not text or text.startswith("["): return
        speak_gtts(text, "en", force=True)

    def _use_stt_as_source(self):
        self._trans_source.set("Speech Text")

    # ─── Sign-sentence translation & TTS ──────────────────────────────── #
    def _do_translate(self):
        if self._trans_locked: return
        src = self._trans_source.get()
        if src == "Sign Sentence" and self.sentence_builder:
            text = self.sentence_builder.get_sentence()
            if text == "...": text = ""
        else:
            text = (self.listener.last_text if self.listener else "").strip()
        if not text: self._trans_status = "⚠  Nothing to translate."; return

        lang_name = self._target_lang.get()
        lang_code = LANG_MAP.get(lang_name, "ml")
        self._trans_locked = True
        self._trans_status = "Translating…"
        self._translate_btn.configure(state="disabled")

        def _w():
            try:
                r = self.translator.translate(text, dest=lang_code)
                self.after(0, lambda: self._on_translate_done(r.text, lang_name))
            except Exception as e:
                self.after(0, lambda: self._on_translate_error(str(e)))
        threading.Thread(target=_w, daemon=True).start()

    def _on_translate_done(self, translated: str, lang_name: str):
        self._trans_box.delete("0.0", "end")
        self._trans_box.insert("0.0", translated)
        self._trans_status = f"✓  Translated to {lang_name}"
        self._trans_locked = False
        self._translate_btn.configure(state="normal")

    def _on_translate_error(self, err: str):
        self._trans_status = f"⚠  Translation failed: {err}"
        self._trans_locked = False
        self._translate_btn.configure(state="normal")

    def _speak_translation(self):
        text = self._trans_box.get("0.0", "end").strip()
        if not text or text.startswith("⚠") or text.startswith("["):
            self._tts_status = "⚠  No translation text to speak."; return
        lang_name = self._target_lang.get()
        lang_code = LANG_MAP.get(lang_name, "ml")
        gtts_lang = GTTS_LOCALE.get(lang_code, "en")
        self._tts_status = f"🔊 Speaking ({lang_name})…"
        self._speak_trans_btn.configure(state="disabled")
        def _on_done():
            self.after(0, lambda: self._tts_status_label.configure(text=f"✓  Spoken ({lang_name})", text_color=SUCCESS))
            self.after(0, lambda: self._speak_trans_btn.configure(state="normal"))
            self._tts_status = ""
        def _on_error(e):
            self.after(0, lambda: self._tts_status_label.configure(text=f"⚠  TTS error: {e}", text_color=DANGER))
            self.after(0, lambda: self._speak_trans_btn.configure(state="normal"))
            self._tts_status = ""
        speak_gtts(text, gtts_lang, on_done=_on_done, on_error=_on_error, force=True)

    # ─── Error dialog ─────────────────────────────────────────────────── #
    def _show_error(self, msg: str):
        print(f"[JARVIS] ERROR: {msg}")
        try:
            w = ctk.CTkToplevel(self)
            w.title("J.A.R.V.I.S. Error")
            w.geometry("480x180")
            w.configure(fg_color=SURF)
            ctk.CTkLabel(w, text="⚠  "+msg, text_color=DANGER, wraplength=440).pack(padx=20, pady=30)
            ctk.CTkButton(w, text="OK", command=w.destroy, fg_color=DANGER).pack()
        except: pass

    # ─── Training Workspace (full-window) ─────────────────────────────── #
    def _open_training_workspace(self):        # Suspend recognition
        self._state = AppState.TRAINING
        self._dc_sign = None  # Force UI update
        self._auto_speak_saved = self._auto_speak.get()
        self._auto_speak.set(False)
        self._current_sign = "..."; self._current_conf = 0.0; self._hold_ratio = 0.0
        self._candidate = None; self._hold_count = 0
        if self.classifier: self.classifier.reset_history()

        # Reset training state
        self._train_samples      = []
        self._train_target       = 200
        self._train_recording    = False
        self._train_last_capture = 0.0

        # Hide main columns
        self._left_col.grid_remove()
        self._right_col.grid_remove()

        # Full-area training frame
        self._train_frame = ctk.CTkFrame(self, fg_color=BG_DARK)
        self._train_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=0, pady=0)
        self._train_frame.grid_columnconfigure(0, weight=6)
        self._train_frame.grid_columnconfigure(1, weight=4)
        self._train_frame.grid_rowconfigure(0, weight=0)
        self._train_frame.grid_rowconfigure(1, weight=1)

        # Header bar
        thdr = ctk.CTkFrame(self._train_frame, fg_color=SURF_VAR, height=44, corner_radius=0)
        thdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        thdr.grid_propagate(False)
        thdr.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(thdr, text="🎓  Training Mode",
                     font=CTkFont("Arial",14,"bold"), text_color=PRIMARY
                     ).grid(row=0, column=0, padx=16, pady=10, sticky="w")
        ctk.CTkButton(thdr, text="← Back to Main Screen",
            fg_color=SURF, hover_color=DANGER, text_color=ON_BG,
            command=self._close_training_workspace, height=30, width=200
            ).grid(row=0, column=2, padx=16, pady=7, sticky="e")

        # Camera pane
        cc = ctk.CTkFrame(self._train_frame, fg_color=SURF, corner_radius=12)
        cc.grid(row=1, column=0, sticky="nsew", padx=(16,8), pady=16)
        cc.grid_rowconfigure(1, weight=1)
        cc.grid_columnconfigure(0, weight=1)
        
        tctr = ctk.CTkFrame(cc, fg_color="transparent")
        tctr.grid(row=0, column=0, sticky="ew", padx=16, pady=(12,6))
        tctr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(tctr, text="Training Camera", font=CTkFont("Arial",16,"bold"),
                     text_color=ON_BG).grid(row=0, column=0, sticky="w")
        tctf = ctk.CTkFrame(tctr, fg_color="transparent")
        tctf.grid(row=0, column=1, sticky="e")
        ctk.CTkLabel(tctf, text="Camera:", text_color=ON_BG_MUTED).pack(side="left", padx=(0,6))
        self._train_cam_switch = ctk.CTkSwitch(tctf, text="", variable=ctk.BooleanVar(value=self._cam_active),
                                         onvalue=True, offvalue=False,
                                         progress_color=PRIMARY, command=self._toggle_camera, width=46)
        self._train_cam_switch.pack(side="left")

        self._train_video_label = ctk.CTkLabel(cc, text="", fg_color="black", corner_radius=8)
        self._train_video_label.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0,16))

        self._train_cam_off_label = ctk.CTkLabel(cc,
            text="📷\nCamera is OFF\nToggle the switch above to start",
            font=CTkFont("Arial",14), text_color=ON_BG_MUTED,
            fg_color="#001830", corner_radius=8)
        self._train_cam_off_label.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0,16))
        
        if not self._cam_active:
            self._train_cam_off_label.lift()
        else:
            self._train_cam_off_label.lower(self._train_video_label)

        # Controls pane
        ctrl = ctk.CTkScrollableFrame(self._train_frame, fg_color=SURF, corner_radius=12,
                                       scrollbar_button_color=SURF_VAR)
        ctrl.grid(row=1, column=1, sticky="nsew", padx=(8,16), pady=16)
        ctrl.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(ctrl, text="Sign Name (label):",
                     font=CTkFont("Arial",14,"bold"), text_color=ON_BG
                     ).pack(anchor="w", padx=16, pady=(16,4))
        self._train_name_entry = ctk.CTkEntry(ctrl, font=CTkFont("Arial",15),
            fg_color=INPUT_BG, text_color=ON_BG, border_width=1, border_color=PRIMARY,
            placeholder_text="e.g.  hello  /  thankyou")
        self._train_name_entry.pack(fill="x", padx=16, pady=(0,12))

        ctk.CTkLabel(ctrl,
            text="Tips:\n• Vary distance, angle & lighting\n• Move slightly between samples\n• Minimum 20 samples (200 recommended)",
            text_color=ON_BG_MUTED, font=CTkFont(size=12), justify="left"
            ).pack(anchor="w", padx=16, pady=(0,12))

        self._train_status_lbl = ctk.CTkLabel(ctrl,
            text="Enter a sign name and click Start Collection.",
            font=CTkFont("Arial",13), text_color=SUCCESS)
        self._train_status_lbl.pack(anchor="w", padx=16, pady=(0,6))

        self._train_progress = ctk.CTkProgressBar(ctrl, progress_color=PRIMARY, fg_color=BG_DARK)
        self._train_progress.pack(fill="x", padx=16, pady=(0,4))
        self._train_progress.set(0)

        self._train_count_lbl = ctk.CTkLabel(ctrl, text=f"0 / {self._train_target} samples",
                                              font=CTkFont("Arial",12), text_color=ON_BG_MUTED)
        self._train_count_lbl.pack(anchor="e", padx=16, pady=(0,10))

        # Target selector
        tgr = ctk.CTkFrame(ctrl, fg_color="transparent")
        tgr.pack(fill="x", padx=16, pady=(0,12))
        tgr.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(tgr, text="Samples target:", text_color=ON_BG_MUTED,
                     font=CTkFont(size=12)).grid(row=0, column=0, sticky="w")
        self._train_target_var = ctk.StringVar(value="200")
        ctk.CTkOptionMenu(tgr, variable=self._train_target_var,
            values=["100","150","200","300","400"],
            fg_color=INPUT_BG, button_color=INPUT_BG,
            button_hover_color=PRIMARY, text_color=ON_BG, width=110,
            command=self._update_train_target
            ).grid(row=0, column=1, sticky="e")

        b1 = ctk.CTkFrame(ctrl, fg_color="transparent")
        b1.pack(fill="x", padx=16, pady=(0,8))
        b1.grid_columnconfigure((0,1), weight=1)
        self._train_start_btn = ctk.CTkButton(b1, text="▶  Start Collection",
            fg_color=PRIMARY, text_color=BG_DARK, font=CTkFont(weight="bold"),
            command=self._start_collection)
        self._train_start_btn.grid(row=0, column=0, padx=(0,4), sticky="ew")
        self._train_pause_btn = ctk.CTkButton(b1, text="⏸  Pause",
            fg_color=WARN, text_color=BG_DARK, text_color_disabled=BG_DARK, state="disabled",
            command=self._pause_collection)
        self._train_pause_btn.grid(row=0, column=1, padx=(4,0), sticky="ew")

        b2 = ctk.CTkFrame(ctrl, fg_color="transparent")
        b2.pack(fill="x", padx=16, pady=(0,8))
        b2.grid_columnconfigure((0,1), weight=1)
        self._train_save_btn = ctk.CTkButton(b2, text="💾  Save & Train",
            fg_color=SUCCESS, text_color=BG_DARK, font=CTkFont(weight="bold"),
            command=self._save_and_train)
        self._train_save_btn.grid(row=0, column=0, padx=(0,4), sticky="ew")
        ctk.CTkButton(b2, text="🗑  Discard",
            fg_color=SURF_VAR, text_color=ON_BG, border_color=DANGER, border_width=1,
            command=self._discard_samples
            ).grid(row=0, column=1, padx=(4,0), sticky="ew")

        self._train_ui_loop()

    def _update_train_target(self, v):
        self._train_target = int(v)
        cnt = len(self._train_samples)
        self._train_count_lbl.configure(text=f"{cnt} / {self._train_target} samples")
        self._train_progress.set(cnt / max(1, self._train_target))

    def _train_ui_loop(self):
        if self._state != AppState.TRAINING: return

        with self._frame_lock:
            img     = self._latest_frame
            feats   = self._latest_features
            n_hands = self._latest_n_hands

        if img:
            try:
                lw = self._train_video_label.winfo_width()
                lh = self._train_video_label.winfo_height()
                if lw > 10 and lh > 10:
                    iw, ih  = img.size
                    ratio   = min(lw/iw, lh/ih)
                    nw, nh  = int(iw*ratio), int(ih*ratio)
                    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
                    photo   = ctk.CTkImage(light_image=resized, dark_image=resized, size=(nw, nh))
                    self._train_video_label.configure(image=photo)
                    self._train_video_label.image = photo
            except: pass

        if self._train_recording and n_hands > 0 and feats is not None:
            now = time.time()
            if now - self._train_last_capture >= 0.1:
                if len(self._train_samples) < self._train_target:
                    self._train_samples.append(feats.copy() if hasattr(feats,"copy") else list(feats))
                    self._train_last_capture = now
                    cnt = len(self._train_samples)
                    self._train_progress.set(cnt / self._train_target)
                    self._train_count_lbl.configure(text=f"{cnt} / {self._train_target} samples")
                    if cnt >= self._train_target:
                        self._train_recording = False
                        self._train_status_lbl.configure(
                            text="✅ Collection complete! Click Save & Train.", text_color=SUCCESS)
                        self._train_start_btn.configure(state="disabled")
                        self._train_pause_btn.configure(state="disabled")

        if self._train_recording:
            if n_hands == 0:
                self._train_status_lbl.configure(text="⚠ No hands detected — show your hand.", text_color=WARN)
            else:
                self._train_status_lbl.configure(
                    text=f"⏺ Recording… ({len(self._train_samples)}/{self._train_target})", text_color=DANGER)

        self.after(33, self._train_ui_loop)

    def _start_collection(self):
        if not self._cam_active:
            self._train_status_lbl.configure(
                text="⚠ Turn the camera ON before starting.", text_color=DANGER); return
        name = self._train_name_entry.get().strip().lower()
        if not name or not name.replace("_","").isalpha():
            self._train_status_lbl.configure(
                text="⚠ Enter a valid sign name (letters only).", text_color=DANGER); return
        self._train_samples      = []
        self._train_recording    = True
        self._train_last_capture = 0.0
        self._train_target       = int(self._train_target_var.get())
        self._train_progress.set(0)
        self._train_count_lbl.configure(text=f"0 / {self._train_target} samples")
        self._train_start_btn.configure(state="disabled")
        self._train_pause_btn.configure(state="normal")
        self._train_status_lbl.configure(text="⏺ Recording…", text_color=DANGER)

    def _pause_collection(self):
        self._train_recording = False
        self._train_start_btn.configure(state="normal", text="▶  Resume Collection")
        self._train_pause_btn.configure(state="disabled")
        self._train_status_lbl.configure(text="⏸ Paused.", text_color=WARN)

    def _discard_samples(self):
        self._train_samples   = []
        self._train_recording = False
        self._train_progress.set(0)
        self._train_count_lbl.configure(text=f"0 / {self._train_target} samples")
        self._train_start_btn.configure(state="normal", text="▶  Start Collection")
        self._train_pause_btn.configure(state="disabled")
        self._train_status_lbl.configure(text="Samples discarded. Ready to start again.", text_color=ON_BG_MUTED)

    def _save_and_train(self):
        name = self._train_name_entry.get().strip().lower()
        if not name or not name.replace("_","").isalpha():
            self._train_status_lbl.configure(text="⚠ Enter a valid sign name.", text_color=DANGER); return
        if len(self._train_samples) < 20:
            self._train_status_lbl.configure(
                text=f"⚠ Need at least 20 samples (got {len(self._train_samples)}).", text_color=DANGER); return

        if self.classifier and hasattr(self.classifier, 'label_encoder'):
            existing = [c.lower() for c in self.classifier.label_encoder.classes_]
            if name in existing:
                self._train_status_lbl.configure(
                    text="⚠ This sign already exists. Please choose a new sign label.", text_color=WARN)
                return

        self._train_status_lbl.configure(text="⏳ Saving and training… please wait.", text_color=WARN)
        self._train_save_btn.configure(state="disabled")
        self._train_start_btn.configure(state="disabled")
        snap = list(self._train_samples)

        def _worker():
            try:
                import pandas as pd, shutil
                from src.ml.train_models import train_and_evaluate
                csv_p = os.path.join(os.path.dirname(__file__), "data", "processed", "landmarks_custom.csv")
                mdl_p = os.path.join(os.path.dirname(__file__), "models", "saved_models", "sign_classifier.pkl")
                for p in (csv_p, mdl_p):
                    if os.path.exists(p): shutil.copy2(p, p+".bak")

                cols   = [f"f{i}" for i in range(126)] + ["label"]
                rows   = [list(f)+[name] for f in snap]
                df_new = pd.DataFrame(rows, columns=cols)
                if os.path.exists(csv_p):
                    df_combined = pd.concat([pd.read_csv(csv_p), df_new], ignore_index=True)
                else:
                    df_combined = df_new
                os.makedirs(os.path.dirname(csv_p), exist_ok=True)
                df_combined.to_csv(csv_p, index=False)
                print(f"[Train] Saved {len(df_new)} samples for '{name}'. Total: {len(df_combined)}")

                train_and_evaluate(csv_p)

                import joblib
                payload = joblib.load(mdl_p)
                assert "model" in payload and "label_encoder" in payload
                cls = payload.get("classes", [])
                assert name in cls, f"'{name}' not in trained classes {cls}"

                self.after(0, self._reload_model_after_training)
            except Exception as e:
                print(f"[Train] Error: {e}")
                try:
                    for p in (mdl_p, csv_p):
                        if os.path.exists(p+".bak"): shutil.copy2(p+".bak", p)
                except: pass
                self.after(0, lambda: self._train_status_lbl.configure(
                    text=f"❌ Failed: {e}\nOld model preserved.", text_color=DANGER))
                self.after(0, lambda: self._train_save_btn.configure(state="normal"))
                self.after(0, lambda: self._train_start_btn.configure(state="normal"))

        threading.Thread(target=_worker, daemon=True, name="TrainWorker").start()

    def _reload_model_after_training(self):
        try:
            self.classifier = SignClassifier(MODEL_PATH, smooth_window=5, confidence_threshold=0.65)
            self._train_status_lbl.configure(text="✅ Training complete! New sign is now active.", text_color=SUCCESS)
        except Exception as e:
            self._train_status_lbl.configure(text=f"❌ Reload failed: {e}", text_color=DANGER)
        finally:
            self._train_save_btn.configure(state="normal")
            self._train_start_btn.configure(state="normal")

    def _close_training_workspace(self):
        self._train_recording = False
        self._state = AppState.MAIN
        self._dc_sign = None  # Force UI update
        self._auto_speak.set(self._auto_speak_saved)
        # Reset stale recognition
        self._current_sign = "..."; self._current_conf = 0.0; self._hold_ratio = 0.0
        self._candidate = None; self._hold_count = 0; self._last_added = None
        if self.classifier: self.classifier.reset_history()
        try: self._train_frame.destroy()
        except: pass
        self._left_col.grid()
        self._right_col.grid()

    # ─── Cleanup ──────────────────────────────────────────────────────── #
    def on_closing(self):
        print("[JARVIS] Shutting down…")
        self._state      = AppState.SHUTTING_DOWN
        self._cam_active = False
        if self.cam:     self.cam.release()
        if self.detector: self.detector.close()
        if self.sapi_speech: self.sapi_speech.stop()
        pygame.mixer.music.stop()
        self.destroy()
        print("[JARVIS] Goodbye.")


# ── Entry point ────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    app = JarvisApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
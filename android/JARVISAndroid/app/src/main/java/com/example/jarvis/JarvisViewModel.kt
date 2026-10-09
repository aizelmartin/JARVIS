package com.example.jarvis

import android.app.Application
import android.util.Base64
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.socket.client.IO
import io.socket.client.Socket
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import org.json.JSONObject
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean

enum class TranslationSource {
    SIGN_SENTENCE, SPEECH_TEXT
}

/**
 * Central ViewModel for JARVIS Android.
 *
 * Owns:
 *  - Socket.IO connection to the Python backend
 *  - Frame transmission pipeline (CameraX → JPEG → Base64 → Socket.IO)
 *  - All backend-derived state (sign, confidence, hold, sentence, translation)
 *  - Connection state machine
 *
 * Does NOT own UI or camera lifecycle.
 */
class JarvisViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "JarvisViewModel"

        // Frame throttle: one frame per this many milliseconds → ~12 FPS
        private const val FRAME_INTERVAL_MS = 85L

        // JPEG compression quality for transmitted frames (0–100)
        private const val JPEG_QUALITY = 60
    }

    // ──────────────────────────────────────────────
    // Persistent settings
    // ──────────────────────────────────────────────

    private val prefs = application.getSharedPreferences("jarvis_prefs", 0)

    private val _backendUrl = MutableStateFlow(
        prefs.getString("backend_url", "http://192.168.1.100:5000") ?: "http://192.168.1.100:5000"
    )
    val backendUrl: StateFlow<String> = _backendUrl.asStateFlow()

    fun setBackendUrl(url: String) {
        _backendUrl.value = url
        prefs.edit().putString("backend_url", url).apply()
    }

    // ──────────────────────────────────────────────
    // Connection state
    // ──────────────────────────────────────────────

    enum class ConnectionState {
        DISCONNECTED, CONNECTING, CONNECTED
    }

    private val _connectionState = MutableStateFlow(ConnectionState.DISCONNECTED)
    val connectionState: StateFlow<ConnectionState> = _connectionState.asStateFlow()

    private val _connectionError = MutableStateFlow<String?>(null)
    val connectionError: StateFlow<String?> = _connectionError.asStateFlow()

    // ──────────────────────────────────────────────
    // Camera state
    // ──────────────────────────────────────────────

    private val _cameraActive = MutableStateFlow(true)
    val cameraActive: StateFlow<Boolean> = _cameraActive.asStateFlow()

    fun toggleCamera() {
        _cameraActive.value = !_cameraActive.value
        if (!_cameraActive.value) {
            stopSending()
            resetMlState()
            _currentSign.value = "Camera off"
        } else if (_connectionState.value == ConnectionState.CONNECTED) {
            startSending()
        }
    }

    // ──────────────────────────────────────────────
    // ML / backend output state
    // ──────────────────────────────────────────────

    private val _currentSign    = MutableStateFlow("No sign detected")
    val currentSign: StateFlow<String> = _currentSign.asStateFlow()

    private val _confidence     = MutableStateFlow(0f)
    val confidence: StateFlow<Float> = _confidence.asStateFlow()

    private val _holdRatio      = MutableStateFlow(0f)
    val holdRatio: StateFlow<Float> = _holdRatio.asStateFlow()

    private val _sentence       = MutableStateFlow("...")
    val sentence: StateFlow<String> = _sentence.asStateFlow()

    private val _translatedText = MutableStateFlow("...")
    val translatedText: StateFlow<String> = _translatedText.asStateFlow()

    fun setTranslatedText(text: String) {
        _translatedText.value = text
    }

    private val _autoSpeakEnabled = MutableStateFlow(false)
    val autoSpeakEnabled: StateFlow<Boolean> = _autoSpeakEnabled.asStateFlow()

    fun toggleAutoSpeak() {
        _autoSpeakEnabled.value = !_autoSpeakEnabled.value
    }

    private val _speakEvent = MutableSharedFlow<String>(extraBufferCapacity = 1)
    val speakEvent: SharedFlow<String> = _speakEvent.asSharedFlow()

    fun manualSpeak(sign: String) {
        _speakEvent.tryEmit(sign)
    }

    private val _translationSource = MutableStateFlow(TranslationSource.SIGN_SENTENCE)
    val translationSource: StateFlow<TranslationSource> = _translationSource.asStateFlow()

    fun setTranslationSource(source: TranslationSource) {
        _translationSource.value = source
    }

    private val _speechInputText = MutableStateFlow("...")
    val speechInputText: StateFlow<String> = _speechInputText.asStateFlow()

    fun setSpeechInputText(text: String) {
        _speechInputText.value = text
    }

    // ──────────────────────────────────────────────
    // Socket.IO
    // ──────────────────────────────────────────────

    private var socket: Socket? = null

    fun connect() {
        val url = _backendUrl.value.trim()

        // If we already have an active socket, tear it down cleanly first.
        // Off() removes all listeners so we don't get duplicate callbacks after reconnect.
        val existing = socket
        if (existing != null) {
            existing.off()
            existing.disconnect()
            socket = null
        }

        _connectionState.value = ConnectionState.CONNECTING
        _connectionError.value = null

        viewModelScope.launch(Dispatchers.IO) {
            try {
                // CRITICAL: "polling" MUST come before "websocket".
                // Engine.IO requires an initial HTTP long-poll handshake to exchange
                // session ID and capabilities.  Only after that does it upgrade to WS.
                // Listing "websocket" first skips the handshake and causes:
                //   EngineIOException: websocket error
                val opts = IO.Options.builder()
                    .setTransports(arrayOf("polling", "websocket"))
                    .setReconnection(true)
                    .setReconnectionAttempts(Int.MAX_VALUE)
                    .setReconnectionDelay(2000)
                    .setReconnectionDelayMax(10000)
                    .setTimeout(20000)
                    .setForceNew(true)
                    .build()

                Log.i(TAG, "Connecting to $url (polling → websocket)")

                socket = IO.socket(url, opts).also { s ->
                    // ── Successful connection (initial or after reconnect) ───────── //
                    s.on(Socket.EVENT_CONNECT) {
                        Log.i(TAG, "Socket.IO connected to $url (transport: polling→ws upgrade)")
                        _connectionState.value = ConnectionState.CONNECTED
                        _connectionError.value = null
                        // Resume frame sending if the camera was on before disconnect
                        if (_cameraActive.value) startSending()
                    }

                    // ── Clean disconnect (server closed or network gone) ──────── //
                    s.on(Socket.EVENT_DISCONNECT) { args ->
                        val reason = args.firstOrNull()?.toString() ?: "unknown"
                        Log.w(TAG, "Socket.IO disconnected: $reason")
                        _connectionState.value = ConnectionState.DISCONNECTED
                        stopSending()
                    }

                    // ── Connection attempt failed (will auto-retry) ───────────── //
                    s.on(Socket.EVENT_CONNECT_ERROR) { args ->
                        val msg = args.firstOrNull()?.toString() ?: "Connection error"
                        Log.e(TAG, "Socket.IO connect error: $msg")
                        // Stay in CONNECTING — the client will auto-retry.
                        // Only update the error text so the user can see what's happening.
                        _connectionState.value = ConnectionState.CONNECTING
                        _connectionError.value = "Retrying… ($msg)"
                    }

                    // ── Reconnect attempt (already handled by lib, just log) ───── //
                    s.on("reconnect_attempt") { args ->
                        val attempt = args.firstOrNull()?.toString() ?: "?"
                        Log.i(TAG, "Socket.IO reconnect attempt #$attempt")
                        _connectionState.value = ConnectionState.CONNECTING
                    }

                    // ── Reconnected successfully ─────────────────────────────── //
                    s.on("reconnect") { args ->
                        val attempt = args.firstOrNull()?.toString() ?: "?"
                        Log.i(TAG, "Socket.IO reconnected after $attempt attempt(s)")
                        _connectionState.value = ConnectionState.CONNECTED
                        _connectionError.value = null
                        if (_cameraActive.value) startSending()
                    }

                    // ── ML / app-level events ─────────────────────────────────── //
                    s.on("processed_frame") { args ->
                        handleProcessedFrame(args.firstOrNull())
                    }
                    s.on("sentence_updated") { args ->
                        val obj = args.firstOrNull() as? JSONObject ?: return@on
                        _sentence.value = obj.optString("sentence", "...")
                    }
                    s.on("translation_result") { args ->
                        val obj = args.firstOrNull() as? JSONObject ?: return@on
                        _translatedText.value = obj.optString("translated", "...")
                    }
                }
                socket?.connect()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to create socket: ${e.message}", e)
                _connectionState.value = ConnectionState.DISCONNECTED
                _connectionError.value = e.message
            }
        }
    }

    fun disconnect() {
        socket?.off()
        socket?.disconnect()
        socket = null
        _connectionState.value = ConnectionState.DISCONNECTED
        resetMlState()
    }

    private fun resetMlState() {
        _currentSign.value = if (_cameraActive.value) "No sign detected" else "Camera off"
        _confidence.value  = 0f
        _holdRatio.value   = 0f
    }

    // ──────────────────────────────────────────────
    // Frame transmission
    // ──────────────────────────────────────────────

    private val sendingActive = AtomicBoolean(false)
    private var frameJob: Job? = null
    // Holds the latest raw JPEG bytes from CameraX; replaced at ~camera FPS
    @Volatile private var pendingFrameBytes: ByteArray? = null

    /**
     * Called by CameraX ImageAnalysis on a background thread.
     * Stores the raw JPEG bytes. The sender coroutine picks them up at
     * the throttled rate, so we never queue more than one frame at a time.
     */
    fun onCameraFrame(jpegBytes: ByteArray) {
        pendingFrameBytes = jpegBytes
    }

    /** Start the background coroutine that drains [pendingFrameBytes] at the throttled FPS. */
    fun startSending() {
        if (sendingActive.getAndSet(true)) return
        frameJob = viewModelScope.launch(Dispatchers.IO) {
            while (sendingActive.get()) {
                val bytes = pendingFrameBytes
                if (bytes != null && _connectionState.value == ConnectionState.CONNECTED && _cameraActive.value) {
                    pendingFrameBytes = null
                    sendFrame(bytes)
                }
                delay(FRAME_INTERVAL_MS)
            }
        }
    }

    /** Stop sending frames (call when camera is paused/stopped). */
    fun stopSending() {
        sendingActive.set(false)
        frameJob?.cancel()
        pendingFrameBytes = null
    }

    private fun sendFrame(jpegBytes: ByteArray) {
        val s = socket ?: return
        if (!s.connected()) return

        // Backend expects: { "image": "data:image/jpeg;base64,<encoded>" }
        val b64 = Base64.encodeToString(jpegBytes, Base64.NO_WRAP)
        val payload = JSONObject()
        payload.put("image", "data:image/jpeg;base64,$b64")
        s.emit("process_frame", payload)
    }

    // ──────────────────────────────────────────────
    // Processed frame handler
    // ──────────────────────────────────────────────

    private fun handleProcessedFrame(raw: Any?) {
        val obj = raw as? JSONObject ?: return
        if (obj.has("error")) {
            Log.e(TAG, "Backend error: ${obj.optString("error")}")
            return
        }

        val sign       = obj.optString("sign", "...")
        val conf       = obj.optDouble("confidence", 0.0).toFloat()
        val holdRatio  = obj.optDouble("hold_ratio", 0.0).toFloat()
        val sentence   = obj.optString("sentence", "...")

        _currentSign.value = if (sign == "..." || sign.isBlank()) "No sign detected" else sign
        _confidence.value  = if (sign == "..." || sign.isBlank()) 0f else conf
        _holdRatio.value   = holdRatio
        
        val oldSentence = _sentence.value
        val newSentence = if (sentence.isBlank() || sentence == "...") "..." else sentence
        _sentence.value = newSentence

        if (_autoSpeakEnabled.value && newSentence != "..." && oldSentence != newSentence) {
            val oldWords = if (oldSentence == "...") emptyList() else oldSentence.split(" ")
            val newWords = newSentence.split(" ")
            if (newWords.size > oldWords.size) {
                _speakEvent.tryEmit(newWords.last())
            }
        }
    }

    // ──────────────────────────────────────────────
    // Sentence controls
    // ──────────────────────────────────────────────

    fun deleteWord() {
        if (_connectionState.value != ConnectionState.CONNECTED) return
        val payload = JSONObject().put("action", "delete")
        socket?.emit("command", payload)
    }

    fun clearSentence() {
        if (_connectionState.value != ConnectionState.CONNECTED) return
        val payload = JSONObject().put("action", "clear")
        socket?.emit("command", payload)
        _sentence.value    = "..."
        _translatedText.value = "..."
    }

    // ──────────────────────────────────────────────
    // Translation
    // ──────────────────────────────────────────────

    private val _selectedLanguage = MutableStateFlow("ml")
    val selectedLanguage: StateFlow<String> = _selectedLanguage.asStateFlow()

    fun setLanguage(langCode: String) {
        _selectedLanguage.value = langCode
    }

    fun requestTranslation() {
        if (_connectionState.value != ConnectionState.CONNECTED) return
        val textToTranslate = if (_translationSource.value == TranslationSource.SPEECH_TEXT) _speechInputText.value else _sentence.value
        val payload = JSONObject().put("lang", _selectedLanguage.value)
        if (textToTranslate != "..." && textToTranslate.isNotBlank()) {
            payload.put("text", textToTranslate)
        }
        socket?.emit("translate", payload)
    }

    // ──────────────────────────────────────────────
    // Cleanup
    // ──────────────────────────────────────────────

    override fun onCleared() {
        super.onCleared()
        stopSending()
        disconnect()
    }
}

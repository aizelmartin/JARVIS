const socket = io();

// ── DOM elements ───────────────────────────────────────────────────────────── //
const videoInput   = document.getElementById('video-input');    // hidden video element
const canvasMain   = document.getElementById('canvas-main');    // main display canvas
const canvasHidden = document.getElementById('canvas-hidden'); // off-screen capture canvas
const ctxMain      = canvasMain.getContext('2d');
const ctxHidden    = canvasHidden.getContext('2d');

const btnStart      = document.getElementById('btn-start');
const btnStop       = document.getElementById('btn-stop');
const cameraStatus  = document.getElementById('camera-status');
const socketStatus  = document.getElementById('socket-status');

const currentSignEl = document.getElementById('current-sign');
const confValEl     = document.getElementById('conf-val');
const confBarEl     = document.getElementById('conf-bar');
const holdBarEl     = document.getElementById('hold-bar');

const sentenceDisplay   = document.getElementById('sentence-display');
const btnDelete         = document.getElementById('btn-delete');
const btnClear          = document.getElementById('btn-clear');
const langSelect        = document.getElementById('lang-select');
const translatedDisplay = document.getElementById('translated-display');
const btnSpeak          = document.getElementById('btn-speak');
const btnMic            = document.getElementById('btn-mic');
const sttStatus         = document.getElementById('stt-status');
const sttDisplay        = document.getElementById('stt-display');

// ── State ─────────────────────────────────────────────────────────────────── //
let cameraStream    = null;
let rafId           = null;       // requestAnimationFrame handle for live preview
let sendInterval    = null;       // setInterval handle for backend sends
let latestHands     = [];         // most recent landmark JSON from backend
const SEND_FPS      = 10;         // frames sent to backend (recognition quality)
const DISPLAY_W     = 640;
const DISPLAY_H     = 480;

// ── MediaPipe hand connection pairs (for drawing skeleton) ─────────────────── //
const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],           // thumb
    [0,5],[5,6],[6,7],[7,8],           // index
    [5,9],[9,10],[10,11],[11,12],      // middle
    [9,13],[13,14],[14,15],[15,16],    // ring
    [13,17],[0,17],[17,18],[18,19],[19,20] // pinky + palm
];

// ── Socket status ─────────────────────────────────────────────────────────── //
socket.on('connect', () => {
    socketStatus.textContent = 'Connected';
    socketStatus.style.color = 'var(--accent)';
});
socket.on('disconnect', () => {
    socketStatus.textContent = 'Disconnected';
    socketStatus.style.color = 'var(--danger)';
});

// ── Live preview loop (runs at ~60fps via rAF) ────────────────────────────── //
let previewFrameCount = 0;
function previewLoop() {
    if (!cameraStream) return;

    if (videoInput.readyState >= videoInput.HAVE_CURRENT_DATA) {
        // 1. Draw the raw camera frame
        ctxMain.drawImage(videoInput, 0, 0, DISPLAY_W, DISPLAY_H);

        // 2. Draw latest landmark skeleton on top (if any)
        if (latestHands.length > 0) {
            drawLandmarks(ctxMain, latestHands, DISPLAY_W, DISPLAY_H);
        }

        // 3. Small debug HUD (top-right corner)
        previewFrameCount++;
        ctxMain.font = 'bold 12px monospace';
        ctxMain.fillStyle = latestHands.length > 0 ? '#00ff9d' : '#ff4757';
        const hudText = `hands: ${latestHands.length}  frame: ${previewFrameCount}`;
        ctxMain.fillText(hudText, DISPLAY_W - 200, 20);
    }
    rafId = requestAnimationFrame(previewLoop);
}

// ── Draw MediaPipe landmarks on a canvas context ──────────────────────────── //
function drawLandmarks(ctx, hands, w, h) {
    hands.forEach(hand => {
        const pts = hand.landmarks;

        // Draw connections (skeleton lines)
        ctx.strokeStyle = hand.label === 'Right' ? '#00e5ff' : '#ff6b6b';
        ctx.lineWidth   = 2;
        HAND_CONNECTIONS.forEach(([a, b]) => {
            ctx.beginPath();
            // mirror x for same reason as dots below
            ctx.moveTo((1.0 - pts[a][0]) * w, pts[a][1] * h);
            ctx.lineTo((1.0 - pts[b][0]) * w, pts[b][1] * h);
            ctx.stroke();
        });

        // Draw landmark dots
        pts.forEach((pt, idx) => {
            // x must be mirrored: backend flips frame before MediaPipe,
            // but display canvas shows the unflipped local feed.
            const x = (1.0 - pt[0]) * w;
            const y = pt[1] * h;
            // Larger dot for wrist (0) and fingertips (4,8,12,16,20)
            const isKeyPoint = [0, 4, 8, 12, 16, 20].includes(idx);
            ctx.beginPath();
            ctx.arc(x, y, isKeyPoint ? 5 : 3, 0, Math.PI * 2);
            ctx.fillStyle = isKeyPoint ? '#ffffff' : '#00e5ff';
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.lineWidth   = 1;
            ctx.stroke();
        });
    });
}

// ── Start Camera ──────────────────────────────────────────────────────────── //
btnStart.addEventListener('click', async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Camera API not available.\n\nUse http://localhost:5000 (not 127.0.0.1).');
        return;
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
        });

        videoInput.srcObject = cameraStream;

        await new Promise(resolve => {
            videoInput.onloadedmetadata = () => { videoInput.play(); resolve(); };
        });

        btnStart.disabled = true;
        btnStop.disabled  = false;
        cameraStatus.textContent = 'Camera Active';
        cameraStatus.style.color = 'var(--accent)';

        // Start smooth live preview
        previewLoop();

        // Send frames to backend at reduced FPS for recognition
        sendInterval = setInterval(() => {
            if (videoInput.readyState < videoInput.HAVE_CURRENT_DATA) return;
            ctxHidden.drawImage(videoInput, 0, 0, canvasHidden.width, canvasHidden.height);
            const imageData = canvasHidden.toDataURL('image/jpeg', 0.6); // lower quality = faster
            socket.emit('process_frame', { image: imageData });
        }, 1000 / SEND_FPS);

    } catch (err) {
        console.error('Camera error:', err.name, err.message);
        let msg = `Camera Error: ${err.name}\n\n`;
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            msg += 'Camera permission denied. Allow camera access and try again.';
        } else if (err.name === 'NotFoundError') {
            msg += 'No camera found.';
        } else if (err.name === 'NotReadableError') {
            msg += 'Camera already in use by another app.';
        } else {
            msg += `Try http://localhost:5000\n\nDetails: ${err.message}`;
        }
        alert(msg);
    }
});

// ── Stop Camera ───────────────────────────────────────────────────────────── //
btnStop.addEventListener('click', () => {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
        videoInput.srcObject = null;
    }
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    if (sendInterval) { clearInterval(sendInterval); sendInterval = null; }

    latestHands = [];
    ctxMain.clearRect(0, 0, DISPLAY_W, DISPLAY_H);

    btnStart.disabled = false;
    btnStop.disabled  = true;
    cameraStatus.textContent = 'Camera Offline';
    cameraStatus.style.color = 'var(--danger)';

    currentSignEl.textContent = '...';
    currentSignEl.classList.remove('highlight');
    confValEl.textContent = '0';
    confBarEl.style.width = '0%';
    holdBarEl.style.width = '0%';
});

// ── Receive landmark data + telemetry from backend ────────────────────────── //
socket.on('processed_frame', (data) => {
    if (data.error) { console.warn('Backend error:', data.error); return; }

    // Store latest hand landmarks – previewLoop draws them on next rAF tick
    latestHands = data.hands || [];
    if (latestHands.length > 0) {
        console.debug(`[JARVIS] hands=${latestHands.length}, sign=${data.sign} (${Math.round(data.confidence*100)}%)`);
    }

    const sign = data.sign;
    const conf = data.confidence;
    const hold = data.hold_ratio;

    if (sign && sign !== '...') {
        currentSignEl.textContent = sign.toUpperCase();
        currentSignEl.classList.add('highlight');
    } else {
        currentSignEl.textContent = '...';
        currentSignEl.classList.remove('highlight');
    }

    confValEl.textContent    = Math.round(conf * 100);
    confBarEl.style.width    = `${Math.round(conf * 100)}%`;
    holdBarEl.style.width    = `${Math.round(hold * 100)}%`;

    const sentence = data.sentence;
    sentenceDisplay.textContent = (sentence && sentence !== '...') ? sentence : 'Add words by signing...';

    if (sentence && sentence !== '...' && sentence !== window.lastSentence) {
        window.lastSentence = sentence;
        socket.emit('translate', { lang: langSelect.value });
    }
});

// ── Controls (Delete / Clear) ─────────────────────────────────────────────── //
btnDelete.addEventListener('click', () => socket.emit('command', { action: 'delete' }));
btnClear.addEventListener('click', () => {
    socket.emit('command', { action: 'clear' });
    window.lastSentence = '...';
    translatedDisplay.textContent = '...';
});

socket.on('sentence_updated', (data) => {
    const sentence = data.sentence;
    sentenceDisplay.textContent = (sentence && sentence !== '...') ? sentence : 'Add words by signing...';
    window.lastSentence = sentence;
    if (sentence && sentence !== '...') {
        socket.emit('translate', { lang: langSelect.value });
    } else {
        translatedDisplay.textContent = '...';
    }
});

// ── Translation ───────────────────────────────────────────────────────────── //
langSelect.addEventListener('change', () => {
    if (window.lastSentence && window.lastSentence !== '...') {
        socket.emit('translate', { lang: langSelect.value });
    }
});
socket.on('translation_result', (data) => {
    translatedDisplay.textContent = data.translated;
});

// ── Speak Translation ─────────────────────────────────────────────────────── //
btnSpeak.addEventListener('click', () => {
    const text = translatedDisplay.textContent;
    if (text && text !== '...') socket.emit('speak', { text });
});

// ── STT ───────────────────────────────────────────────────────────────────── //
btnMic.addEventListener('click', () => {
    btnMic.disabled = true;
    sttStatus.textContent = 'Requesting Microphone...';
    socket.emit('start_stt', {});
});
socket.on('stt_status', (data) => { sttStatus.textContent = data.status; });
socket.on('stt_result', (data) => {
    btnMic.disabled = false;
    sttStatus.textContent = 'Ready';
    if (data.text) sttDisplay.textContent = data.text;
    else sttStatus.textContent = data.status || 'Could not understand audio.';
});

package com.example.jarvis

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.Bundle as OsBundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import java.io.ByteArrayOutputStream
import java.util.Locale
import java.util.concurrent.Executors

private const val TAG = "JarvisMain"

class MainActivity : ComponentActivity() {

    private var tts: TextToSpeech? = null
    // Must be a Compose State so that recomposition fires when TTS initialises
    // asynchronously (the callback runs *after* setContent). Plain var = buttons
    // check the initial `false` forever and silently do nothing when tapped.
    private var ttsReady by mutableStateOf(false)

    override fun onCreate(savedInstanceState: OsBundle?) {
        super.onCreate(savedInstanceState)

        tts = TextToSpeech(this) { status ->
            ttsReady = (status == TextToSpeech.SUCCESS)
            if (ttsReady) {
                tts?.language = Locale.getDefault()
            }
        }

        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    primary         = Color(0xFF29B6F6),
                    secondary       = Color(0xFF00E5FF),
                    background      = Color(0xFF002B5E),
                    surface         = Color(0xFF003F87),
                    surfaceVariant  = Color(0xFF0056B3),
                    onBackground    = Color.White,
                    onSurface       = Color.White
                )
            ) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color    = MaterialTheme.colorScheme.background
                ) {
                    val vm: JarvisViewModel = viewModel()
                    JarvisDashboard(
                        vm      = vm,
                        tts     = tts,
                        ttsReady = ttsReady
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tts?.stop()
        tts?.shutdown()
    }
}

// ─── Language map ─────────────────────────────────────────────────────────────

private val LANGUAGES = listOf(
    "English"  to "en",
    "Malayalam" to "ml",
    "Hindi"    to "hi",
    "Tamil"    to "ta",
    "Kannada"  to "kn",
    "Telugu"   to "te"
)

// ─── Root composable ──────────────────────────────────────────────────────────

@Composable
fun JarvisDashboard(
    vm:       JarvisViewModel,
    tts:      TextToSpeech?,
    ttsReady: Boolean
) {
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        HeaderSection(vm)
        val context = LocalContext.current
        // Key on `ttsReady` so the collect coroutine restarts the instant TTS
        // becomes ready. LaunchedEffect(Unit) would capture ttsReady=false
        // forever because the coroutine is never restarted on recomposition.
        LaunchedEffect(ttsReady) {
            vm.speakEvent.collect { text ->
                if (ttsReady && text.isNotBlank()) {
                    tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
                }
            }
        }

        SettingsSection(vm)
        CameraSection(vm)
        RecognitionSection(vm)
        SentenceSection(vm, tts, ttsReady)
        TranslationSection(vm, tts, ttsReady)
        SpeechInputSection(vm)
        ManageSignsSection()
    }
}

// ─── Header ───────────────────────────────────────────────────────────────────

@Composable
fun HeaderSection(vm: JarvisViewModel) {
    val connState by vm.connectionState.collectAsState()

    val (dotColor, statusText) = when (connState) {
        JarvisViewModel.ConnectionState.CONNECTED    ->
            Color.Green to "Backend Connected"
        JarvisViewModel.ConnectionState.CONNECTING  ->
            Color.Yellow to "Connecting..."
        JarvisViewModel.ConnectionState.DISCONNECTED ->
            Color.Red to "Backend Disconnected"
    }

    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text          = "JARVIS",
            fontSize      = 32.sp,
            fontWeight    = FontWeight.Bold,
            color         = MaterialTheme.colorScheme.primary,
            letterSpacing = 2.sp
        )
        Text(
            text     = "Universal Sign Language Translator",
            fontSize = 14.sp,
            color    = Color.Gray
        )
        Spacer(modifier = Modifier.height(8.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(10.dp)
                    .clip(RoundedCornerShape(50))
                    .background(dotColor)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(statusText, color = dotColor, fontSize = 14.sp)
        }
    }
}

// ─── Settings / Backend URL ───────────────────────────────────────────────────

@Composable
fun SettingsSection(vm: JarvisViewModel) {
    val backendUrl    by vm.backendUrl.collectAsState()
    val connState     by vm.connectionState.collectAsState()
    val connError     by vm.connectionError.collectAsState()
    var urlEdit       by remember { mutableStateOf(backendUrl) }
    var expanded      by remember { mutableStateOf(false) }

    Card(
        shape  = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier          = Modifier.fillMaxWidth()
            ) {
                Icon(Icons.Default.Settings, contentDescription = null, tint = Color.Gray)
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    "Backend Configuration",
                    fontWeight = FontWeight.SemiBold,
                    fontSize   = 16.sp,
                    modifier   = Modifier.weight(1f)
                )
                TextButton(onClick = { expanded = !expanded }) {
                    Text(if (expanded) "Hide" else "Show")
                }
            }

            if (expanded) {
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedTextField(
                    value         = urlEdit,
                    onValueChange = { urlEdit = it },
                    label         = { Text("Backend URL") },
                    placeholder   = { Text("http://192.168.1.X:5000") },
                    singleLine    = true,
                    modifier      = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    modifier              = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Button(
                        onClick  = {
                            vm.setBackendUrl(urlEdit.trim())
                            vm.connect()
                        },
                        modifier = Modifier.weight(1f),
                        enabled  = connState != JarvisViewModel.ConnectionState.CONNECTING
                    ) {
                        Text(
                            when (connState) {
                                JarvisViewModel.ConnectionState.CONNECTING  -> "Connecting..."
                                JarvisViewModel.ConnectionState.CONNECTED   -> "Reconnect"
                                else                                         -> "Connect"
                            }
                        )
                    }
                    if (connState == JarvisViewModel.ConnectionState.CONNECTED) {
                        OutlinedButton(
                            onClick  = { vm.disconnect() },
                            modifier = Modifier.weight(1f),
                            colors   = ButtonDefaults.outlinedButtonColors(
                                contentColor = Color.Red
                            )
                        ) {
                            Text("Disconnect")
                        }
                    }
                }
                connError?.let {
                    Spacer(modifier = Modifier.height(4.dp))
                    Text("Error: $it", color = Color.Red, fontSize = 12.sp)
                }
            }
        }
    }
}

// ─── Camera ───────────────────────────────────────────────────────────────────

@Composable
fun CameraSection(vm: JarvisViewModel) {
    val context        = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val connState      by vm.connectionState.collectAsState()
    val cameraActive   by vm.cameraActive.collectAsState()

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED
        )
    }

    val permLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { hasCameraPermission = it }
    )

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) permLauncher.launch(Manifest.permission.CAMERA)
    }

    // Start/stop frame sending based on connection state
    LaunchedEffect(connState, cameraActive) {
        if (connState == JarvisViewModel.ConnectionState.CONNECTED && cameraActive) {
            vm.startSending()
        } else {
            vm.stopSending()
        }
    }

    Card(
        shape  = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Live Camera", fontWeight = FontWeight.SemiBold, fontSize = 18.sp)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(if (cameraActive) "ON" else "OFF", fontSize = 14.sp)
                    Spacer(modifier = Modifier.width(8.dp))
                    Switch(checked = cameraActive, onCheckedChange = { vm.toggleCamera() })
                }
            }
            Spacer(modifier = Modifier.height(12.dp))

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(280.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(Color.Black),
                contentAlignment = Alignment.Center
            ) {
                if (hasCameraPermission) {
                    if (cameraActive) {
                        val analysisExecutor = remember { Executors.newSingleThreadExecutor() }

                        DisposableEffect(Unit) {
                            onDispose {
                                val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
                                cameraProviderFuture.addListener({
                                    val provider = cameraProviderFuture.get()
                                    provider.unbindAll()
                                }, ContextCompat.getMainExecutor(context))
                            }
                        }

                        AndroidView(
                            factory = { ctx ->
                                val previewView = PreviewView(ctx).apply {
                                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                                    scaleType          = PreviewView.ScaleType.FILL_CENTER
                                }
                                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

                                cameraProviderFuture.addListener({
                                    val provider = cameraProviderFuture.get()

                                    val preview = Preview.Builder().build().also {
                                        it.setSurfaceProvider(previewView.surfaceProvider)
                                    }

                                    val imageAnalysis = ImageAnalysis.Builder()
                                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                                        .setTargetResolution(android.util.Size(640, 480))
                                        .build()
                                        .also { analysis ->
                                            analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                                                processFrame(imageProxy, vm)
                                            }
                                        }

                                    val cameraSelector = CameraSelector.Builder()
                                        .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                                        .build()

                                    try {
                                        provider.unbindAll()
                                        provider.bindToLifecycle(
                                            lifecycleOwner,
                                            cameraSelector,
                                            preview,
                                            imageAnalysis
                                        )
                                    } catch (e: Exception) {
                                        Log.e(TAG, "Camera bind failed", e)
                                    }
                                }, ContextCompat.getMainExecutor(ctx))

                                previewView
                            },
                            modifier = Modifier.fillMaxSize()
                        )

                        // Connection overlay
                        if (connState != JarvisViewModel.ConnectionState.CONNECTED) {
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .background(Color.Black.copy(alpha = 0.6f)),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    "Camera active — backend not connected",
                                    color     = Color.White,
                                    fontSize  = 13.sp,
                                    textAlign = TextAlign.Center,
                                    modifier  = Modifier.padding(16.dp)
                                )
                            }
                        }
                    } else {
                        // Camera is OFF
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Text("Camera Off", color = Color.Gray, fontSize = 16.sp)
                        }
                    }
                } else {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("Camera permission required", color = Color.Red)
                        Spacer(modifier = Modifier.height(8.dp))
                        Button(onClick = { permLauncher.launch(Manifest.permission.CAMERA) }) {
                            Text("Grant Permission")
                        }
                    }
                }
            }
        }
    }
}

/**
 * Convert a YUV_420_888 ImageProxy to a JPEG byte array and hand it to the VM.
 *
 * The backend already applies cv2.flip(frame, 1) before running MediaPipe,
 * so we send the un-flipped frame. The on-screen preview is mirrored for UX,
 * which CameraX handles automatically for the front camera.
 */
private fun processFrame(imageProxy: ImageProxy, vm: JarvisViewModel) {
    try {
        val bitmap    = imageProxy.toBitmap()
        val rotated   = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees.toFloat())
        val out       = ByteArrayOutputStream()
        rotated.compress(Bitmap.CompressFormat.JPEG, 50, out)
        vm.onCameraFrame(out.toByteArray())
    } catch (e: Exception) {
        Log.e(TAG, "Frame processing error", e)
    } finally {
        imageProxy.close()
    }
}

private fun rotateBitmap(src: Bitmap, degrees: Float): Bitmap {
    if (degrees == 0f) return src
    val matrix = Matrix().apply { postRotate(degrees) }
    return Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
}

// ─── Recognition ─────────────────────────────────────────────────────────────

@Composable
fun RecognitionSection(vm: JarvisViewModel) {
    val sign       by vm.currentSign.collectAsState()
    val confidence by vm.confidence.collectAsState()
    val holdRatio  by vm.holdRatio.collectAsState()
    val connState  by vm.connectionState.collectAsState()
    val autoSpeak  by vm.autoSpeakEnabled.collectAsState()

    val isConnected = connState == JarvisViewModel.ConnectionState.CONNECTED

    Card(
        shape    = RoundedCornerShape(12.dp),
        colors   = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Sign Recognition", fontWeight = FontWeight.SemiBold, fontSize = 18.sp)
            Spacer(modifier = Modifier.height(16.dp))

            Text("Current Sign", color = Color.Gray, fontSize = 14.sp)
            Text(
                text       = if (isConnected) sign else "Waiting for camera input",
                fontSize   = 24.sp,
                fontWeight = FontWeight.Bold,
                color      = if (isConnected && sign != "No sign detected")
                    MaterialTheme.colorScheme.secondary
                else
                    MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Confidence
            val confDisplay = if (isConnected) (confidence * 100).toInt() else 0
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("Confidence", fontSize = 14.sp)
                Text("$confDisplay%", fontSize = 14.sp)
            }
            Spacer(modifier = Modifier.height(4.dp))
            LinearProgressIndicator(
                progress     = { if (isConnected) confidence else 0f },
                modifier     = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)),
                color        = MaterialTheme.colorScheme.secondary,
                trackColor   = MaterialTheme.colorScheme.surfaceVariant,
            )

            Spacer(modifier = Modifier.height(12.dp))

            // Hold
            val holdDisplay = if (isConnected) (holdRatio * 100).toInt() else 0
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("Hold to Confirm", fontSize = 14.sp)
                Text("$holdDisplay%", fontSize = 14.sp)
            }
            Spacer(modifier = Modifier.height(4.dp))
            LinearProgressIndicator(
                progress   = { if (isConnected) holdRatio else 0f },
                modifier   = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)),
                color      = Color(0xFFFF9800),
                trackColor = MaterialTheme.colorScheme.surfaceVariant,
            )

            Spacer(modifier = Modifier.height(16.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(
                        checked = autoSpeak,
                        onCheckedChange = { vm.toggleAutoSpeak() }
                    )
                    Text("Auto Speak")
                }
                
                Button(
                    onClick = {
                        val currentSign = vm.currentSign.value
                        if (currentSign != "No sign detected" && currentSign.isNotBlank()) {
                            vm.manualSpeak(currentSign)
                        }
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                    enabled = isConnected
                ) {
                    Text("Speak Sign")
                }
            }
        }
    }
}

// ─── Sentence ─────────────────────────────────────────────────────────────────

@Composable
fun SentenceSection(vm: JarvisViewModel, tts: TextToSpeech?, ttsReady: Boolean) {
    val sentence  by vm.sentence.collectAsState()
    val connState by vm.connectionState.collectAsState()
    val connected = connState == JarvisViewModel.ConnectionState.CONNECTED

    Card(
        shape    = RoundedCornerShape(12.dp),
        colors   = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Constructed Sentence", fontWeight = FontWeight.SemiBold, fontSize = 18.sp)
            Spacer(modifier = Modifier.height(12.dp))

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant)
                    .padding(16.dp)
            ) {
                Text(
                    text     = sentence,
                    fontSize = 18.sp,
                    color    = if (sentence == "...") Color.LightGray else Color.White
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                OutlinedButton(
                    onClick  = { vm.deleteWord() },
                    modifier = Modifier.weight(1f),
                    enabled  = connected
                ) { Text("Delete Word") }

                Spacer(modifier = Modifier.width(8.dp))

                Button(
                    onClick  = { vm.clearSentence() },
                    colors   = ButtonDefaults.buttonColors(containerColor = Color(0xFFD32F2F)),
                    modifier = Modifier.weight(1f),
                    enabled  = connected
                ) { Text("Clear All") }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Button(
                onClick = { 
                    if (ttsReady && sentence != "...") {
                        tts?.speak(sentence, TextToSpeech.QUEUE_FLUSH, null, null)
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                enabled = connected
            ) {
                Text("Speak Sentence")
            }
        }
    }
}

// ─── Translation ──────────────────────────────────────────────────────────────

@Composable
fun TranslationSection(
    vm:       JarvisViewModel,
    tts:      TextToSpeech?,
    ttsReady: Boolean
) {
    var expanded          by remember { mutableStateOf(false) }
    val selectedLangCode  by vm.selectedLanguage.collectAsState()
    val translatedText    by vm.translatedText.collectAsState()
    val connState         by vm.connectionState.collectAsState()
    val translationSource by vm.translationSource.collectAsState()
    val connected         = connState == JarvisViewModel.ConnectionState.CONNECTED

    val selectedLangName = LANGUAGES.find { it.second == selectedLangCode }?.first ?: "Malayalam"

    Card(
        shape    = RoundedCornerShape(12.dp),
        colors   = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Translation & Speech", fontWeight = FontWeight.SemiBold, fontSize = 18.sp)
            Spacer(modifier = Modifier.height(12.dp))

            // Source Selector
            Row(verticalAlignment = Alignment.CenterVertically) {
                RadioButton(
                    selected = translationSource == TranslationSource.SIGN_SENTENCE,
                    onClick = { vm.setTranslationSource(TranslationSource.SIGN_SENTENCE) }
                )
                Text("Sign Sentence", fontSize = 14.sp)
                Spacer(modifier = Modifier.width(16.dp))
                RadioButton(
                    selected = translationSource == TranslationSource.SPEECH_TEXT,
                    onClick = { vm.setTranslationSource(TranslationSource.SPEECH_TEXT) }
                )
                Text("Speech Text", fontSize = 14.sp)
            }

            Spacer(modifier = Modifier.height(12.dp))

            Text("Translate to:", fontSize = 14.sp, color = Color.Gray)
            Box(modifier = Modifier.fillMaxWidth()) {
                OutlinedButton(
                    onClick  = { expanded = true },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        selectedLangName,
                        modifier  = Modifier.weight(1f),
                        textAlign = TextAlign.Start
                    )
                    Icon(Icons.Default.ArrowDropDown, contentDescription = null)
                }
                DropdownMenu(
                    expanded          = expanded,
                    onDismissRequest  = { expanded = false },
                    modifier          = Modifier.fillMaxWidth(0.9f)
                ) {
                    LANGUAGES.forEach { (name, code) ->
                        DropdownMenuItem(
                            text    = { Text(name) },
                            onClick = {
                                vm.setLanguage(code)
                                expanded = false
                            }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Button(
                onClick  = { vm.requestTranslation() },
                modifier = Modifier.fillMaxWidth(),
                enabled  = connected,
                colors   = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF0277BD)
                )
            ) { Text("Translate") }

            Spacer(modifier = Modifier.height(12.dp))

            Text("Translated Text:", fontSize = 14.sp, color = Color.Gray)
            OutlinedTextField(
                value = translatedText,
                onValueChange = { vm.setTranslatedText(it) },
                modifier = Modifier.fillMaxWidth(),
                textStyle = LocalTextStyle.current.copy(fontSize = 18.sp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = MaterialTheme.colorScheme.primary,
                    unfocusedBorderColor = MaterialTheme.colorScheme.surfaceVariant,
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White
                )
            )

            Spacer(modifier = Modifier.height(12.dp))

            // Speak button — Android TTS
            Button(
                onClick = {
                    val speak = translatedText.trim()
                    if (ttsReady && speak.isNotEmpty() && speak != "...") {
                        // Set locale for the target language
                        val locale = when (selectedLangCode) {
                            "ml" -> Locale("ml", "IN")
                            "hi" -> Locale("hi", "IN")
                            "ta" -> Locale("ta", "IN")
                            "kn" -> Locale("kn", "IN")
                            "te" -> Locale("te", "IN")
                            else -> Locale.ENGLISH
                        }
                        tts?.language = locale
                        tts?.speak(speak, TextToSpeech.QUEUE_FLUSH, null, "jarvis_tts")
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                colors   = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.primary
                )
            ) {
                Icon(Icons.Default.PlayArrow, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Speak Translation")
            }
        }
    }
}

// ─── Speech Input ─────────────────────────────────────────────────────────────

@Composable
fun SpeechInputSection(vm: JarvisViewModel) {
    val context = LocalContext.current
    val speechInput by vm.speechInputText.collectAsState()

    var hasAudioPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED
        )
    }

    val permLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { hasAudioPermission = it }
    )

    var sttStatus      by remember { mutableStateOf("Ready") }
    var isListening    by remember { mutableStateOf(false) }

    // SpeechRecognizer must be created/destroyed on the main thread
    var recognizer by remember { mutableStateOf<SpeechRecognizer?>(null) }

    DisposableEffect(Unit) {
        onDispose {
            recognizer?.destroy()
            recognizer = null
        }
    }

    fun startListening() {
        if (!hasAudioPermission) {
            permLauncher.launch(Manifest.permission.RECORD_AUDIO)
            return
        }
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            sttStatus = "Speech recognition not available on this device"
            return
        }

        recognizer?.destroy()
        recognizer = SpeechRecognizer.createSpeechRecognizer(context).also { sr ->
            sr.setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(p: OsBundle?)       { sttStatus = "Listening..."; isListening = true }
                override fun onBeginningOfSpeech()                 { sttStatus = "Listening..." }
                override fun onRmsChanged(rmsdB: Float)            {}
                override fun onBufferReceived(buffer: ByteArray?)  {}
                override fun onEndOfSpeech()                       { sttStatus = "Processing..." }
                override fun onError(error: Int) {
                    sttStatus = "Error: ${speechError(error)}"
                    isListening = false
                }
                override fun onResults(results: OsBundle?) {
                    val matches = results
                        ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    val recognizedText = matches?.firstOrNull() ?: "..."
                    vm.setSpeechInputText(recognizedText)
                    vm.setTranslationSource(TranslationSource.SPEECH_TEXT)
                    sttStatus      = "Done"
                    isListening    = false
                }
                override fun onPartialResults(p: OsBundle?) {}
                override fun onEvent(type: Int, p: OsBundle?) {}
            })
        }

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }
        recognizer?.startListening(intent)
    }

    fun stopListening() {
        recognizer?.stopListening()
        isListening = false
        sttStatus   = "Stopped"
    }

    Card(
        shape    = RoundedCornerShape(12.dp),
        colors   = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Hearing Person Reply", fontWeight = FontWeight.SemiBold, fontSize = 18.sp)
            Spacer(modifier = Modifier.height(4.dp))
            Text("Listen to speech and convert it to text", fontSize = 14.sp, color = Color.Gray)
            Spacer(modifier = Modifier.height(12.dp))

            Text(
                text     = "Status: $sttStatus",
                fontSize = 12.sp,
                color    = if (isListening) Color.Green else Color.Gray
            )
            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick  = { startListening() },
                    modifier = Modifier.weight(1f),
                    enabled  = !isListening,
                    colors   = ButtonDefaults.buttonColors(containerColor = Color(0xFF0288D1))
                ) {
                    Icon(Icons.Default.Mic, contentDescription = null)
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Listen")
                }
                if (isListening) {
                    Button(
                        onClick  = { stopListening() },
                        modifier = Modifier.weight(1f),
                        colors   = ButtonDefaults.buttonColors(containerColor = Color(0xFFD32F2F))
                    ) {
                        Icon(Icons.Default.Stop, contentDescription = null)
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Stop")
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))
            Text("Recognized Speech:", fontSize = 14.sp, color = Color.Gray)
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant)
                    .padding(16.dp)
            ) {
                Text(
                    text      = speechInput,
                    fontSize  = 16.sp,
                    color     = if (speechInput == "..." || speechInput.isBlank()) Color.LightGray else Color.White,
                    fontStyle = if (speechInput == "..." || speechInput.isBlank()) androidx.compose.ui.text.font.FontStyle.Italic
                    else        androidx.compose.ui.text.font.FontStyle.Normal
                )
            }
        }
    }
}

@Composable
fun ManageSignsSection() {
    Card(
        shape  = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Manage Signs", fontWeight = FontWeight.SemiBold, fontSize = 18.sp)
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                "To add a new sign to the system:\n" +
                "1. Open the JARVIS web interface on your laptop.\n" +
                "2. Navigate to the 'Collect Data' section.\n" +
                "3. Enter the new sign name.\n" +
                "4. Capture frames using the laptop webcam.\n" +
                "5. Train the model using the 'Train Model' button.",
                fontSize = 14.sp,
                color = Color.LightGray
            )
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

private fun speechError(code: Int): String = when (code) {
    SpeechRecognizer.ERROR_AUDIO                -> "Audio error"
    SpeechRecognizer.ERROR_CLIENT               -> "Client error"
    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "No permission"
    SpeechRecognizer.ERROR_NETWORK              -> "Network error"
    SpeechRecognizer.ERROR_NETWORK_TIMEOUT      -> "Network timeout"
    SpeechRecognizer.ERROR_NO_MATCH             -> "No match found"
    SpeechRecognizer.ERROR_RECOGNIZER_BUSY      -> "Recognizer busy"
    SpeechRecognizer.ERROR_SERVER               -> "Server error"
    SpeechRecognizer.ERROR_SPEECH_TIMEOUT       -> "No speech detected"
    else                                         -> "Unknown error ($code)"
}

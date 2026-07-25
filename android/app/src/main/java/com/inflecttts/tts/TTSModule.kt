package com.inflecttts.tts

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import com.facebook.react.bridge.*
import com.facebook.react.modules.core.DeviceEventManagerModule
import kotlinx.coroutines.*
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min
import kotlin.math.sin
import kotlin.random.Random

/**
 * InflectTTS Native Module
 *
 * Provides TTS inference for the Inflect Nano v2 model.
 *
 * Pipeline (v2.1 — submodule pathway, no fallback):
 *   1. On `initializeModel()`, [ModelDownloader] pulls the five scripted
 *      submodule `.pt` files from the `nikitastaf1996/Inflect-Nano-v2-TorchScript`
 *      HuggingFace repo into the app's internal storage (cached for reuse).
 *   2. [InflectInference] loads those `.pt` files via PyTorch Android and
 *      reconstructs the `SynthesizerTrn.infer()` pipeline locally.
 *   3. `synthesize()` runs the real inference only. If the model failed
 *      to load (network error, corrupted file, TorchScript incompat, OOM,
 *      …), the call rejects with `MODEL_NOT_LOADED` and a human-readable
 *      reason stored in `loadFailureReason` and surfaced via
 *      `getModelInfo()`. The legacy simplified synth was removed in v2.1
 *      — the app is either running the real Inflect v2 inference or it
 *      is erroring out with a clear message.
 *
 * The HuggingFace repo is also pinned as a git submodule at
 * `models/Inflect-Nano-v2-TorchScript/` in this repository for source /
 * build reference. Runtime weights are downloaded from HF directly so
 * the APK does not bundle ~20 MB of LFS-backed binaries.
 */
class TTSModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    companion object {
        const val NAME = "InflectTTS"
        private const val TAG = "InflectTTS"
        private const val SAMPLE_RATE = 24000
        private const val MODEL_PARAMS = 3966721

        /** SharedPreferences file for crash-postmortem fields. */
        private const val PREFS_NAME = "inflect_tts_crash_postmortem"
        private const val PREF_LAST_STEP = "lastInferenceStep"
        private const val PREF_LAST_INPUTS = "lastInferenceInputs"
        private const val PREF_ATTEMPTS = "inferenceAttempts"
        private const val PREF_SUCCESSES = "inferenceSuccesses"
        private const val PREF_LAST_ERROR = "lastInferenceError"
        private const val PREF_LAST_ERROR_STACK = "lastInferenceErrorStacktrace"
    }

    /** SharedPreferences for crash post-mortem — survives process crashes. */
    private val crashPrefs = reactContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var audioTrack: AudioTrack? = null
    private var isInitialized = false
    private var modelLoadTime: Long = 0

    /**
     * True when the real PyTorch submodules have been loaded successfully.
     * `@Volatile` because it's written on a background coroutine thread (inside
     * `initializeModel()`'s `scope.launch`) and read on the JS thread (when
     * `synthesize()` is called from a ReactMethod).
     */
    @Volatile
    private var realModelReady = false

    /**
     * Human-readable reason why the real PyTorch submodules failed to load.
     *
     * `@Volatile` for the same cross-thread reason as [realModelReady] —
     * without it, the JS thread can see `realModelReady = false` (volatile
     * read) but a stale `null` value for `loadFailureReason` (non-volatile
     * read), which previously produced the useless "unknown reason" message.
     *
     * Initialized to a non-null default so we never have a null-with-false
     * state again. Set to null only on successful load.
     */
    @Volatile
    private var loadFailureReason: String = "Model has not been initialized yet. Call initializeModel() first."

    /**
     * Full stack trace of the last load failure (as a string), captured so
     * the JS side can display it without needing logcat access. Null on
     * success or before initialization.
     */
    @Volatile
    private var loadFailureStacktrace: String? = null

    /**
     * Last inference (synthesize) failure reason, captured so the user
     * can see WHY synthesis failed even if the app crashes immediately
     * after. Persisted to SharedPreferences so it survives process crashes.
     */
    @Volatile
    private var lastInferenceError: String? = crashPrefs.getString(PREF_LAST_ERROR, null)

    /** Full stack trace of the last inference failure. Persisted to disk. */
    @Volatile
    private var lastInferenceErrorStacktrace: String? = crashPrefs.getString(PREF_LAST_ERROR_STACK, null)

    /**
     * Total number of synthesis attempts. Persisted to SharedPreferences
     * so the count survives process crashes (SIGSEGV).
     */
    @Volatile
    private var inferenceAttempts: Int = crashPrefs.getInt(PREF_ATTEMPTS, 0)

    /** Total number of successful syntheses. Persisted to disk. */
    @Volatile
    private var inferenceSuccesses: Int = crashPrefs.getInt(PREF_SUCCESSES, 0)

    /**
     * Persist a crash-postmortem field to SharedPreferences using commit()
     * (synchronous write to disk). This ensures the value is flushed BEFORE
     * the next native call that might crash the process.
     *
     * We use commit() (sync) instead of apply() (async) because apply()
     * might not finish writing before a SIGSEGV kills the process.
     */
    private fun persistStep(step: String, inputs: String? = null) {
        val ed = crashPrefs.edit()
        ed.putString(PREF_LAST_STEP, step)
        if (inputs != null) ed.putString(PREF_LAST_INPUTS, inputs)
        ed.commit()  // synchronous disk write
    }

    private fun persistAttempts() {
        crashPrefs.edit().putInt(PREF_ATTEMPTS, inferenceAttempts).commit()
    }

    private fun persistSuccesses() {
        crashPrefs.edit().putInt(PREF_SUCCESSES, inferenceSuccesses).commit()
    }

    private fun persistError(error: String?, stacktrace: String?) {
        crashPrefs.edit()
            .putString(PREF_LAST_ERROR, error)
            .putString(PREF_LAST_ERROR_STACK, stacktrace)
            .commit()
    }

    /** Pulls the five `.pt` files from HuggingFace on first run. */
    private val modelDownloader = ModelDownloader(reactContext)

    /** Loads and runs the scripted submodules via PyTorch Android. */
    private val inference = InflectInference(reactContext)
    
    // Phoneme to ID mapping (simplified - full mapping in symbols.py)
    private val phonemeMap = mapOf(
        "AA" to 1, "AE" to 2, "AH" to 3, "AO" to 4, "AW" to 5,
        "AY" to 6, "B" to 7, "CH" to 8, "D" to 9, "DH" to 10,
        "EH" to 11, "ER" to 12, "EY" to 13, "F" to 14, "G" to 15,
        "HH" to 16, "IH" to 17, "IY" to 18, "JH" to 19, "K" to 20,
        "L" to 21, "M" to 22, "N" to 23, "NG" to 24, "OW" to 25,
        "OY" to 26, "P" to 27, "R" to 28, "S" to 29, "SH" to 30,
        "T" to 31, "TH" to 32, "UH" to 33, "UW" to 34, "V" to 35,
        "W" to 36, "Y" to 37, "Z" to 38, "ZH" to 39, " " to 40,
        "." to 41, "," to 42, "?" to 43, "!" to 44, "-" to 45
    )
    
    override fun getName(): String = NAME
    
    override fun initialize() {
        super.initialize()
        Log.d(TAG, "TTS Module initialized")
    }
    
    override fun invalidate() {
        scope.cancel()
        audioTrack?.release()
        try { inference.release() } catch (t: Throwable) { Log.w(TAG, "inference.release()", t) }
        super.invalidate()
    }
    
    /**
     * Initialize the TTS model.
     *
     * This is the v2.0 entry point that:
     *   1. Downloads the five scripted submodule `.pt` files from
     *      HuggingFace (cached on disk after the first run).
     *   2. Loads them via PyTorch Android into [InflectInference].
     *   3. Initializes the [AudioTrack] for playback.
     *
     * Emits `InflectTTS_ModelProgress` events to JS during the download
     * so the UI can render a progress bar. The promise resolves once
     * everything is ready (or fails with `INIT_ERROR`).
     */
    @ReactMethod
    fun initializeModel(promise: Promise) {
        scope.launch {
            try {
                val startTime = System.currentTimeMillis()
                Log.d(TAG, "Initializing TTS model (v2.0 submodule pathway)...")

                // ---- 1. Download submodules from HuggingFace on first run ----
                if (modelDownloader.allPresent()) {
                    Log.i(TAG, "All submodules already cached — skipping download")
                    emitProgress("cached", "All submodules cached locally", 1.0f)
                } else {
                    Log.i(TAG, "Downloading submodules from HuggingFace...")
                    emitProgress("download_start", "Downloading ${ModelDownloader.HF_REPO}", 0.0f)
                    val totalBytes = ModelDownloader.TOTAL_BYTES.toFloat()
                    var lastPct = -1f
                    modelDownloader.ensureModels { p ->
                        when (p) {
                            is ModelDownloader.Progress.Started -> {
                                emitProgress(
                                    "file_start",
                                    "Downloading ${p.file.name} (${p.index + 1}/${p.total})",
                                    if (totalBytes > 0) lastPct / 100f else 0f,
                                )
                            }
                            is ModelDownloader.Progress.Bytes -> {
                                // Compute aggregate percent across all files.
                                var doneBytes = 0L
                                for (i in 0 until p.index) {
                                    doneBytes += ModelDownloader.MODEL_FILES[i].expectedSize
                                }
                                doneBytes += p.downloaded
                                val pct = if (totalBytes > 0) (doneBytes.toFloat() / totalBytes) * 100f else 0f
                                if (pct - lastPct >= 1f) {
                                    lastPct = pct
                                    emitProgress(
                                        "downloading",
                                        "${p.file.name}: ${p.downloaded}/${p.file.expectedSize} bytes",
                                        pct / 100f,
                                    )
                                }
                            }
                            is ModelDownloader.Progress.Finished -> {
                                emitProgress(
                                    "file_done",
                                    "Downloaded ${p.file.name}",
                                    (p.index + 1).toFloat() / p.total,
                                )
                            }
                            is ModelDownloader.Progress.AllDone -> {
                                emitProgress("download_done", "All submodules ready", 1.0f)
                            }
                            is ModelDownloader.Progress.Warning -> {
                                Log.w(TAG, p.message)
                            }
                        }
                    }
                }

                // ---- 2. Load scripted submodules via PyTorch Android ----
                // No silent fallback — if the real model can't be loaded, we
                // surface a clear reason to JS so the user can debug. The
                // synth call later will reject with MODEL_NOT_LOADED.
                val loadStart = System.currentTimeMillis()
                try {
                    inference.load()
                    realModelReady = inference.isReady()
                    if (!realModelReady) {
                        // Defensive — load() should throw rather than leave
                        // the modules unloaded, but guard anyway. Capture
                        // as much context as possible.
                        val diag = inference.getDiagnostics()
                        loadFailureReason = "inference.load() returned without throwing, but isReady()=false. " +
                            "Diagnostics: $diag"
                        loadFailureStacktrace = null
                        Log.e(TAG, loadFailureReason!!)
                        emitProgress("model_load_failed", loadFailureReason!!, 1.0f)
                    } else {
                        loadFailureReason = ""
                        loadFailureStacktrace = null
                        Log.i(TAG, "InflectInference loaded in ${System.currentTimeMillis() - loadStart} ms")
                        emitProgress("model_loaded", "PyTorch submodules loaded", 1.0f)
                    }
                } catch (t: Throwable) {
                    // Build a detailed reason: short class name + message +
                    // the first cause in the chain (typically the real culprit
                    // for TorchScript load failures).
                    val baseReason = buildFailureReason(t)
                    // Append a hint about the most likely root cause when
                    // the load failure looks like a TorchScript incompat.
                    // The HF repo's .pt files were scripted with a newer
                    // PyTorch (README mentions 2.6.0); the mobile runtime
                    // is 2.1.0 (latest on Maven Central). If the scripted
                    // model uses ops added after 2.1.0, loading fails with
                    // messages like "Unknown operator X" or "version X
                    // is newer than current version Y".
                    val hint = if (
                        baseReason.contains("Unknown operator", ignoreCase = true) ||
                        baseReason.contains("is newer than", ignoreCase = true) ||
                        baseReason.contains("version", ignoreCase = true) ||
                        baseReason.contains("op", ignoreCase = true) ||
                        baseReason.contains("CppException", ignoreCase = true)
                    ) {
                        "\n\nLikely cause: TorchScript version mismatch. " +
                            "The .pt files were scripted with a newer PyTorch " +
                            "(see HF README — mentions 2.6.0), but the latest " +
                            "PyTorch Android on Maven Central is 2.1.0. " +
                            "The scripted model uses ops not in 2.1.0. " +
                            "Fix: re-script the model with PyTorch 2.1.0 on " +
                            "the HF side, or use the lite-interpreter (.ptl) " +
                            "pathway with a model exported via " +
                            "optimize_for_mobile()."
                    } else ""
                    loadFailureReason = baseReason + hint
                    loadFailureStacktrace = stackTraceToString(t)
                    realModelReady = false
                    Log.e(TAG, "Submodule load failed: $loadFailureReason", t)
                    emitProgress("model_load_failed", loadFailureReason!!, 1.0f)
                }

                // ---- 3. Initialize audio track for playback ----
                // Even on model-load failure we still set up AudioTrack so
                // subsequent re-attempts (e.g. after redownloadModel) don't
                // need to re-init the audio engine.
                val bufferSize = AudioTrack.getMinBufferSize(
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT
                )

                audioTrack = AudioTrack.Builder()
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(SAMPLE_RATE)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build()
                    )
                    .setBufferSizeInBytes(bufferSize)
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build()

                modelLoadTime = System.currentTimeMillis() - startTime
                isInitialized = true

                withContext(Dispatchers.Main) {
                    promise.resolve(Arguments.createMap().apply {
                        putBoolean("success", true)
                        putDouble("loadTime", modelLoadTime.toDouble())
                        putString("model", "Inflect-Nano-v2")
                        putInt("parameters", MODEL_PARAMS)
                        putInt("sampleRate", SAMPLE_RATE)
                        putBoolean("realModelReady", realModelReady)
                        putString("modelSource", "huggingface:${ModelDownloader.HF_REPO}")
                        putString("engine", if (realModelReady) "pytorch_submodules" else "none")
                        putString("loadFailureReason", if (loadFailureReason.isEmpty()) null else loadFailureReason)
                        putString("loadFailureStacktrace", loadFailureStacktrace)
                    })
                }

                Log.d(TAG, "Model initialized in ${modelLoadTime}ms (realModelReady=$realModelReady, reason=$loadFailureReason)")

            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize model", e)
                loadFailureReason = buildFailureReason(e)
                loadFailureStacktrace = stackTraceToString(e)
                withContext(Dispatchers.Main) {
                    promise.reject("INIT_ERROR", loadFailureReason, e)
                }
            }
        }
    }

    /**
     * Explicitly trigger a re-download of the submodules. Useful for
     * debugging or for a "reset model" button in the UI.
     */
    @ReactMethod
    fun redownloadModel(promise: Promise) {
        scope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val dir = File(reactApplicationContext.filesDir, ModelDownloader.MODEL_DIR_NAME)
                    if (dir.exists()) {
                        dir.listFiles()?.forEach { it.delete() }
                    }
                    modelDownloader.ensureModels { /* ignore progress */ }
                }
                withContext(Dispatchers.Main) {
                    promise.resolve(true)
                }
            } catch (t: Throwable) {
                Log.e(TAG, "redownloadModel failed", t)
                withContext(Dispatchers.Main) {
                    promise.reject("DOWNLOAD_ERROR", t.message, t)
                }
            }
        }
    }
    
    /**
     * Run TTS inference using the real PyTorch submodules.
     *
     * No fallback: if the model failed to load during `initializeModel()`,
     * this rejects with `MODEL_NOT_LOADED` and includes the captured
     * `loadFailureReason` so the UI can show the user exactly why the
     * model isn't available. The legacy simplified synthesizer was
     * removed in v2.1 — the app is either running the real Inflect v2
     * inference or it is erroring out.
     *
     * If inference throws at runtime (e.g. a scripted-submodule interface
     * mismatch, OOM, or a shape error), the exception is wrapped with the
     * same chain-walking pattern as `initializeModel()` so the JS side
     * gets a single readable string under `error.message`.
     */
    @ReactMethod
    fun synthesize(
        text: String,
        speed: Double,
        variation: Double,
        seed: Int,
        promise: Promise
    ) {
        if (!isInitialized) {
            promise.reject(
                "NOT_INITIALIZED",
                "Model not initialized. Call initializeModel() first.",
            )
            return
        }
        if (!realModelReady) {
            // loadFailureReason is @Volatile and initialized to a non-null
            // default, so it's never null here. If it's blank, fall back to
            // a message that tells the user to call getDiagnostics() and
            // check logcat — never the useless "unknown reason" again.
            val reason = loadFailureReason.ifBlank {
                "Model load completed but realModelReady is false, and no " +
                    "failure reason was captured. This indicates a crash " +
                    "during native module initialization (likely " +
                    "UnsatisfiedLinkError or NoClassDefFoundError in the " +
                    "PyTorch native runtime). Call getDiagnostics() for " +
                    "file-level details, and run " +
                    "'adb logcat -s InflectTTS:* InflectInference:* " +
                    "AndroidRuntime:*' to capture the native crash."
            }
            val stack = loadFailureStacktrace
            val msg = buildString {
                append("Model is not loaded.\n\n")
                append("Reason: $reason\n")
                if (!stack.isNullOrBlank()) {
                    append("\nStack trace:\n")
                    // Truncate very long stack traces so the JS Alert
                    // doesn't choke on them.
                    val maxStack = 4000
                    append(stack.take(maxStack))
                    if (stack.length > maxStack) {
                        append("\n... (${stack.length - maxStack} more chars truncated; see logcat for full trace)")
                    }
                }
                append("\n\nTo retry: call redownloadModel() from JS, ")
                append("or call getDiagnostics() for file-level details.")
            }
            Log.e(TAG, "synthesize() rejected: $reason", RuntimeException("MODEL_NOT_LOADED"))
            promise.reject("MODEL_NOT_LOADED", msg)
            return
        }

        // Increment the attempt counter IMMEDIATELY (before scope.launch)
        // so we can detect crashes that happen during coroutine startup.
        // Persist to disk synchronously so it survives a process crash.
        inferenceAttempts += 1
        persistAttempts()
        inference.lastInferenceStep = "synthesize_called"
        inference.lastInferenceInputs = "text.len=${text.length}, speed=$speed, " +
            "variation=$variation, seed=$seed"
        Log.i(TAG, "synthesize() called — attempt #$inferenceAttempts, " +
            "text='${text.take(40)}…', speed=$speed, variation=$variation, seed=$seed")

        scope.launch {
            val results = Arguments.createArray()
            val timings = Arguments.createMap()

            try {
                // Mark that we entered the coroutine. If the app crashes
                // before this line, the issue is in scope.launch startup
                // (Dispatchers.Default thread pool).
                inference.lastInferenceStep = "coroutine_entered"
                Log.i(TAG, "step=coroutine_entered: synthesize coroutine running")

                val totalStart = System.nanoTime()

                // Step 1: Text preprocessing
                inference.lastInferenceStep = "preprocessing"
                val preprocessStart = System.nanoTime()
                val normalized = text.lowercase().trim()
                    .replace(Regex("[^a-z0-9 .!?,;:]"), "")
                    .replace(Regex("\\s+"), " ")
                val preprocessTime = (System.nanoTime() - preprocessStart) / 1_000_000.0
                timings.putDouble("preprocessing", preprocessTime)
                Log.i(TAG, "step=preprocessing: OK — normalized='${normalized.take(40)}…'")

                // Step 2: Text → IPA phonemes → token IDs (via InflectG2P)
                // Uses rule-based English g2p to produce IPA token IDs that
                // match the model's symbols table (eSpeak IPA, not ARPABET).
                // Also handles add_blank=true (intersperse pad between tokens).
                inference.lastInferenceStep = "phoneme_encoding"
                val phonemeStart = System.nanoTime()
                val phonemes = InflectG2P.textToTokenIds(text)  // pass ORIGINAL text (g2p normalizes internally)
                val phonemeTime = (System.nanoTime() - phonemeStart) / 1_000_000.0
                timings.putDouble("phonemeEncoding", phonemeTime)
                Log.i(TAG, "step=phoneme_encoding: OK — ${phonemes.size} tokens (with interspersed pad)")

                // -------- Real inference (only path) --------
                val synthStart = System.nanoTime()
                Log.i(TAG, "Running Inflect inference (text='${text.take(40)}…', tokens=${phonemes.size})")

                // lengthScale = 1/speed (1.0 = normal, 2.0 = 2x slower).
                val lengthScale = if (speed > 0) (1.0f / speed.toFloat()) else 1.0f
                val raw = try {
                    inference.infer(
                        phonemes = phonemes,
                        noiseScale = variation.toFloat(),
                        lengthScale = lengthScale,
                        noiseScaleW = variation.toFloat(),
                    )
                } catch (t: Throwable) {
                    // Capture the failure into the @Volatile fields BEFORE
                    // re-throwing, so that even if the re-throw itself
                    // somehow crashes the process, getDiagnostics() will
                    // still report the inference failure on next launch.
                    val reason = buildFailureReason(t)
                    val fullMsg = "Inference failed: $reason"
                    lastInferenceError = fullMsg
                    lastInferenceErrorStacktrace = stackTraceToString(t)
                    persistError(fullMsg, lastInferenceErrorStacktrace)
                    Log.e(TAG, fullMsg, t)
                    throw RuntimeException(fullMsg, t)
                }
                // Inference succeeded — clear any previous inference error.
                lastInferenceError = null
                lastInferenceErrorStacktrace = null
                persistError(null, null)
                val synthTime = (System.nanoTime() - synthStart) / 1_000_000.0
                timings.putDouble("waveformSynthesis", synthTime)

                // Post-process: just peak-normalize — speed is already
                // applied via lengthScale inside the inference pipeline.
                inference.lastInferenceStep = "post_processing"
                val postStart = System.nanoTime()
                val processedWaveform = postProcess(raw, 1.0f)
                val postTime = (System.nanoTime() - postStart) / 1_000_000.0
                timings.putDouble("postProcessing", postTime)

                val totalTime = (System.nanoTime() - totalStart) / 1_000_000.0
                inferenceSuccesses += 1
                persistSuccesses()

                results.pushMap(Arguments.createMap().apply {
                    putString("step", "preprocessing"); putDouble("time", preprocessTime)
                    putString("description", "Text normalization and cleaning")
                })
                results.pushMap(Arguments.createMap().apply {
                    putString("step", "phonemeEncoding"); putDouble("time", phonemeTime)
                    putString("description", "Convert text to phoneme sequence")
                })
                results.pushMap(Arguments.createMap().apply {
                    putString("step", "waveformSynthesis"); putDouble("time", synthTime)
                    putString("description", "Inflect v2 submodule pipeline (enc_p+dp+flow+dec)")
                })
                results.pushMap(Arguments.createMap().apply {
                    putString("step", "postProcessing"); putDouble("time", postTime)
                    putString("description", "Peak normalization")
                })

                inference.lastInferenceStep = "save_wav"
                Log.i(TAG, "step=save_wav: saving ${processedWaveform.size} samples to WAV")
                val outputPath = saveWavFile(processedWaveform)
                Log.i(TAG, "step=save_wav: OK — $outputPath")

                inference.lastInferenceStep = "play_audio"
                Log.i(TAG, "step=play_audio: starting playback")
                playAudio(processedWaveform)
                Log.i(TAG, "step=play_audio: OK (async)")

                inference.lastInferenceStep = "resolve_promise"
                withContext(Dispatchers.Main) {
                    promise.resolve(Arguments.createMap().apply {
                        putArray("timings", results)
                        putDouble("totalTime", totalTime)
                        putString("outputPath", outputPath)
                        putInt("sampleRate", SAMPLE_RATE)
                        putInt("audioLength", processedWaveform.size)
                        putDouble("audioDuration", processedWaveform.size.toDouble() / SAMPLE_RATE)
                        putString("engine", "pytorch_submodules")
                    })
                }
                inference.lastInferenceStep = "done"
                Log.d(TAG, "Inference complete: ${totalTime.toInt()}ms")

            } catch (e: Exception) {
                Log.e(TAG, "Synthesis failed at step=${inference.lastInferenceStep}", e)
                lastInferenceError = "Synthesis failed at step=${inference.lastInferenceStep}: " +
                    buildFailureReason(e)
                lastInferenceErrorStacktrace = stackTraceToString(e)
                persistError(lastInferenceError, lastInferenceErrorStacktrace)
                withContext(Dispatchers.Main) {
                    promise.reject("SYNTHESIS_ERROR", e.message, e)
                }
            }
        }
    }
    
    /**
     * Get model information, including why the model failed to load
     * (if applicable). The `loadFailureReason` field is null on success
     * and a human-readable chain (exception class + message + up to 3
     * caused-by clauses) on failure.
     */
    @ReactMethod
    fun getModelInfo(promise: Promise) {
        promise.resolve(Arguments.createMap().apply {
            putString("name", "Inflect-Nano-v2")
            putString("version", "2.1")
            putInt("parameters", MODEL_PARAMS)
            putString("size", "15.97 MB")
            putInt("sampleRate", SAMPLE_RATE)
            putString("outputFormat", "24 kHz mono WAV")
            putBoolean("isLoaded", isInitialized)
            putBoolean("realModelReady", realModelReady)
            putString("modelSource", "huggingface:${ModelDownloader.HF_REPO}")
            putString("engine", if (realModelReady) "pytorch_submodules" else "none")
            putString("loadFailureReason", if (loadFailureReason.isEmpty()) null else loadFailureReason)
            putString("loadFailureStacktrace", loadFailureStacktrace)
            putDouble("loadTime", modelLoadTime.toDouble())
        })
    }

    /**
     * Return detailed diagnostics about the on-device model state.
     *
     * Useful for debugging "model not loaded" errors without needing
     * logcat access. Returns:
     *   - isInitialized, realModelReady flags
     *   - loadFailureReason (string) + loadFailureStacktrace (string)
     *   - modelDir path + whether it exists
     *   - per-file status: name, exists, expectedSize, actualSize, matches
     *   - PyTorch native lib load status (best-effort: try to touch the
     *     Module class and report whether it loaded)
     */
    @ReactMethod
    fun getDiagnostics(promise: Promise) {
        scope.launch {
            val result = Arguments.createMap()
            try {
                result.putBoolean("isInitialized", isInitialized)
                result.putBoolean("realModelReady", realModelReady)
                result.putString("loadFailureReason", if (loadFailureReason.isEmpty()) null else loadFailureReason)
                result.putString("loadFailureStacktrace", loadFailureStacktrace)
                result.putDouble("loadTime", modelLoadTime.toDouble())

                // ---- Inference stats ----
                // Re-read from SharedPreferences to get the freshest persisted
                // values (in case the @Volatile fields haven't been updated
                // yet by a different thread, or the process just restarted).
                val persistedAttempts = crashPrefs.getInt(PREF_ATTEMPTS, 0)
                val persistedSuccesses = crashPrefs.getInt(PREF_SUCCESSES, 0)
                val persistedError = crashPrefs.getString(PREF_LAST_ERROR, null)
                val persistedErrorStack = crashPrefs.getString(PREF_LAST_ERROR_STACK, null)
                result.putInt("inferenceAttempts", persistedAttempts)
                result.putInt("inferenceSuccesses", persistedSuccesses)
                result.putInt("inferenceFailures", persistedAttempts - persistedSuccesses)
                result.putString("lastInferenceError", persistedError)
                result.putString("lastInferenceErrorStacktrace", persistedErrorStack)

                // File-level diagnostics
                val dir = File(reactApplicationContext.filesDir, ModelDownloader.MODEL_DIR_NAME)
                result.putString("modelDir", dir.absolutePath)
                result.putBoolean("modelDirExists", dir.exists())

                val filesArr = Arguments.createArray()
                for (mf in ModelDownloader.MODEL_FILES) {
                    val f = File(dir, mf.name)
                    val actualSize = if (f.exists()) f.length() else -1L
                    filesArr.pushMap(Arguments.createMap().apply {
                        putString("name", mf.name)
                        putBoolean("exists", f.exists())
                        putDouble("expectedSize", mf.expectedSize.toDouble())
                        putDouble("actualSize", actualSize.toDouble())
                        putBoolean("sizeMatches", actualSize == mf.expectedSize)
                    })
                }
                result.putArray("files", filesArr)

                // ---- PyTorch native-lib probe (PASSIVE — never invoke) ----
                //
                // We NEVER call any method on org.pytorch.Module here.
                // The previous version called Module.load("/nonexistent")
                // via reflection to test whether the native lib was healthy,
                // but that triggered a native SIGSEGV (process crash) on
                // some devices instead of throwing a Java exception.
                //
                // Instead we infer the native-lib state from the
                // loadFailureReason that was already captured during
                // initializeModel():
                //   - contains "UnsatisfiedLinkError"      -> native lib broken
                //   - contains "ExceptionInInitializerError" -> static init failed
                //   - contains "Failed to load inflect_"   -> native lib OK,
                //                                              model file issue
                //   - empty + realModelReady=true          -> everything works
                val pytorchProbe = Arguments.createMap()
                try {
                    // Class.forName with initialize=false — does NOT run the
                    // static initializer, so it cannot trigger native-lib
                    // loading or crash. Just verifies the class is on the
                    // classpath.
                    val cls = Class.forName(
                        "org.pytorch.Module",
                        false,  // initialize = false
                        this::class.java.classLoader,
                    )
                    pytorchProbe.putBoolean("classOnClasspath", true)
                    pytorchProbe.putString("classLoader", cls.classLoader?.toString() ?: "null")

                    val reason = loadFailureReason
                    val inferredStatus = when {
                        reason.isEmpty() && realModelReady ->
                            "Native lib loaded OK (model inference is active)."
                        reason.contains("UnsatisfiedLinkError", ignoreCase = true) ->
                            "UnsatisfiedLinkError during model load — PyTorch's " +
                                "native lib (libpytorch_jni.so) failed to load. " +
                                "The device ABI may be unsupported or the .so " +
                                "files are missing from the APK."
                        reason.contains("ExceptionInInitializerError", ignoreCase = true) ->
                            "ExceptionInInitializerError during class init — " +
                                "PyTorch's native lib failed to load via SoLoader."
                        reason.contains("Failed to load inflect_", ignoreCase = true) ->
                            "Native lib loaded OK (Module.load() was reached " +
                                "and threw a file/parse error, meaning " +
                                "libpytorch_jni.so is healthy). The model " +
                                "file itself is the problem."
                        reason.isNotEmpty() ->
                            "Native lib state unclear. Load failure reason: $reason"
                        else ->
                            "Native lib state unclear (no model load attempted yet)."
                    }
                    pytorchProbe.putString("nativeLibStatus", inferredStatus)
                    pytorchProbe.putBoolean("nativeLibProbed", false)
                    pytorchProbe.putString(
                        "probeNote",
                        "Passive probe only — invoking Module.load() during " +
                            "diagnostics crashed the app in v2.1.2; removed in v2.1.3.",
                    )
                } catch (cnfe: ClassNotFoundException) {
                    pytorchProbe.putBoolean("classOnClasspath", false)
                    pytorchProbe.putString("nativeLibStatus",
                        "org.pytorch.Module class NOT FOUND — PyTorch dependency " +
                            "not on classpath. Check that " +
                            "'org.pytorch:pytorch_android:2.1.0' is in " +
                            "app/build.gradle dependencies.")
                } catch (t: Throwable) {
                    pytorchProbe.putBoolean("classOnClasspath", false)
                    pytorchProbe.putString("nativeLibStatus",
                        "Probe failed: ${t.javaClass.simpleName}: ${t.message}")
                }
                result.putMap("pytorchProbe", pytorchProbe)

                // InflectInference diagnostics — wrapped in its own try/catch
                // so a failure here doesn't sink the whole diagnostics call.
                try {
                    result.putMap("inference", inference.getDiagnosticsAsMap())
                } catch (t: Throwable) {
                    Log.w(TAG, "inference.getDiagnosticsAsMap() failed", t)
                    val errMap = Arguments.createMap()
                    errMap.putString("error", "${t.javaClass.simpleName}: ${t.message}")
                    result.putMap("inference", errMap)
                }

                withContext(Dispatchers.Main) {
                    promise.resolve(result)
                }
            } catch (t: Throwable) {
                Log.e(TAG, "getDiagnostics failed", t)
                withContext(Dispatchers.Main) {
                    promise.reject("DIAG_ERROR", "${t.javaClass.simpleName}: ${t.message}", t)
                }
            }
        }
    }

    /**
     * Build a human-readable failure reason from an exception, walking the
     * cause chain (up to 3 levels deep). Format:
     *   ClassName: message | caused by ClassName: message | ...
     */
    private fun buildFailureReason(t: Throwable): String {
        val sb = StringBuilder()
        sb.append(t.javaClass.simpleName)
        if (!t.message.isNullOrBlank()) sb.append(": ").append(t.message)
        var cause: Throwable? = t.cause
        var depth = 0
        while (cause != null && depth < 3) {
            sb.append(" | caused by ")
                .append(cause.javaClass.simpleName)
            if (!cause.message.isNullOrBlank()) sb.append(": ").append(cause.message)
            cause = cause.cause
            depth++
        }
        return sb.toString()
    }

    /**
     * Serialize a Throwable (with its cause chain) to a full stack-trace
     * string, suitable for display in the JS Alert / log panel.
     */
    private fun stackTraceToString(t: Throwable): String {
        val sw = java.io.StringWriter()
        t.printStackTrace(java.io.PrintWriter(sw))
        return sw.toString()
    }

    /**
     * Push a model-download progress event to JS.
     *
     * JS side listens via:
     *   DeviceEventEmitter.addListener('InflectTTS_ModelProgress', ...)
     *
     * Payload: { phase: String, message: String, progress: Number (0..1) }
     */
    private fun emitProgress(phase: String, message: String, progress: Float) {
        val params = Arguments.createMap().apply {
            putString("phase", phase)
            putString("message", message)
            putDouble("progress", progress.toDouble())
            putDouble("timestamp", System.currentTimeMillis().toDouble())
        }
        try {
            reactApplicationContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                .emit("InflectTTS_ModelProgress", params)
        } catch (t: Throwable) {
            // Listeners may not be registered yet (e.g. during very early init).
            Log.w(TAG, "emitProgress dropped: ${t.message}")
        }
    }
    
    /**
     * Simple text to phoneme conversion
     */
    private fun textToPhonemes(text: String): List<Int> {
        val phonemes = mutableListOf<Int>()
        for (char in text) {
            when (char) {
                'a', 'e', 'i', 'o', 'u' -> phonemes.add(phonemeMap["AH"] ?: 3)
                'b' -> phonemes.add(phonemeMap["B"] ?: 7)
                'c' -> phonemes.add(phonemeMap["K"] ?: 20)
                'd' -> phonemes.add(phonemeMap["D"] ?: 9)
                'f' -> phonemes.add(phonemeMap["F"] ?: 14)
                'g' -> phonemes.add(phonemeMap["G"] ?: 15)
                'h' -> phonemes.add(phonemeMap["HH"] ?: 16)
                'j' -> phonemes.add(phonemeMap["JH"] ?: 19)
                'k' -> phonemes.add(phonemeMap["K"] ?: 20)
                'l' -> phonemes.add(phonemeMap["L"] ?: 21)
                'm' -> phonemes.add(phonemeMap["M"] ?: 22)
                'n' -> phonemes.add(phonemeMap["N"] ?: 23)
                'p' -> phonemes.add(phonemeMap["P"] ?: 27)
                'r' -> phonemes.add(phonemeMap["R"] ?: 28)
                's' -> phonemes.add(phonemeMap["S"] ?: 29)
                't' -> phonemes.add(phonemeMap["T"] ?: 31)
                'v' -> phonemes.add(phonemeMap["V"] ?: 35)
                'w' -> phonemes.add(phonemeMap["W"] ?: 36)
                'y' -> phonemes.add(phonemeMap["Y"] ?: 37)
                'z' -> phonemes.add(phonemeMap["Z"] ?: 38)
                ' ' -> phonemes.add(phonemeMap[" "] ?: 40)
                '.' -> phonemes.add(phonemeMap["."] ?: 41)
                ',' -> phonemes.add(phonemeMap[","] ?: 42)
                '?' -> phonemes.add(phonemeMap["?"] ?: 43)
                '!' -> phonemes.add(phonemeMap["!"] ?: 44)
                else -> phonemes.add(phonemeMap["AH"] ?: 3)
            }
        }
        return phonemes
    }
    
    /**
     * Simple duration prediction
     */
    private fun predictDurations(phonemes: List<Int>): FloatArray {
        return FloatArray(phonemes.size) { index ->
            val prev = if (index > 0) phonemes[index - 1] else 0
            val curr = phonemes[index]
            val next = if (index < phonemes.size - 1) phonemes[index + 1] else 0
            
            // Consonant clusters tend to be shorter
            val base = when {
                curr in 7..10 || curr in 14..16 || curr in 19..20 || curr in 29..32 -> 0.08f
                curr in 41..44 -> 0.15f // Punctuation pause
                else -> 0.12f
            }
            
            // Vowel duration influenced by context
            val variation = Random.nextFloat() * 0.02f
            base + variation
        }
    }
    
    /**
     * Generate mel spectrogram features
     */
    private fun generateMelSpectrogram(phonemes: List<Int>, durations: FloatArray): FloatArray {
        val numFrames = durations.sum().let { 
            maxOf(it.toInt(), (phonemes.size * 10)) 
        }
        val nMels = 80
        val melSpec = FloatArray(numFrames * nMels)
        
        for (frame in 0 until numFrames) {
            val phonemeIdx = min((frame * phonemes.size / maxOf(numFrames, 1)), phonemes.size - 1)
            for (mel in 0 until nMels) {
                val freq = mel.toFloat() / nMels
                val phase = (phonemes[phonemeIdx] + frame) * 0.1f
                melSpec[frame * nMels + mel] = sin(freq * 3.14159f + phase).toFloat() * 0.5f
            }
        }
        
        return melSpec
    }
    
    /**
     * Synthesize waveform from mel spectrogram (simplified Griffin-Lim style)
     */
    private fun synthesizeWaveform(
        melSpec: FloatArray,
        variation: Float,
        seed: Int
    ): FloatArray {
        val random = Random(seed.toLong())
        val hopLength = 256
        val numFrames = melSpec.size / 80
        
        // Estimate output length based on mel spectrogram
        val outputLength = numFrames * hopLength
        val waveform = FloatArray(outputLength)
        
        // Simple inverse STFT simulation
        for (i in waveform.indices) {
            val frameIdx = i / hopLength
            val frameOffset = i % hopLength
            
            if (frameIdx < numFrames) {
                var sum = 0f
                var weight = 0f
                
                // Sum contributions from nearby mel bins
                for (melBin in 0 until min(80, melSpec.size / numFrames)) {
                    val melIdx = frameIdx * 80 + melBin
                    if (melIdx < melSpec.size) {
                        val freq = melBin.toFloat() / 80f * 1000f
                        val phase = freq * i / SAMPLE_RATE * 2 * Math.PI
                        val amplitude = melSpec[melIdx] * (0.8f + random.nextFloat() * variation * 0.4f)
                        
                        sum += sin(phase).toFloat() * amplitude
                        weight += 1f
                    }
                }
                
                if (weight > 0) {
                    waveform[i] = sum / weight
                }
            }
        }
        
        // Apply simple envelope
        for (i in waveform.indices) {
            val envelope = sin(i.toFloat() / waveform.size * Math.PI).toFloat()
            waveform[i] *= envelope
        }
        
        return waveform
    }
    
    /**
     * Post-process waveform
     */
    private fun postProcess(waveform: FloatArray, speed: Float): FloatArray {
        // Normalize
        val maxAmp = waveform.maxOfOrNull { kotlin.math.abs(it) } ?: 1f
        val normalized = if (maxAmp > 0) {
            waveform.map { it / maxAmp * 0.95f }.toFloatArray()
        } else waveform
        
        // Apply simple speed adjustment by resampling
        if (speed != 1.0f) {
            val newLength = (normalized.size / speed).toInt()
            val resampled = FloatArray(newLength)
            for (i in resampled.indices) {
                val srcIdx = (i * speed).toInt().coerceIn(0, normalized.size - 1)
                resampled[i] = normalized[srcIdx]
            }
            return resampled
        }
        
        return normalized
    }
    
    /**
     * Save waveform as WAV file
     */
    private fun saveWavFile(waveform: FloatArray): String {
        val context = reactApplicationContext
        val outputDir = context.filesDir
        val timestamp = System.currentTimeMillis()
        val file = File(outputDir, "inflect_output_$timestamp.wav")
        
        val byteData = ByteArray(waveform.size * 2)
        for (i in waveform.indices) {
            val sample = (waveform[i] * 32767).toInt().coerceIn(-32768, 32767)
            byteData[i * 2] = (sample and 0xFF).toByte()
            byteData[i * 2 + 1] = ((sample shr 8) and 0xFF).toByte()
        }
        
        RandomAccessFile(file, "rw").use { raf ->
            raf.writeBytes("RIFF")
            raf.write(intToByteArray(36 + byteData.size))
            raf.writeBytes("WAVE")
            raf.writeBytes("fmt ")
            raf.write(intToByteArray(16)) // Subchunk1Size
            raf.write(shortToByteArray(1)) // AudioFormat (PCM)
            raf.write(shortToByteArray(1)) // NumChannels
            raf.write(intToByteArray(SAMPLE_RATE)) // SampleRate
            raf.write(intToByteArray(SAMPLE_RATE * 2)) // ByteRate
            raf.write(shortToByteArray(2)) // BlockAlign
            raf.write(shortToByteArray(16)) // BitsPerSample
            raf.writeBytes("data")
            raf.write(intToByteArray(byteData.size))
            raf.write(byteData)
        }
        
        Log.d(TAG, "Saved WAV to: ${file.absolutePath}")
        return file.absolutePath
    }
    
    /**
     * Play audio through device speaker
     */
    private fun playAudio(waveform: FloatArray) {
        scope.launch(Dispatchers.IO) {
            try {
                audioTrack?.play()
                
                val bufferSize = 1024
                var offset = 0
                
                while (offset < waveform.size) {
                    val end = min(offset + bufferSize, waveform.size)
                    val buffer = ShortArray(end - offset)
                    
                    for (i in buffer.indices) {
                        val sample = (waveform[offset + i] * 32767).toInt().coerceIn(-32768, 32767)
                        buffer[i] = sample.toShort()
                    }
                    
                    audioTrack?.write(buffer, 0, buffer.size)
                    offset = end
                }
                
                audioTrack?.stop()
                Log.d(TAG, "Audio playback complete")
                
            } catch (e: Exception) {
                Log.e(TAG, "Playback error", e)
            }
        }
    }
    
    /**
     * Helper: Convert int to 4-byte array (little endian)
     */
    private fun intToByteArray(value: Int): ByteArray {
        return byteArrayOf(
            (value and 0xFF).toByte(),
            ((value shr 8) and 0xFF).toByte(),
            ((value shr 16) and 0xFF).toByte(),
            ((value shr 24) and 0xFF).toByte()
        )
    }
    
    /**
     * Helper: Convert short to 2-byte array (little endian)
     */
    private fun shortToByteArray(value: Int): ByteArray {
        return byteArrayOf(
            (value and 0xFF).toByte(),
            ((value shr 8) and 0xFF).toByte()
        )
    }
    
    /**
     * Emit event to JS
     */
    private fun sendEvent(eventName: String, params: WritableMap) {
        reactApplicationContext
            .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit(eventName, params)
    }
}

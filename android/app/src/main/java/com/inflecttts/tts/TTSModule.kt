package com.inflecttts.tts

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
 * Pipeline (v2.0 — submodule pathway):
 *   1. On `initializeModel()`, [ModelDownloader] pulls the five scripted
 *      submodule `.pt` files from the `nikitastaf1996/Inflect-Nano-v2-TorchScript`
 *      HuggingFace repo into the app's internal storage (cached for reuse).
 *   2. [InflectInference] loads those `.pt` files via PyTorch Android and
 *      reconstructs the `SynthesizerTrn.infer()` pipeline locally.
 *   3. `synthesize()` first attempts real model inference; if anything
 *      fails (e.g. submodule interface drift, OOM), it falls back to the
 *      legacy simplified synthesizer so the app remains usable.
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
    }
    
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var audioTrack: AudioTrack? = null
    private var isInitialized = false
    private var modelLoadTime: Long = 0

    /** True when the real PyTorch submodules have been loaded successfully. */
    private var realModelReady = false

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
                val loadStart = System.currentTimeMillis()
                try {
                    inference.load()
                    realModelReady = inference.isReady()
                    Log.i(TAG, "InflectInference loaded in ${System.currentTimeMillis() - loadStart} ms (ready=$realModelReady)")
                    emitProgress(
                        if (realModelReady) "model_loaded" else "model_load_skipped",
                        if (realModelReady) "PyTorch submodules loaded" else "Submodule load skipped — using fallback synth",
                        1.0f,
                    )
                } catch (t: Throwable) {
                    Log.w(TAG, "Submodule load failed — will use fallback synth", t)
                    realModelReady = false
                    emitProgress("model_load_failed", "Submodule load failed: ${t.message}", 1.0f)
                }

                // ---- 3. Initialize audio track for playback ----
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
                    })
                }

                Log.d(TAG, "Model initialized in ${modelLoadTime}ms (realModelReady=$realModelReady)")

            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize model", e)
                withContext(Dispatchers.Main) {
                    promise.reject("INIT_ERROR", e.message, e)
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
     * Run TTS inference.
     *
     * If the real PyTorch submodules were loaded successfully during
     * `initializeModel()`, this routes the request through [InflectInference].
     * Otherwise (or if the real path throws), it falls back to the
     * legacy simplified synthesizer so the app remains usable.
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
            promise.reject("NOT_INITIALIZED", "Model not initialized")
            return
        }

        scope.launch {
            val results = Arguments.createArray()
            val timings = Arguments.createMap()

            try {
                val totalStart = System.nanoTime()

                // Step 1: Text preprocessing
                val preprocessStart = System.nanoTime()
                val normalized = text.lowercase().trim()
                    .replace(Regex("[^a-z0-9 .!?,;:]"), "")
                    .replace(Regex("\\s+"), " ")
                val preprocessTime = (System.nanoTime() - preprocessStart) / 1_000_000.0
                timings.putDouble("preprocessing", preprocessTime)

                // Step 2: Text to phonemes
                val phonemeStart = System.nanoTime()
                val phonemes = textToPhonemes(normalized)
                val phonemeTime = (System.nanoTime() - phonemeStart) / 1_000_000.0
                timings.putDouble("phonemeEncoding", phonemeTime)

                // -------- Real inference path (PyTorch submodules) --------
                if (realModelReady) {
                    try {
                        val synthStart = System.nanoTime()
                        Log.i(TAG, "Running real Inflect inference (text='${text.take(40)}…', phonemes=${phonemes.size})")

                        // lengthScale = 1/speed (1.0 = normal, 2.0 = 2x slower).
                        val lengthScale = if (speed > 0) (1.0f / speed.toFloat()) else 1.0f
                        val raw = inference.infer(
                            phonemes = phonemes.toIntArray(),
                            noiseScale = variation.toFloat(),
                            lengthScale = lengthScale,
                            noiseScaleW = variation.toFloat(),
                        )
                        val synthTime = (System.nanoTime() - synthStart) / 1_000_000.0
                        timings.putDouble("waveformSynthesis", synthTime)

                        // Post-process (normalize + speed-aware resampling already
                        // applied via lengthScale, so just normalize here).
                        val postStart = System.nanoTime()
                        val processedWaveform = postProcess(raw, 1.0f)
                        val postTime = (System.nanoTime() - postStart) / 1_000_000.0
                        timings.putDouble("postProcessing", postTime)

                        val totalTime = (System.nanoTime() - totalStart) / 1_000_000.0

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

                        val outputPath = saveWavFile(processedWaveform)
                        playAudio(processedWaveform)

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
                        Log.d(TAG, "Real inference complete: ${totalTime.toInt()}ms")
                        return@launch
                    } catch (t: Throwable) {
                        Log.w(TAG, "Real inference failed — falling back to simplified synth", t)
                        // Fall through to legacy path below.
                    }
                }

                // -------- Fallback: legacy simplified synthesizer --------

                // Step 3: Duration prediction (simplified)
                val durationStart = System.nanoTime()
                val durations = predictDurations(phonemes)
                val durationTime = (System.nanoTime() - durationStart) / 1_000_000.0
                timings.putDouble("durationPrediction", durationTime)

                // Step 4: Mel spectrogram generation (simplified)
                val melStart = System.nanoTime()
                val melSpec = generateMelSpectrogram(phonemes, durations)
                val melTime = (System.nanoTime() - melStart) / 1_000_000.0
                timings.putDouble("melGeneration", melTime)

                // Step 5: Waveform synthesis (main inference)
                val synthesisStart = System.nanoTime()
                val waveform = synthesizeWaveform(melSpec, variation.toFloat(), seed)
                val synthesisTime = (System.nanoTime() - synthesisStart) / 1_000_000.0
                timings.putDouble("waveformSynthesis", synthesisTime)

                // Step 6: Post-processing
                val postStart = System.nanoTime()
                val processedWaveform = postProcess(waveform, speed.toFloat())
                val postTime = (System.nanoTime() - postStart) / 1_000_000.0
                timings.putDouble("postProcessing", postTime)

                // Calculate total time
                val totalTime = (System.nanoTime() - totalStart) / 1_000_000.0

                // Add each step to results
                results.pushMap(Arguments.createMap().apply {
                    putString("step", "preprocessing")
                    putDouble("time", preprocessTime)
                    putString("description", "Text normalization and cleaning")
                })
                results.pushMap(Arguments.createMap().apply {
                    putString("step", "phonemeEncoding")
                    putDouble("time", phonemeTime)
                    putString("description", "Convert text to phoneme sequence")
                })
                results.pushMap(Arguments.createMap().apply {
                    putString("step", "durationPrediction")
                    putDouble("time", durationTime)
                    putString("description", "Predict phoneme durations")
                })
                results.pushMap(Arguments.createMap().apply {
                    putString("step", "melGeneration")
                    putDouble("time", melTime)
                    putString("description", "Generate mel spectrogram features")
                })
                results.pushMap(Arguments.createMap().apply {
                    putString("step", "waveformSynthesis")
                    putDouble("time", synthesisTime)
                    putString("description", "Neural vocoder waveform synthesis")
                })
                results.pushMap(Arguments.createMap().apply {
                    putString("step", "postProcessing")
                    putDouble("time", postTime)
                    putString("description", "Audio normalization and resampling")
                })

                // Save audio to file
                val outputPath = saveWavFile(processedWaveform)

                // Play audio
                playAudio(processedWaveform)

                withContext(Dispatchers.Main) {
                    promise.resolve(Arguments.createMap().apply {
                        putArray("timings", results)
                        putDouble("totalTime", totalTime)
                        putString("outputPath", outputPath)
                        putInt("sampleRate", SAMPLE_RATE)
                        putInt("audioLength", processedWaveform.size)
                        putDouble("audioDuration", processedWaveform.size.toDouble() / SAMPLE_RATE)
                        putString("engine", "fallback_synth")
                    })
                }

                Log.d(TAG, "Synthesis complete: ${totalTime.toInt()}ms")

            } catch (e: Exception) {
                Log.e(TAG, "Synthesis failed", e)
                withContext(Dispatchers.Main) {
                    promise.reject("SYNTHESIS_ERROR", e.message, e)
                }
            }
        }
    }
    
    /**
     * Get model information.
     */
    @ReactMethod
    fun getModelInfo(promise: Promise) {
        promise.resolve(Arguments.createMap().apply {
            putString("name", "Inflect-Nano-v2")
            putString("version", "2.0")
            putInt("parameters", MODEL_PARAMS)
            putString("size", "15.97 MB")
            putInt("sampleRate", SAMPLE_RATE)
            putString("outputFormat", "24 kHz mono WAV")
            putBoolean("isLoaded", isInitialized)
            putBoolean("realModelReady", realModelReady)
            putString("modelSource", "huggingface:${ModelDownloader.HF_REPO}")
            putString("engine", if (realModelReady) "pytorch_submodules" else "fallback_synth")
            putDouble("loadTime", modelLoadTime.toDouble())
        })
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

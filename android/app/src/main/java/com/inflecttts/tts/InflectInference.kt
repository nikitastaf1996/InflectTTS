package com.inflecttts.tts

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.WritableMap
import java.io.File
import java.nio.LongBuffer
import java.nio.FloatBuffer
import kotlin.math.min

/**
 * InflectInference (ONNX Runtime version)
 *
 * Loads the two ONNX graphs from `nikitastaf1996/Inflect-Nano-v2-Mobile`
 * and runs the inference pipeline:
 *
 *   1. duration.onnx: tokens → (m_p_exp, logs_p_exp, y_mask)
 *      Inputs:  tokens [1, seq_len] int64, lengths [1] int64, length_scale [1] float32
 *      Outputs: m_p_exp [1, 128, T_audio], logs_p_exp [1, 128, T_audio], y_mask [1, 1, T_audio]
 *      This single graph bakes in: enc_p, dp, attention matrix, generate_path, matmul expansion.
 *
 *   2. decode.onnx: (m_p_exp, logs_p_exp, y_mask, zp_noise, noise_scale) → waveform
 *      Inputs:  m_p_exp, logs_p_exp, y_mask (from step 1), zp_noise [1, 128, T_audio] float32, noise_scale [1] float32
 *      Outputs: waveform [1, 1, samples] float32
 *      This single graph bakes in: z_p sampling, flow.forward(reverse=True), dec.forward, max_len slicing.
 *
 * Compare to the PyTorch submodule pathway (4 native calls + ~200 lines of
 * Kotlin orchestration for attention/matmul/sampling). The ONNX version is
 * 2 native calls + ~20 lines of Kotlin.
 *
 * The crash-postmortem fields (lastInferenceStep, lastInferenceInputs) are
 * preserved and persisted to SharedPreferences, same as the PyTorch version.
 */
class InflectInference(private val context: Context) {

    companion object {
        private const val TAG = "InflectInference"

        /** Audio sample rate (matches `config.json` `sampling_rate`). */
        const val SAMPLE_RATE = 24000

        /** SharedPreferences file for crash-postmortem (shared with TTSModule). */
        private const val PREFS_NAME = "inflect_tts_crash_postmortem"
        private const val PREF_LAST_STEP = "lastInferenceStep"
        private const val PREF_LAST_INPUTS = "lastInferenceInputs"
    }

    /** SharedPreferences for crash post-mortem — survives process crashes. */
    private val crashPrefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    /** ONNX Runtime environment (singleton, thread-safe). */
    private val env: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }

    /** Duration session — enc_p + dp + attention + expand (baked into one graph). */
    @Volatile private var durationSession: OrtSession? = null

    /** Decode session — flow + dec (baked into one graph). */
    @Volatile private var decodeSession: OrtSession? = null

    @Volatile private var isLoaded = false

    /** True iff both ONNX sessions have been loaded. */
    fun isReady(): Boolean = isLoaded

    /**
     * The name of the inference step currently being executed (or the last
     * one attempted if a crash killed the process). Persisted to disk so
     * it survives process crashes.
     *
     * Values: "init", "synthesize_called", "coroutine_entered",
     * "preprocessing", "phoneme_encoding", "duration_session",
     * "decode_session", "post_processing", "save_wav", "play_audio",
     * "resolve_promise", "done".
     */
    @Volatile
    var lastInferenceStep: String = crashPrefs.getString(PREF_LAST_STEP, "init") ?: "init"
        set(value) {
            field = value
            crashPrefs.edit().putString(PREF_LAST_STEP, value).commit()
        }

    /** Inputs used for the last inference (for crash post-mortem). */
    @Volatile
    var lastInferenceInputs: String = crashPrefs.getString(PREF_LAST_INPUTS, "") ?: ""
        set(value) {
            field = value
            crashPrefs.edit().putString(PREF_LAST_INPUTS, value).commit()
        }

    /** Path of the on-disk model directory. */
    private val modelDir: File
        get() = File(context.filesDir, ModelDownloader.MODEL_DIR_NAME)

    /**
     * Load the two ONNX sessions. Throws on any failure with a message
     * that identifies WHICH file failed.
     *
     * Call on a background thread.
     */
    fun load() {
        if (isLoaded) return
        require(modelDir.exists()) {
            "Model directory ${modelDir.absolutePath} missing — call ModelDownloader first"
        }
        Log.i(TAG, "Loading ONNX sessions from ${modelDir.absolutePath}")

        val t0 = System.currentTimeMillis()

        // Configure session options: use all available threads for parallel ops.
        val sessionOptions = OrtSession.SessionOptions().apply {
            try {
                setCPUArenaAllocator(true)
                // XNNPACK EP for optimized CPU inference ( analogous to PyTorch Lite's XNNPACK)
                // Available in onnxruntime-android >= 1.16
                addXnnpack(mapOf("intra_op_num_threads" to Runtime.getRuntime().availableProcessors().toString()))
                Log.i(TAG, "XNNPACK EP enabled with ${Runtime.getRuntime().availableProcessors()} threads")
            } catch (t: Throwable) {
                Log.w(TAG, "XNNPACK EP setup failed (non-fatal, will use CPU EP): ${t.message}")
            }
        }

        durationSession = loadOne("duration.onnx", sessionOptions)
        decodeSession = loadOne("decode.onnx", sessionOptions)

        isLoaded = true
        Log.i(TAG, "Both ONNX sessions loaded in ${System.currentTimeMillis() - t0} ms")
    }

    /**
     * Load a single ONNX session, wrapping any exception with the filename
     * and file size so the failure is easy to diagnose.
     */
    private fun loadOne(name: String, options: OrtSession.SessionOptions): OrtSession {
        val file = File(modelDir, name)
        val sizeBytes = if (file.exists()) file.length() else -1L
        val sizeStr = if (sizeBytes >= 0) "${sizeBytes / 1024} KB" else "MISSING"
        Log.d(TAG, "Loading $name ($sizeStr) from ${modelDir.absolutePath}")
        return try {
            env.createSession(file.absolutePath, options)
        } catch (t: Throwable) {
            val msg = "Failed to load $name " +
                "(path=${file.absolutePath}, sizeOnDisk=$sizeStr): " +
                "${t.javaClass.simpleName}: ${t.message}"
            Log.e(TAG, msg, t)
            throw RuntimeException(msg, t)
        }
    }

    /** Release native session memory. */
    fun release() {
        durationSession?.close(); durationSession = null
        decodeSession?.close();   decodeSession = null
        isLoaded = false
    }

    /**
     * Run the full inference pipeline.
     *
     * Two ONNX calls:
     *   1. duration.onnx: tokens + lengths + length_scale → m_p_exp, logs_p_exp, y_mask
     *   2. decode.onnx: m_p_exp + logs_p_exp + y_mask + zp_noise + noise_scale → waveform
     *
     * The latent noise (zp_noise) is sampled in Kotlin with java.util.Random
     * (seeded) — matching the Python wrapper's `np.random.default_rng(seed)`.
     *
     * @param phonemes token IDs (already interspersed with pad for add_blank=true).
     *                 Shape: [seq_len] (will be reshaped to [1, seq_len] for ONNX).
     * @param noiseScale variation in latent sampling (0..1+). Maps to `noise_scale` input.
     * @param lengthScale inverse of speed (1.0 = normal, 2.0 = 2x slower). Maps to `length_scale` input (as 1/speed).
     * @param seed RNG seed for latent noise sampling.
     * @return float PCM samples at 24000 Hz, range approximately [-1, 1].
     */
    fun infer(
        phonemes: IntArray,
        noiseScale: Float = 1.0f,
        lengthScale: Float = 1.0f,
        noiseScaleW: Float = 1.0f,  // unused in ONNX version (baked into duration.onnx)
        seed: Int = 0,
    ): FloatArray {
        check(isLoaded) { "InflectInference not loaded — call load() first" }

        lastInferenceInputs = "phonemes.size=${phonemes.size}, noiseScale=$noiseScale, " +
            "lengthScale=$lengthScale, seed=$seed"
        Log.i(TAG, "infer() START — $lastInferenceInputs")

        val seqLen = phonemes.size.toLong()
        // -------- Step 1: duration.onnx --------
        lastInferenceStep = "duration_session"
        Log.i(TAG, "step=$lastInferenceStep: running duration.onnx (tokens len=$seqLen, length_scale=$lengthScale)")

        // Build inputs for duration.onnx:
        //   tokens: [1, seq_len] int64
        //   lengths: [1] int64
        //   length_scale: [1] float32
        //
        // OnnxTensor.createTensor(env, LongBuffer, long[]) creates an int64 tensor.
        // OnnxTensor.createTensor(env, FloatBuffer, long[]) creates a float32 tensor.
        // (There's no overload for raw arrays — must wrap in a Buffer.)
        val tokensLongArr = LongArray(phonemes.size) { phonemes[it].toLong() }
        val tokensTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(tokensLongArr), longArrayOf(1, seqLen))
        val lengthsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(seqLen)), longArrayOf(1))
        val lengthScaleArr = floatArrayOf(lengthScale)
        val lengthScaleTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(lengthScaleArr), longArrayOf(1))

        val durationInputs = mapOf(
            "tokens" to tokensTensor,
            "lengths" to lengthsTensor,
            "length_scale" to lengthScaleTensor,
        )

        val durationOutputs = try {
            durationSession!!.run(durationInputs)
        } catch (t: Throwable) {
            tokensTensor.close()
            lengthsTensor.close()
            lengthScaleTensor.close()
            val msg = "duration.onnx run failed: ${t.javaClass.simpleName}: ${t.message}"
            Log.e(TAG, msg, t)
            throw RuntimeException(msg, t)
        }

        // Outputs: m_p_exp [1, 128, T_audio], logs_p_exp [1, 128, T_audio], y_mask [1, 1, T_audio]
        // Result.get(int) returns OnnxValue — cast to OnnxTensor.
        val mPExpTensor = durationOutputs.get(0) as OnnxTensor
        val logsPExpTensor = durationOutputs.get(1) as OnnxTensor
        val yMaskTensor = durationOutputs.get(2) as OnnxTensor

        val mPExpShape = mPExpTensor.info.shape
        val logsPExpShape = logsPExpTensor.info.shape
        val yMaskShape = yMaskTensor.info.shape
        Log.i(TAG, "step=$lastInferenceStep: OK — " +
            "m_p_exp=${mPExpShape.contentToString()}, " +
            "logs_p_exp=${logsPExpShape.contentToString()}, " +
            "y_mask=${yMaskShape.contentToString()}")

        // -------- Sample latent noise (zp_noise) --------
        // Python: rng = np.random.default_rng(seed); latent_noise = rng.standard_normal(m_p_exp.shape, dtype=np.float32)
        // We use java.util.Random with the seed — note: NOT identical to numpy's RNG,
        // but produces statistically equivalent Gaussian noise. The model is robust to
        // the exact noise values (they just add variation).
        val tAudio = mPExpShape[2].toInt()
        val interChannels = mPExpShape[1].toInt()
        val noiseSize = interChannels * tAudio
        val rand = java.util.Random(seed.toLong())
        val zpNoise = FloatArray(noiseSize)
        for (i in zpNoise.indices) {
            zpNoise[i] = rand.nextGaussian().toFloat()
        }
        Log.i(TAG, "step=$lastInferenceStep: sampled zp_noise size=$noiseSize (seed=$seed)")

        // -------- Step 2: decode.onnx --------
        lastInferenceStep = "decode_session"
        val zpNoiseTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(zpNoise), longArrayOf(1, interChannels.toLong(), tAudio.toLong()))
        val noiseScaleArr = floatArrayOf(noiseScale)
        val noiseScaleTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(noiseScaleArr), longArrayOf(1))

        val decodeInputs = mapOf(
            "m_p_exp" to mPExpTensor,
            "logs_p_exp" to logsPExpTensor,
            "y_mask" to yMaskTensor,
            "zp_noise" to zpNoiseTensor,
            "noise_scale" to noiseScaleTensor,
        )

        Log.i(TAG, "step=$lastInferenceStep: running decode.onnx " +
            "(m_p_exp=${mPExpShape.contentToString()}, noise_scale=$noiseScale)")

        val decodeOutputs = try {
            decodeSession!!.run(decodeInputs)
        } catch (t: Throwable) {
            tokensTensor.close()
            lengthsTensor.close()
            lengthScaleTensor.close()
            mPExpTensor.close()
            logsPExpTensor.close()
            yMaskTensor.close()
            zpNoiseTensor.close()
            noiseScaleTensor.close()
            durationOutputs.close()
            val msg = "decode.onnx run failed: ${t.javaClass.simpleName}: ${t.message}"
            Log.e(TAG, msg, t)
            throw RuntimeException(msg, t)
        }

        val waveformTensor = decodeOutputs.get(0) as OnnxTensor
        val waveformShape = waveformTensor.info.shape
        Log.i(TAG, "step=$lastInferenceStep: OK — waveform=${waveformShape.contentToString()}")

        // Extract the waveform as a float array.
        // waveform shape is [1, 1, samples] — getValue() returns nested arrays.
        val waveformObj = waveformTensor.value
        val result = FloatArray(waveformShape[2].toInt())
        when (waveformObj) {
            is FloatArray -> {
                // Flat array — direct copy
                System.arraycopy(waveformObj, 0, result, 0, min(result.size, waveformObj.size))
            }
            is Array<*> -> {
                // Nested [1, 1, samples] — unwrap two levels
                @Suppress("UNCHECKED_CAST")
                val outer = waveformObj[0] as Array<*>
                @Suppress("UNCHECKED_CAST")
                val inner = outer[0] as FloatArray
                System.arraycopy(inner, 0, result, 0, min(result.size, inner.size))
            }
            else -> {
                throw RuntimeException("Unexpected waveform type: ${waveformObj?.javaClass?.name}")
            }
        }

        // -------- Cleanup --------
        tokensTensor.close()
        lengthsTensor.close()
        lengthScaleTensor.close()
        mPExpTensor.close()
        logsPExpTensor.close()
        yMaskTensor.close()
        zpNoiseTensor.close()
        noiseScaleTensor.close()
        durationOutputs.close()
        decodeOutputs.close()

        lastInferenceStep = "done"
        Log.i(TAG, "infer() DONE — returning ${result.size} samples")
        return result
    }

    /**
     * Return a plain-map snapshot of the inference state for debugging.
     */
    fun getDiagnostics(): Map<String, Any?> {
        val dir = modelDir
        return mapOf(
            "isLoaded" to isLoaded,
            "isReady" to isReady(),
            "modelDir" to dir.absolutePath,
            "modelDirExists" to dir.exists(),
            "backend" to "onnxruntime",
            "sessionsLoaded" to mapOf(
                "duration" to (durationSession != null),
                "decode" to (decodeSession != null),
            ),
            "files" to ModelDownloader.MODEL_FILES.associate { mf ->
                val f = File(dir, mf.name)
                val actual = if (f.exists()) f.length() else -1L
                mf.name to mapOf(
                    "exists" to f.exists(),
                    "expectedSize" to mf.expectedSize,
                    "actualSize" to actual,
                    "sizeMatches" to (actual == mf.expectedSize),
                )
            },
        )
    }

    /**
     * Same as [getDiagnostics] but returns a React Native WritableMap.
     */
    fun getDiagnosticsAsMap(): WritableMap {
        val m = Arguments.createMap()
        m.putBoolean("isLoaded", isLoaded)
        m.putBoolean("isReady", isReady())
        m.putString("modelDir", modelDir.absolutePath)
        m.putBoolean("modelDirExists", modelDir.exists())
        m.putString("backend", "onnxruntime")
        m.putString("lastInferenceStep", lastInferenceStep)
        m.putString("lastInferenceInputs", lastInferenceInputs)
        val sessions = Arguments.createMap()
        sessions.putBoolean("duration", durationSession != null)
        sessions.putBoolean("decode", decodeSession != null)
        m.putMap("sessionsLoaded", sessions)
        val files = Arguments.createArray()
        for (mf in ModelDownloader.MODEL_FILES) {
            val f = File(modelDir, mf.name)
            val actual = if (f.exists()) f.length() else -1L
            files.pushMap(Arguments.createMap().apply {
                putString("name", mf.name)
                putBoolean("exists", f.exists())
                putDouble("expectedSize", mf.expectedSize.toDouble())
                putDouble("actualSize", actual.toDouble())
                putBoolean("sizeMatches", actual == mf.expectedSize)
            })
        }
        m.putArray("files", files)
        return m
    }
}

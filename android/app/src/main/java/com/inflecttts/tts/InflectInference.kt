package com.inflecttts.tts

import android.content.Context
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * InflectInference
 *
 * Loads the five TorchScript submodule files produced by
 * `nikitastaf1996/Inflect-Nano-v2-TorchScript` and reconstructs
 * the `SynthesizerTrn.infer()` pipeline from `runtime/models.py`.
 *
 * Pipeline (matching the HF README "Submodule TorchScript pathway"):
 *
 *   1. enc_p.forward(tokens, token_lengths) -> (x, m_p, logs_p, x_mask)
 *   2. dp.forward(x, x_mask, reverse=True, noise_scale=noise_scale_w) -> logw
 *   3. w = exp(logw) * x_mask * length_scale
 *      w_ceil = ceil(w)
 *      y_lengths = clamp(sum(w_ceil), 1).long()
 *   4. y_mask = sequence_mask(y_lengths, max(y_lengths))
 *      attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
 *      attn = generate_path(w_ceil, attn_mask)
 *   5. z_p = m_p + randn_like(m_p) * exp(logs_p) * noise_scale
 *   6. z = flow.forward(z_p, y_mask, reverse=True)
 *   7. o = dec.forward(z * y_mask)
 *
 * Inputs:
 *   - tokens: Long tensor [1, seq_len], padded to MAX_SEQ_LEN (256)
 *   - lengths: Long tensor [1]
 *
 * Output:
 *   - FloatArray of PCM samples at 24000 Hz mono.
 *
 * If the model fails to load or infer (e.g. submodule interface
 * differs from what the README documents), callers should fall back
 * to the legacy waveform synthesis in [TTSModule].
 */
class InflectInference(private val context: Context) {

    companion object {
        private const val TAG = "InflectInference"

        /** Pad/truncate input token sequences to this length. */
        const val MAX_SEQ_LEN = 256

        /** Audio sample rate (matches `config.json` `sampling_rate`). */
        const val SAMPLE_RATE = 24000

        /** Model inter_channels (matches `config.json` `inter_channels`). */
        const val INTER_CHANNELS = 128
    }

    /** Path of the on-disk model directory. */
    private val modelDir: File
        get() = File(context.filesDir, ModelDownloader.MODEL_DIR_NAME)

    // The five scripted submodules, loaded lazily.
    @Volatile private var encP: Module? = null
    @Volatile private var dec: Module? = null
    @Volatile private var encQ: Module? = null
    @Volatile private var flow: Module? = null
    @Volatile private var dp: Module? = null
    @Volatile private var isLoaded = false

    /** True iff all five submodule Modules have been loaded. */
    fun isReady(): Boolean = isLoaded

    /**
     * Load all five `.pt` submodules. Throws on any failure with a
     * message that identifies WHICH file failed — so the user can
     * tell whether (e.g.) `inflect_dec.pt` is corrupted vs.
     * `inflect_enc_p.pt` having a TorchScript incompat.
     *
     * Call on a background thread.
     */
    fun load() {
        if (isLoaded) return
        require(modelDir.exists()) {
            "Model directory ${modelDir.absolutePath} missing — call ModelDownloader first"
        }
        Log.i(TAG, "Loading scripted submodules from ${modelDir.absolutePath}")

        val t0 = System.currentTimeMillis()

        // Per-module load with descriptive errors. Each Module.load() can
        // throw RuntimeException (file not found, TorchScript parse error,
        // unsupported op, …). We wrap each one so the stack trace shows
        // exactly which file was being loaded when the failure happened.
        encP = loadOne("inflect_enc_p.pt")
        dec  = loadOne("inflect_dec.pt")
        encQ = loadOne("inflect_enc_q.pt")
        flow = loadOne("inflect_flow.pt")
        dp   = loadOne("inflect_dp.pt")

        isLoaded = true
        Log.i(TAG, "All submodules loaded in ${System.currentTimeMillis() - t0} ms")
    }

    /**
     * Load a single `.pt` file, wrapping any exception with the filename
     * and file size so the failure is easy to diagnose.
     */
    private fun loadOne(name: String): Module {
        val file = File(modelDir, name)
        val sizeBytes = if (file.exists()) file.length() else -1L
        val sizeStr = if (sizeBytes >= 0) "${sizeBytes / 1024} KB" else "MISSING"
        Log.d(TAG, "Loading $name ($sizeStr) from ${modelDir.absolutePath}")
        return try {
            Module.load(file.absolutePath)
        } catch (t: Throwable) {
            // Re-throw with a richer message that includes the filename
            // and on-disk size — the original exception is preserved as
            // the cause so the full stack trace is still available.
            val msg = "Failed to load $name " +
                "(path=${file.absolutePath}, sizeOnDisk=$sizeStr): " +
                "${t.javaClass.simpleName}: ${t.message}"
            Log.e(TAG, msg, t)
            throw RuntimeException(msg, t)
        }
    }

    /** Release native module memory. */
    fun release() {
        encP?.destroy(); encP = null
        dec?.destroy();  dec  = null
        encQ?.destroy(); encQ = null
        flow?.destroy(); flow = null
        dp?.destroy();   dp   = null
        isLoaded = false
    }

    /**
     * Return a plain-map (Kotlin Map<String, Any?>) snapshot of the
     * inference state for debugging. Used by [TTSModule.getDiagnostics].
     */
    fun getDiagnostics(): Map<String, Any?> {
        val dir = modelDir
        val filesStatus = ModelDownloader.MODEL_FILES.associate { mf ->
            val f = File(dir, mf.name)
            val actual = if (f.exists()) f.length() else -1L
            mf.name to mapOf(
                "exists" to f.exists(),
                "expectedSize" to mf.expectedSize,
                "actualSize" to actual,
                "sizeMatches" to (actual == mf.expectedSize),
            )
        }
        return mapOf(
            "isLoaded" to isLoaded,
            "isReady" to isReady(),
            "modelDir" to dir.absolutePath,
            "modelDirExists" to dir.exists(),
            "modulesLoaded" to mapOf(
                "encP" to (encP != null),
                "dec"  to (dec  != null),
                "encQ" to (encQ != null),
                "flow" to (flow != null),
                "dp"   to (dp   != null),
            ),
            "files" to filesStatus,
        )
    }

    /**
     * Same as [getDiagnostics] but returns a React Native WritableMap,
     * for direct inclusion in a Promise resolve payload.
     */
    fun getDiagnosticsAsMap(): com.facebook.react.bridge.WritableMap {
        val m = com.facebook.react.bridge.Arguments.createMap()
        m.putBoolean("isLoaded", isLoaded)
        m.putBoolean("isReady", isReady())
        m.putString("modelDir", modelDir.absolutePath)
        m.putBoolean("modelDirExists", modelDir.exists())
        val modules = com.facebook.react.bridge.Arguments.createMap()
        modules.putBoolean("encP", encP != null)
        modules.putBoolean("dec",  dec  != null)
        modules.putBoolean("encQ", encQ != null)
        modules.putBoolean("flow", flow != null)
        modules.putBoolean("dp",   dp   != null)
        m.putMap("modulesLoaded", modules)
        val files = com.facebook.react.bridge.Arguments.createArray()
        for (mf in ModelDownloader.MODEL_FILES) {
            val f = File(modelDir, mf.name)
            val actual = if (f.exists()) f.length() else -1L
            files.pushMap(com.facebook.react.bridge.Arguments.createMap().apply {
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

    /**
     * Run the full inference pipeline.
     *
     * @param phonemes token IDs in the model's symbol table (1-indexed phonemes;
     *                  0 is reserved for blank/pad).
     * @param noiseScale variation in latent sampling (0..1+).
     * @param lengthScale inverse of speed (1.0 = normal, 2.0 = 2x slower, 0.5 = 2x faster).
     * @param noiseScaleW variation in duration sampling (0..1+).
     * @return float PCM samples at 24000 Hz, range approximately [-1, 1].
     */
    fun infer(
        phonemes: IntArray,
        noiseScale: Float = 1.0f,
        lengthScale: Float = 1.0f,
        noiseScaleW: Float = 1.0f,
    ): FloatArray {
        check(isLoaded) { "InflectInference not loaded — call load() first" }

        // -------- 0. Prepare token tensor [1, MAX_SEQ_LEN] (int64) --------
        // PyTorch Android 2.1.0 exposes `Tensor.fromBlob(long[], long[])` (no `fromBlobLong`).
        val seq = IntArray(MAX_SEQ_LEN) { 0 }
        val realLen = min(phonemes.size, MAX_SEQ_LEN)
        System.arraycopy(phonemes, 0, seq, 0, realLen)
        val seqLong = LongArray(MAX_SEQ_LEN) { i -> seq[i].toLong() }
        val tokensTensor = Tensor.fromBlob(seqLong, longArrayOf(1, MAX_SEQ_LEN.toLong()))
        val lengthsTensor = Tensor.fromBlob(longArrayOf(realLen.toLong()), longArrayOf(1))

        // -------- 1. enc_p.forward(tokens, lengths) -> (x, m_p, logs_p, x_mask) --------
        val encOut = encP!!.forward(IValue.from(tokensTensor), IValue.from(lengthsTensor)).toTuple()
        val xTensor = encOut[0].toTensor()       // [1, hidden, T_text]
        val mPTensor = encOut[1].toTensor()      // [1, inter, T_text]
        val logsPTensor = encOut[2].toTensor()   // [1, inter, T_text]
        val xMaskTensor = encOut[3].toTensor()   // [1, 1, T_text]
        Log.d(TAG, "enc_p: x=${xTensor.shape().contentToString()}, m_p=${mPTensor.shape().contentToString()}, " +
            "logs_p=${logsPTensor.shape().contentToString()}, x_mask=${xMaskTensor.shape().contentToString()}")

        // -------- 2. dp.forward(x, x_mask[, reverse, noise_scale]) --------
        // With `use_sdp=false` (see config.json), `dp` is a deterministic
        // `DurationPredictor` whose scripted forward signature is
        // `forward(x, x_mask)` and returns `logw` directly. With
        // `use_sdp=true` it would be a `StochasticDurationPredictor` whose
        // forward is `forward(x, x_mask, reverse, noise_scale)`.
        // We try the deterministic signature first (matches our config),
        // then fall back to the stochastic one.
        val logw: Tensor = try {
            dp!!.forward(
                IValue.from(xTensor),
                IValue.from(xMaskTensor),
            ).toTensor()
        } catch (t1: Throwable) {
            Log.w(TAG, "dp.forward(x, x_mask) failed (${t1.message}); trying stochastic signature")
            dp!!.forward(
                IValue.from(xTensor),
                IValue.from(xMaskTensor),
                IValue.from(true),
                IValue.from(noiseScaleW.toDouble()),
            ).toTensor()
        }
        Log.d(TAG, "dp: logw=${logw.shape().contentToString()}")

        // -------- 3. Compute durations and y_lengths --------
        // w = exp(logw) * x_mask * length_scale
        // w_ceil = ceil(w)
        // y_lengths = clamp(sum(w_ceil), 1)
        val logwArr = logw.getDataAsFloatArray()
        val xMaskArr = xMaskTensor.getDataAsFloatArray()
        val tText = (xMaskTensor.shape()[2]).toInt()
        val wCeil = IntArray(tText)
        var yLenSum = 0L
        for (i in 0 until tText) {
            val maskVal = xMaskArr[i]
            val w = if (maskVal > 0f) {
                exp(logwArr[i].toDouble()).toFloat() * maskVal * lengthScale
            } else 0f
            val c = ceil(w.toDouble()).toInt().coerceAtLeast(0)
            wCeil[i] = c
            yLenSum += c
        }
        val yLengths = max(yLenSum, 1L).toInt()
        Log.d(TAG, "y_lengths=$yLengths (sum of $tText durations)")

        // -------- 4. Build y_mask, attn_mask, attn --------
        // y_mask = sequence_mask(y_lengths, y_lengths).unsqueeze(1)  -> [1, 1, y_len]
        val yMask = FloatArray(yLengths) { 1f }  // [y_len]
        // attn_mask[i, j] = x_mask[i] * y_mask[j]  (broadcasting)
        // attn[t_text, t_y] = 1 if (t_text valid) and (t_y < y_lengths)

        // generate_path: produces alignment [t_y, t_text] where attn[t_y, t_text] = 1
        // if t_y falls within the duration span of phoneme t_text.
        val attn = Array(yLengths) { FloatArray(tText) }  // [t_y, t_text]
        var pos = 0
        for (t in 0 until tText) {
            val dur = wCeil[t]
            for (k in 0 until dur) {
                if (pos < yLengths) {
                    attn[pos][t] = 1f
                    pos++
                }
            }
        }

        // -------- 5. Expand m_p, logs_p along time via attn --------
        // m_p_expanded[1, inter, t_y] = sum_t(attn[t_y, t] * m_p[1, inter, t])
        // We do this in Kotlin because matmul via Tensor is awkward here.
        val mPArr = mPTensor.getDataAsFloatArray()   // [1, inter, t_text] flattened row-major
        val logsPArr = logsPTensor.getDataAsFloatArray()
        val inter = (mPTensor.shape()[1]).toInt()
        val mPExp = FloatArray(inter * yLengths)
        val logsPExp = FloatArray(inter * yLengths)
        for (c in 0 until inter) {
            for (y in 0 until yLengths) {
                var acc = 0f
                var accL = 0f
                for (t in 0 until tText) {
                    if (attn[y][t] > 0f) {
                        acc += mPArr[c * tText + t]
                        accL += logsPArr[c * tText + t]
                    }
                }
                mPExp[c * yLengths + y] = acc
                logsPExp[c * yLengths + y] = accL
            }
        }

        // z_p = m_p + randn_like(m_p) * exp(logs_p) * noise_scale
        // Use java.util.Random for nextGaussian() — Kotlin's Random doesn't
        // expose nextGaussian() in older stdlib versions and may need
        // ExperimentalStdlibApi opt-in.
        val rand = java.util.Random(System.nanoTime())
        val zP = FloatArray(inter * yLengths)
        for (i in zP.indices) {
            val r = rand.nextGaussian().toFloat()
            zP[i] = mPExp[i] + r * exp(logsPExp[i].toDouble()).toFloat() * noiseScale
        }

        // -------- 6. flow.forward(z_p, y_mask, reverse=True) -> z --------
        val zPTensor = Tensor.fromBlob(zP, longArrayOf(1, inter.toLong(), yLengths.toLong()))
        val yMaskT = Tensor.fromBlob(
            FloatArray(yLengths) { 1f },
            longArrayOf(1, 1, yLengths.toLong()),
        )
        val zTensor = flow!!.forward(
            IValue.from(zPTensor),
            IValue.from(yMaskT),
            IValue.from(true),
        ).toTensor()
        Log.d(TAG, "flow: z=${zTensor.shape().contentToString()}")

        // z * y_mask (y_mask already 1.0 in valid region)
        val zArr = zTensor.getDataAsFloatArray()

        // -------- 7. dec.forward(z * y_mask) -> waveform --------
        val outTensor = dec!!.forward(IValue.from(zTensor)).toTensor()
        Log.d(TAG, "dec: out=${outTensor.shape().contentToString()}")

        return outTensor.getDataAsFloatArray()
    }
}

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
 * `nikitastaf1996/Inflect-Nano-v2-Mobile` and reconstructs
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

        /** SharedPreferences file for crash-postmortem (shared with TTSModule). */
        private const val PREFS_NAME = "inflect_tts_crash_postmortem"
        private const val PREF_LAST_STEP = "lastInferenceStep"
        private const val PREF_LAST_INPUTS = "lastInferenceInputs"
    }

    /** SharedPreferences for crash post-mortem — survives process crashes. */
    private val crashPrefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    /** Path of the on-disk model directory. */
    private val modelDir: File
        get() = File(context.filesDir, ModelDownloader.MODEL_DIR_NAME)

    // The four scripted submodules used in inference, loaded lazily.
    // NOTE: enc_q (PosteriorEncoder) is NOT loaded — it's training-only.
    // See runtime/models.py SynthesizerTrn.infer() (lines 559-583):
    //   infer() calls enc_p, dp, flow, dec — never enc_q.
    // enc_q is only used in forward() (training) and voice_conversion()
    // (which is disabled: inference_only=True raises RuntimeError).
    // The HF README confirms: "infer() uses enc_p, not enc_q".
    // Skipping enc_q saves ~1.2 MB download + ~4 MB native memory.
    @Volatile private var encP: Module? = null
    @Volatile private var dec: Module? = null
    @Volatile private var flow: Module? = null
    @Volatile private var dp: Module? = null
    @Volatile private var isLoaded = false

    /** True iff all four inference submodules have been loaded. */
    fun isReady(): Boolean = isLoaded

    /**
     * The name of the inference step currently being executed (or the last
     * one attempted if a crash killed the process).
     *
     * IMPORTANT: This field is ALSO persisted to SharedPreferences on every
     * write, because @Volatile only guarantees cross-thread visibility WITHIN
     * the same process — it does NOT survive a process crash (SIGSEGV).
     * When the app crashes and relaunches, the field is re-loaded from disk
     * so getDiagnostics() can report which step crashed.
     *
     * Values: "init", "synthesize_called", "coroutine_entered",
     * "preprocessing", "phoneme_encoding", "prepare_tokens",
     * "enc_p_forward", "dp_forward", "compute_durations",
     * "build_attention", "expand_m_p", "sample_z_p", "flow_forward",
     * "dec_forward", "post_processing", "save_wav", "play_audio",
     * "resolve_promise", "done".
     *
     * Setter is public so [TTSModule] can write it from its own
     * orchestration steps (before/after infer()). Every write is
     * synchronously flushed to disk via commit().
     */
    @Volatile
    var lastInferenceStep: String = crashPrefs.getString(PREF_LAST_STEP, "init") ?: "init"
        set(value) {
            field = value
            crashPrefs.edit().putString(PREF_LAST_STEP, value).commit()
        }

    /**
     * Inputs used for the last inference (for crash post-mortem).
     * Persisted to SharedPreferences on every write.
     */
    @Volatile
    var lastInferenceInputs: String = crashPrefs.getString(PREF_LAST_INPUTS, "") ?: ""
        set(value) {
            field = value
            crashPrefs.edit().putString(PREF_LAST_INPUTS, value).commit()
        }

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

        // ---- Set PyTorch thread count ----
        // PyTorch Android defaults to using all available cores, but on
        // big.LITTLE SoCs the OS scheduler may pin inference threads to
        // LITTLE cores for thermal/battery reasons. Setting an explicit
        // count tells PyTorch's backend (XNNPACK) to use that many threads
        // for parallel ops (matmul, conv).
        //
        // We use availableProcessors() (= core count). On modern phones
        // (e.g. 1+3+4 big.LITTLE), this is typically 8, and PyTorch will
        // spread work across all cores. On older 1+3 designs it's 4.
        //
        // This is a hint, not a guarantee — XNNPACK may use fewer threads
        // for small tensors. But it ensures we don't underutilize the CPU.
        val numThreads = Runtime.getRuntime().availableProcessors()
        try {
            // IMPORTANT: use LitePyTorchAndroid (not PyTorchAndroid) because
            // we're on pytorch_android_lite. PyTorchAndroid would try to load
            // libpytorch_jni.so (full runtime) which doesn't exist in the
            // lite AAR — only libpytorch_jni_lite.so exists.
            org.pytorch.LitePyTorchAndroid.setNumThreads(numThreads)
            Log.i(TAG, "Set PyTorch thread count to $numThreads (availableProcessors)")
        } catch (t: Throwable) {
            Log.w(TAG, "setNumThreads($numThreads) failed (non-fatal): ${t.message}")
        }

        val t0 = System.currentTimeMillis()

        // Per-module load with descriptive errors. Each Module.load() can
        // throw RuntimeException (file not found, TorchScript parse error,
        // unsupported op, …). We wrap each one so the stack trace shows
        // exactly which file was being loaded when the failure happened.
        encP = loadOne("inflect_enc_p.ptl")
        dec  = loadOne("inflect_dec.ptl")
        // enc_q skipped — training-only, not used in infer(). See class comment.
        flow = loadOne("inflect_flow.ptl")
        dp   = loadOne("inflect_dp.ptl")

        isLoaded = true
        Log.i(TAG, "All submodules loaded in ${System.currentTimeMillis() - t0} ms")
    }

    /**
     * Load a single `.ptl` (lite interpreter) file, wrapping any exception
     * with the filename and file size so the failure is easy to diagnose.
     *
     * IMPORTANT: uses LiteModuleLoader.load() (not Module.load()) because
     * we're on pytorch_android_lite. Module.load() would use the full
     * NativePeer which tries to load libpytorch_jni.so — that .so doesn't
     * exist in the lite AAR (only libpytorch_jni_lite.so exists), causing
     * "SoLoader dsonotfound couldn't find dso to load libpytorch_jni.so".
     */
    private fun loadOne(name: String): Module {
        val file = File(modelDir, name)
        val sizeBytes = if (file.exists()) file.length() else -1L
        val sizeStr = if (sizeBytes >= 0) "${sizeBytes / 1024} KB" else "MISSING"
        Log.d(TAG, "Loading $name ($sizeStr) from ${modelDir.absolutePath}")
        return try {
            org.pytorch.LiteModuleLoader.load(file.absolutePath)
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
                "encQ" to "(skipped — training-only)",
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
        // Crash post-mortem fields — these survive a process crash because
        // they're @Volatile and written before each native call.
        m.putString("lastInferenceStep", lastInferenceStep)
        m.putString("lastInferenceInputs", lastInferenceInputs)
        val modules = com.facebook.react.bridge.Arguments.createMap()
        modules.putBoolean("encP", encP != null)
        modules.putBoolean("dec",  dec  != null)
        modules.putString("encQ", "(skipped — training-only, not used in infer())")
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
     * Each native call is preceded by setting [lastInferenceStep] — a
     * @Volatile field that survives a process crash. So if a native
     * SIGSEGV kills the app during (e.g.) `flow_forward`, the next
     * launch's getDiagnostics() will report `lastInferenceStep = "flow_forward"`,
     * telling us exactly which call crashed.
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

        // Record inputs for crash post-mortem.
        lastInferenceInputs = "phonemes.size=${phonemes.size}, " +
            "noiseScale=$noiseScale, lengthScale=$lengthScale, noiseScaleW=$noiseScaleW"
        Log.i(TAG, "infer() START — $lastInferenceInputs")

        // -------- 0. Prepare token tensor [1, MAX_SEQ_LEN] (int64) --------
        lastInferenceStep = "prepare_tokens"
        Log.i(TAG, "step=$lastInferenceStep: building tokens tensor [1, $MAX_SEQ_LEN]")
        val seq = IntArray(MAX_SEQ_LEN) { 0 }
        val realLen = min(phonemes.size, MAX_SEQ_LEN)
        System.arraycopy(phonemes, 0, seq, 0, realLen)
        val seqLong = LongArray(MAX_SEQ_LEN) { i -> seq[i].toLong() }
        val tokensTensor = Tensor.fromBlob(seqLong, longArrayOf(1, MAX_SEQ_LEN.toLong()))
        val lengthsTensor = Tensor.fromBlob(longArrayOf(realLen.toLong()), longArrayOf(1))
        Log.i(TAG, "step=$lastInferenceStep: tokens shape=${tokensTensor.shape().contentToString()}, " +
            "lengths shape=${lengthsTensor.shape().contentToString()}, realLen=$realLen")

        // -------- 1. enc_p.forward(tokens, lengths) -> (x, m_p, logs_p, x_mask) --------
        lastInferenceStep = "enc_p_forward"
        Log.i(TAG, "step=$lastInferenceStep: calling encP.forward(tokens, lengths)")
        val encOut = encP!!.forward(IValue.from(tokensTensor), IValue.from(lengthsTensor)).toTuple()
        val xTensor = encOut[0].toTensor()       // [1, hidden, T_text]
        val mPTensor = encOut[1].toTensor()      // [1, inter, T_text]
        val logsPTensor = encOut[2].toTensor()   // [1, inter, T_text]
        val xMaskTensor = encOut[3].toTensor()   // [1, 1, T_text]
        Log.i(TAG, "step=$lastInferenceStep: OK — " +
            "x=${xTensor.shape().contentToString()}, " +
            "m_p=${mPTensor.shape().contentToString()}, " +
            "logs_p=${logsPTensor.shape().contentToString()}, " +
            "x_mask=${xMaskTensor.shape().contentToString()}")

        // -------- 2. dp.forward(x, x_mask) -> logw --------
        // CRITICAL: config.json has `use_sdp: false`, which means `dp` is
        // a DETERMINISTIC DurationPredictor (NOT a StochasticDurationPredictor).
        // Its scripted forward signature is:
        //   dp.forward(x, x_mask) -> logw        (2 args, deterministic)
        //
        // The HF README's reconstruction step 2 mentions a 4-arg form
        //   dp.forward(x, x_mask, reverse=True, noise_scale=noise_scale_w)
        // but that only applies when use_sdp=true (StochasticDurationPredictor).
        // For our config (use_sdp=false), calling dp with 4 args triggers a
        // native SIGSEGV — TorchScript's native dispatch crashes before the
        // arg-count check can throw a clean Java exception.
        //
        // v2.2.1 tried the 4-arg form first and crashed at dp_forward.
        // v2.3.1 uses the 2-arg form directly (matches use_sdp=false).
        // The noiseScaleW parameter is simply ignored — deterministic
        // DurationPredictor doesn't use noise.
        lastInferenceStep = "dp_forward"
        Log.i(TAG, "step=$lastInferenceStep: calling dp.forward(x=${xTensor.shape().contentToString()}, " +
            "x_mask=${xMaskTensor.shape().contentToString()}) [2-arg deterministic, use_sdp=false]")
        val logw: Tensor = dp!!.forward(
            IValue.from(xTensor),
            IValue.from(xMaskTensor),
        ).toTensor()
        Log.i(TAG, "step=$lastInferenceStep: OK — logw=${logw.shape().contentToString()}")

        // -------- 3. Compute durations and y_lengths --------
        // Pure Kotlin — no native calls, can't crash.
        lastInferenceStep = "compute_durations"
        Log.i(TAG, "step=$lastInferenceStep: computing durations from logw")
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
        Log.i(TAG, "step=$lastInferenceStep: OK — y_lengths=$yLengths " +
            "(sum of $tText durations, lengthScale=$lengthScale)")

        // -------- 4. Build attention matrix --------
        // Pure Kotlin.
        lastInferenceStep = "build_attention"
        Log.i(TAG, "step=$lastInferenceStep: building attention matrix [$yLengths x $tText]")
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
        Log.i(TAG, "step=$lastInferenceStep: OK")

        // -------- 5. Expand m_p, logs_p along time via attn --------
        // Pure Kotlin.
        lastInferenceStep = "expand_m_p"
        Log.i(TAG, "step=$lastInferenceStep: expanding m_p and logs_p via attention")
        val mPArr = mPTensor.getDataAsFloatArray()
        val logsPArr = logsPTensor.getDataAsFloatArray()
        val inter = (mPTensor.shape()[1]).toInt()
        Log.i(TAG, "step=$lastInferenceStep: inter=$inter, yLengths=$yLengths, " +
            "mPExp size=${inter * yLengths}")
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
        Log.i(TAG, "step=$lastInferenceStep: OK")

        // -------- 5b. Sample z_p = m_p + randn * exp(logs_p) * noise_scale --------
        // Pure Kotlin.
        lastInferenceStep = "sample_z_p"
        Log.i(TAG, "step=$lastInferenceStep: sampling z_p (Gaussian)")
        val rand = java.util.Random(System.nanoTime())
        val zP = FloatArray(inter * yLengths)
        for (i in zP.indices) {
            val r = rand.nextGaussian().toFloat()
            zP[i] = mPExp[i] + r * exp(logsPExp[i].toDouble()).toFloat() * noiseScale
        }
        Log.i(TAG, "step=$lastInferenceStep: OK — zP size=${zP.size}")

        // -------- 6. flow.forward(z_p, y_mask, g=None, reverse=True) -> z --------
        // From runtime/models.py SynthesizerTrn.infer() line 581:
        //   z = self.flow(z_p, y_mask, g=g, reverse=True)
        // ResidualCouplingBlock.forward signature (line 215):
        //   forward(self, x, x_mask, g: Optional[Tensor] = None, reverse: bool = False)
        //
        // 4 args: x (z_p), x_mask (y_mask), g (None for n_speakers=0), reverse.
        // v2.3.1 passed only 3 args (missing g) — crashed at flow_forward
        // with native SIGSEGV.
        //
        // For Optional[Tensor] args, TorchScript accepts IValue.optionalNull()
        // which represents Python None.
        lastInferenceStep = "flow_forward"
        val zPTensor = Tensor.fromBlob(zP, longArrayOf(1, inter.toLong(), yLengths.toLong()))
        val yMaskT = Tensor.fromBlob(
            FloatArray(yLengths) { 1f },
            longArrayOf(1, 1, yLengths.toLong()),
        )
        Log.i(TAG, "step=$lastInferenceStep: calling flow.forward(z_p=${zPTensor.shape().contentToString()}, " +
            "y_mask=${yMaskT.shape().contentToString()}, g=None, reverse=true) [4-arg, per source]")
        val zTensor = flow!!.forward(
            IValue.from(zPTensor),
            IValue.from(yMaskT),
            IValue.optionalNull(),
            IValue.from(true),
        ).toTensor()
        Log.i(TAG, "step=$lastInferenceStep: OK — z=${zTensor.shape().contentToString()}")

        // -------- 7. dec.forward(z * y_mask[:,:,:max_len], g=None) -> waveform --------
        // From runtime/models.py SynthesizerTrn.infer() line 582:
        //   o = self.dec((z * y_mask)[:,:,:max_len], g=g)
        //
        // Generator.forward signature takes (z, g) — NOT (z, max_len)!
        // max_len is applied via SLICING before the call: (z * y_mask)[:,:,:max_len]
        // v2.2.1 incorrectly passed max_len as the second arg — that would
        // crash at dec_forward (we haven't reached it yet because flow_forward
        // crashed first, but it needs fixing too).
        //
        // Apply y_mask to z (y_mask is all-ones in valid region, so z is
        // unchanged), then slice to max_len, then forward with g=None.
        lastInferenceStep = "dec_forward"
        val maxLen = 4000
        // Slice z to [:,:,:maxLen] — z shape is [1, inter, yLengths]
        val zSlicedLen = min(yLengths, maxLen)
        Log.i(TAG, "step=$lastInferenceStep: calling dec.forward(z[:,:,:$zSlicedLen], g=None) " +
            "[2-arg: z + g, per source; max_len applied via slicing]")
        // Build a sliced z tensor (z is already masked since y_mask is all-ones)
        val zSliced = Tensor.fromBlob(
            zTensor.getDataAsFloatArray().copyOfRange(0, inter * zSlicedLen),
            longArrayOf(1, inter.toLong(), zSlicedLen.toLong()),
        )
        val outTensor = dec!!.forward(
            IValue.from(zSliced),
            IValue.optionalNull(),
        ).toTensor()
        Log.i(TAG, "step=$lastInferenceStep: OK — out=${outTensor.shape().contentToString()}")

        lastInferenceStep = "done"
        Log.i(TAG, "infer() DONE — returning ${outTensor.getDataAsFloatArray().size} samples")
        return outTensor.getDataAsFloatArray()
    }
}

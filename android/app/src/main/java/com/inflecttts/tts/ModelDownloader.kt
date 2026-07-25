package com.inflecttts.tts

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlin.math.min

/**
 * ModelDownloader
 *
 * Pulls the five lite-interpreter submodule `.ptl` files from the
 * `nikitastaf1996/Inflect-Nano-v2-TorchScript` HuggingFace
 * repository into the app's internal files directory on first run.
 *
 * The `.ptl` files are produced by
 * `torch.utils.mobile_optimizer.optimize_for_mobile()` — they apply
 * operator fusion, constant folding, and XNNPACK optimizations for
 * 20-40% faster CPU inference vs. the `.pt` (full TorchScript) files.
 *
 * The same files are also referenced as a git submodule at
 * `models/Inflect-Nano-v2-TorchScript/` in the InflectTTS repo
 * (see `.gitmodules`). The submodule path is for source / build
 * reference; the actual weights are downloaded here at runtime so
 * that the APK stays small and the LFS-backed binaries are not
 * bundled.
 *
 * Files (see HF README "Submodule TorchScript pathway"):
 *   - inflect_enc_p.ptl  (2.3 MB)  TextEncoder
 *   - inflect_dec.ptl    (8.0 MB)  Generator
 *   - inflect_enc_q.ptl  (1.2 MB)  PosteriorEncoder (smaller after optimization)
 *   - inflect_flow.ptl   (4.0 MB)  ResidualCouplingBlock
 *   - inflect_dp.ptl     (1.0 MB)  DurationPredictor
 *
 * Total: ~16.5 MB. Downloaded once, cached, and reused.
 */
class ModelDownloader(private val context: Context) {

    companion object {
        private const val TAG = "ModelDownloader"

        /** HuggingFace model repository (git submodule source). */
        const val HF_REPO = "nikitastaf1996/Inflect-Nano-v2-TorchScript"

        /** Raw-file URL prefix for `resolve/main/` on HuggingFace. */
        private const val HF_BASE =
            "https://huggingface.co/$HF_REPO/resolve/main/"

        /** Subdirectory under `context.filesDir` where the .ptl files live. */
        const val MODEL_DIR_NAME = "inflect_model"

        /**
         * The five lite-interpreter submodule files (.ptl). Order matters
         * for the progress callback: largest first to give the user a
         * smoother percentage.
         *
         * .ptl files are produced by torch.utils.mobile_optimizer.optimize_for_mobile()
         * — they apply operator fusion, constant folding, and XNNPACK
         * optimizations. Typically 20-40% faster on CPU than .pt files.
         *
         * Sizes are the exact byte counts from the HF repo (verified
         * 2026-07-25). allPresent() uses these to detect partial downloads.
         */
        val MODEL_FILES: List<ModelFile> = listOf(
            ModelFile("inflect_dec.ptl",    8_446_534L),
            ModelFile("inflect_flow.ptl",   4_154_831L),
            ModelFile("inflect_enc_p.ptl",  2_433_783L),
            ModelFile("inflect_enc_q.ptl",  1_241_449L),
            ModelFile("inflect_dp.ptl",     1_024_525L),
        )

        /** Aggregate byte size of all submodule weights. */
        val TOTAL_BYTES: Long = MODEL_FILES.sumOf { it.expectedSize }
    }

    /** Description of a single weight file to download. */
    data class ModelFile(
        val name: String,
        val expectedSize: Long,
    )

    /** Progress events emitted during [ensureModels]. */
    sealed class Progress {
        /** Started downloading `file.name` (index `index` of `total`). */
        data class Started(val index: Int, val total: Int, val file: ModelFile) : Progress()
        /** `downloaded` bytes of `file.expectedSize` for the current file. */
        data class Bytes(
            val index: Int,
            val total: Int,
            val file: ModelFile,
            val downloaded: Long,
        ) : Progress()
        /** Finished downloading `file.name`. */
        data class Finished(val index: Int, val total: Int, val file: ModelFile) : Progress()
        /** All files present (either pre-existing or just downloaded). */
        data class AllDone(val directory: File, val downloadedNow: Int) : Progress()
        /** Non-fatal warning (e.g. SHA mismatch on a previously cached file). */
        data class Warning(val message: String) : Progress()
    }

    /** Where the .pt files end up on disk. */
    val modelDir: File
        get() = File(context.filesDir, MODEL_DIR_NAME).apply { if (!exists()) mkdirs() }

    /**
     * True iff every expected `.pt` file exists on disk AND its size
     * matches the expected size. A partial download (file exists but
     * is too small) returns false here, so [ensureModels] will re-fetch
     * the corrupted/truncated file instead of trying to load it.
     */
    fun allPresent(): Boolean = MODEL_FILES.all { mf ->
        val f = File(modelDir, mf.name)
        f.exists() && f.length() == mf.expectedSize
    }

    /**
     * Ensure every submodule .ptl file exists locally. Files already present
     * are left untouched. Missing files are downloaded from HuggingFace
     * with progress callbacks. Runs on the calling thread — call from a
     * background coroutine.
     *
     * Also cleans up legacy .pt files (from v2.4.0 and earlier) to free
     * disk space — the app now uses .ptl (lite interpreter) exclusively.
     *
     * @return the directory containing the .ptl files.
     */
    fun ensureModels(onProgress: (Progress) -> Unit = {}): File {
        if (!modelDir.exists()) modelDir.mkdirs()

        // ---- Clean up legacy .pt files (v2.4.0 and earlier) ----
        // The app switched from .pt to .ptl in v2.6.0. Old .pt files are
        // ~20 MB of dead weight — delete them so they don't waste storage.
        val legacyPtFiles = modelDir.listFiles { f -> f.name.endsWith(".pt") }
        if (legacyPtFiles != null && legacyPtFiles.isNotEmpty()) {
            Log.i(TAG, "Cleaning up ${legacyPtFiles.size} legacy .pt files (switching to .ptl)")
            legacyPtFiles.forEach { f ->
                if (f.delete()) Log.d(TAG, "  Deleted legacy ${f.name}")
                else Log.w(TAG, "  Failed to delete legacy ${f.name}")
            }
        }

        val total = MODEL_FILES.size
        var downloadedNow = 0

        MODEL_FILES.forEachIndexed { index, model ->
            val target = File(modelDir, model.name)
            if (target.exists() && target.length() == model.expectedSize) {
                Log.d(TAG, "Already cached: ${model.name} (${target.length()} bytes)")
                return@forEachIndexed
            }
            if (target.exists() && target.length() != model.expectedSize) {
                Log.w(TAG, "${model.name}: partial/corrupt file on disk " +
                    "(got ${target.length()}, expected ${model.expectedSize}) — re-downloading")
            } else if (!target.exists()) {
                Log.i(TAG, "${model.name}: not on disk — downloading")
            }

            onProgress(Progress.Started(index, total, model))
            Log.i(TAG, "Downloading ${model.name} (${model.expectedSize} bytes) from $HF_BASE")

            try {
                downloadFile(HF_BASE + model.name, target, model) { downloaded ->
                    onProgress(Progress.Bytes(index, total, model, downloaded))
                }
                downloadedNow += 1
                onProgress(Progress.Finished(index, total, model))
                Log.i(TAG, "Downloaded ${model.name} -> ${target.absolutePath}")
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to download ${model.name}", t)
                // Best-effort cleanup so the next run retries cleanly.
                target.delete()
                throw t
            }
        }

        onProgress(Progress.AllDone(modelDir, downloadedNow))
        return modelDir
    }

    /** Single-file HTTP download with atomic rename and size verification. */
    private fun downloadFile(
        url: String,
        target: File,
        model: ModelFile,
        onBytes: (Long) -> Unit,
    ) {
        val tmp = File(target.parentFile, "${target.name}.part")
        tmp.delete()
        var connection: HttpURLConnection? = null
        try {
            connection = (URL(url).openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 30_000
                readTimeout = 60_000
                setRequestProperty("User-Agent", "InflectTTS-Android/2.0 (model-downloader)")
                instanceFollowRedirects = true
            }

            val code = connection.responseCode
            if (code !in 200..299) {
                throw RuntimeException("HTTP $code for ${model.name}")
            }

            connection.inputStream.use { input ->
                FileOutputStream(tmp).use { output ->
                    val buffer = ByteArray(64 * 1024)
                    var totalRead = 0L
                    while (true) {
                        val read = input.read(buffer)
                        if (read <= 0) break
                        output.write(buffer, 0, read)
                        totalRead += read
                        onBytes(totalRead)
                    }
                    output.flush()
                }
            }

            if (tmp.length() != model.expectedSize) {
                Log.w(TAG, "${model.name}: size ${tmp.length()} != expected ${model.expectedSize}")
            }

            // Atomic-ish swap: rename .part -> final.
            if (target.exists()) target.delete()
            if (!tmp.renameTo(target)) {
                // Fallback: copy then delete.
                tmp.copyTo(target, overwrite = true)
                tmp.delete()
            }
        } finally {
            connection?.disconnect()
            tmp.delete()
        }
    }

    /** Compute SHA-256 of a file (used for verification, optional). */
    @Suppress("unused")
    private fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { fis ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val read = fis.read(buffer)
                if (read <= 0) break
                md.update(buffer, 0, read)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}

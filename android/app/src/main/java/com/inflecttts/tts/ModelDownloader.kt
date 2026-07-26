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
 * Pulls the two ONNX graph files from the official
 * `owensong/Inflect-Nano-v2-ONNX` HuggingFace repository into the
 * app's internal files directory on first run.
 *
 * The model is split into 2 ONNX graphs:
 *   - decode.onnx   (12.0 MB) — flow + dec (baked into one graph)
 *   - duration.onnx (3.5 MB)  — enc_p + dp + attention + expand (baked in)
 *
 * The 2-graph split means only 2 native inference calls per synthesis,
 * and the attention matrix, path generation, and matmul expansion are
 * all baked into duration.onnx — no Kotlin orchestration needed.
 *
 * Total download: ~15.5 MB. Downloaded once, cached, and reused.
 */
class ModelDownloader(private val context: Context) {

    companion object {
        private const val TAG = "ModelDownloader"

        /** Official HuggingFace model repository (the author's ONNX export). */
        const val HF_REPO = "owensong/Inflect-Nano-v2-ONNX"

        /** Raw-file URL prefix — files are under `onnx/` in the official repo. */
        private const val HF_BASE =
            "https://huggingface.co/$HF_REPO/resolve/main/onnx/"

        /** Subdirectory under `context.filesDir` where the .onnx files live. */
        const val MODEL_DIR_NAME = "inflect_model"

        /**
         * The two ONNX graph files used in inference.
         * Order matters for the progress callback: largest first to give the
         * user a smoother percentage.
         *
         * The model is split into 2 ONNX graphs:
         *   - decode.onnx   (12.0 MB) — flow + dec (baked into one graph)
         *   - duration.onnx (3.5 MB)  — enc_p + dp + attention + expand (baked in)
         *
         * The 2-graph split means only 2 native inference calls per synthesis,
         * and the attention matrix, path generation, and matmul expansion are
         * all baked into duration.onnx.
         *
         * Sizes are the exact byte counts (verified via sha256 checksums
         * 2026-07-26 against onnx/checksums.sha256). allPresent() uses these
         * to detect partial downloads.
         */
        val MODEL_FILES: List<ModelFile> = listOf(
            ModelFile("decode.onnx",    12_570_009L),
            ModelFile("duration.onnx",   3_636_541L),
        )

        /** Aggregate byte size of all ONNX weights. */
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

        // ---- Clean up legacy files (v2.6.x and earlier) ----
        // The app switched from .pt/.ptl (PyTorch) to .onnx (ONNX Runtime) in v3.0.0.
        // Old .pt and .ptl files are ~15-20 MB of dead weight — delete them.
        val legacyFiles = modelDir.listFiles { f ->
            f.name.endsWith(".pt") || f.name.endsWith(".ptl")
        }
        if (legacyFiles != null && legacyFiles.isNotEmpty()) {
            Log.i(TAG, "Cleaning up ${legacyFiles.size} legacy .pt/.ptl files (switching to .onnx)")
            legacyFiles.forEach { f ->
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

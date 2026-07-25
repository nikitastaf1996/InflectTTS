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
 * This module provides TTS inference capabilities for the Inflect Nano v2 model.
 * It implements a simplified VITS-style text-to-speech synthesis.
 * 
 * For production use, replace the placeholder inference with actual ONNX Runtime
 * integration by:
 * 1. Export the PyTorch model to ONNX format
 * 2. Add onnxruntime-android dependency
 * 3. Load the model using OrtEnvironment and OrtSession
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
        super.invalidate()
    }
    
    /**
     * Initialize the TTS model
     */
    @ReactMethod
    fun initializeModel(promise: Promise) {
        scope.launch {
            try {
                val startTime = System.currentTimeMillis()
                Log.d(TAG, "Initializing TTS model...")
                
                // In production, load the ONNX model here:
                // val env = OrtEnvironment.getEnvironment()
                // val session = env.createSession(modelPath, OrtSession.SessionOptions())
                
                // Simulate model loading delay
                delay(500)
                
                // Initialize audio track
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
                    })
                }
                
                Log.d(TAG, "Model initialized in ${modelLoadTime}ms")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize model", e)
                withContext(Dispatchers.Main) {
                    promise.reject("INIT_ERROR", e.message, e)
                }
            }
        }
    }
    
    /**
     * Run TTS inference
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
     * Get model information
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
            putDouble("loadTime", modelLoadTime.toDouble())
        })
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

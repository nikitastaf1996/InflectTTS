# InflectTTS — On-device Text-to-Speech for Android

<p align="center">
  <strong>Real-time English TTS using the Inflect Nano v2 model — 2.5x RT on CPU</strong>
</p>

<p align="center">
  <a href="https://github.com/nikitastaf1996/InflectTTS/releases/latest">
    <img src="https://img.shields.io/badge/Download-APK-4CAF50?style=for-the-badge" alt="Download APK" />
  </a>
  <a href="https://huggingface.co/owensong/Inflect-Nano-v2-ONNX">
    <img src="https://img.shields.io/badge/Model-Inflect%20Nano%20v2%20ONNX-1769E0?style=for-the-badge" alt="Model" />
  </a>
  <a href="https://github.com/nikitastaf1996/InflectTTS/actions/workflows/build-apk.yml">
    <img src="https://github.com/nikitastaf1996/InflectTTS/actions/workflows/build-apk.yml/badge.svg" alt="Build APK" />
  </a>
</p>

---

## Overview

InflectTTS is a React Native Android app that runs the [Inflect Nano v2](https://huggingface.co/owensong/Inflect-Nano-v2) text-to-speech model **entirely on-device** using ONNX Runtime. The model is a compact VITS-style architecture with **3.97M parameters** (~16 MB) that synthesizes 24 kHz English speech.

### Performance

| Metric | Value |
|--------|-------|
| **Realtime factor** | **2.5x RT** (measured on a mid-range Snapdragon phone) |
| **Inference engine** | ONNX Runtime 1.27 + XNNPACK EP |
| **Native calls per synthesis** | 2 (duration.onnx + decode.onnx) |
| **APK size** | 87 MB |
| **First-run download** | 15.5 MB (cached afterwards) |

2.5x RT means a 10-second utterance synthesizes in ~4 seconds — comfortable headroom for real-time playback, pre-buffering, and thermal sustainability.

## How it works

```
Text → InflectG2P (rule-based IPA) → token IDs → [duration.onnx] → m_p_exp, logs_p_exp, y_mask
                                                       ↓
                                              sample zp_noise (seeded)
                                                       ↓
                              [decode.onnx] → waveform (24 kHz float32 PCM)
                                                       ↓
                                              AudioTrack playback
```

The model is split into **2 ONNX graphs** (from the official [`owensong/Inflect-Nano-v2-ONNX`](https://huggingface.co/owensong/Inflect-Nano-v2-ONNX) repo):

| File | Size | Bakes in |
|------|------|----------|
| `duration.onnx` | 3.5 MB | TextEncoder + DurationPredictor + attention matrix + path generation + matmul expansion |
| `decode.onnx` | 12.0 MB | z_p sampling + ResidualCouplingBlock (flow) + HiFi-GAN vocoder (dec) + max_len slicing |

The 2-graph split (vs the PyTorch submodule pathway's 4 graphs + 200 lines of Kotlin orchestration) is what enables the 2.5x RT speedup.

### Phonemization

The model was trained on **eSpeak IPA phonemes** with stress markers. Since eSpeak isn't available on Android, the app uses a **rule-based English g2p** (`InflectG2P.kt`) that produces plausible IPA token IDs. Quality is lower than eSpeak (some words sound "Flemish") but the audio is recognizable English. For production-quality phonemization, bundle eSpeak NG as a native library.

## Quick start

### For users

1. Download the latest APK from [Releases](https://github.com/nikitastaf1996/InflectTTS/releases/latest)
2. Install it (allow unknown sources)
3. Open the app — first launch downloads ~15.5 MB of ONNX model files from HuggingFace
4. Enter text, tap **Synthesize Speech**

### For developers

**Prerequisites:** Node.js ≥ 22.11.0, JDK 17, Android SDK (API 24+), Android NDK (27.1.12297006)

```bash
git clone https://github.com/nikitastaf1996/InflectTTS.git
cd InflectTTS
npm install
npm run bundle:android
npm run build:apk
adb install android/app/build/outputs/apk/debug/app-debug.apk
```

No git submodules, no git-lfs, no model files in the repo — everything is downloaded at runtime.

## Project structure

```
InflectTTS/
├── .github/workflows/
│   └── build-apk.yml                    # CI: build APK + auto-release
├── android/app/src/main/java/com/inflecttts/tts/
│   ├── TTSModule.kt                     # React Native module (orchestration + crash post-mortem)
│   ├── ModelDownloader.kt               # Downloads .onnx files from HF on first run
│   ├── InflectInference.kt              # ONNX Runtime: 2 sessions, 2 run() calls
│   ├── InflectG2P.kt                    # Rule-based English → IPA token IDs
│   └── TTSPackage.kt                    # React Native package registration
├── src/TTSBridge.ts                     # TypeScript bridge to native module
├── App.tsx                              # Main UI (text input, controls, logs, diagnostics)
└── package.json
```

## Native module API

The `InflectTTS` native module exposes:

| Method | Description |
|--------|-------------|
| `initializeModel()` | Downloads `.onnx` files from HF (first run only), loads ONNX sessions, initializes AudioTrack. Emits `InflectTTS_ModelProgress` events during download. |
| `synthesize(text, speed, variation, seed)` | Runs the 2-graph inference pipeline. Rejects with `MODEL_NOT_LOADED` if the model didn't load; `SYNTHESIS_ERROR` if inference throws. |
| `getModelInfo()` | Returns model metadata + `realModelReady` flag + `loadFailureReason`. |
| `getDiagnostics()` | Returns detailed state: file sizes, session load status, `lastInferenceStep` (crash post-mortem), PyTorch/ONNX probe. |
| `redownloadModel()` | Clears the cache and re-downloads the `.onnx` files. |

### Crash post-mortem

The app persists `lastInferenceStep` and `lastInferenceInputs` to SharedPreferences (synchronous `commit()`) before every native call. If a native crash (SIGSEGV) kills the process, the next launch's `getDiagnostics()` shows exactly which step crashed — no logcat access needed.

## Configuration

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| `speed` | 0.5 – 2.0 | 1.0 | 1.0 = normal, 2.0 = 2x slower, 0.5 = 2x faster |
| `variation` | 0.0 – 1.0 | 0.667 | Latent noise scale (prosodic variation) |
| `seed` | any int | 7 | RNG seed for latent noise (same seed = same audio) |

## Build & CI

The GitHub Actions workflow (`.github/workflows/build-apk.yml`) builds a debug APK on every push to `main` and publishes it as a prerelease at [Releases](https://github.com/nikitastaf1996/InflectTTS/releases). The APK is overwritten on every push.

## References

- **Model**: [owensong/Inflect-Nano-v2-ONNX](https://huggingface.co/owensong/Inflect-Nano-v2-ONNX) — official ONNX export
- **Original model**: [owensong/Inflect-Nano-v2](https://huggingface.co/owensong/Inflect-Nano-v2) — PyTorch base model
- **VITS paper**: [arXiv:2106.06103](https://arxiv.org/abs/2106.06103)
- **ONNX Runtime Android**: [onnxruntime.ai/docs/tutorials/android](https://onnxruntime.ai/docs/tutorials/android/)

## License

MIT. The model weights inherit the license from [`owensong/Inflect-Nano-v2-ONNX`](https://huggingface.co/owensong/Inflect-Nano-v2-ONNX).

---

<p align="center">
  Built with ONNX Runtime + React Native — 2.5x realtime on a phone CPU
</p>

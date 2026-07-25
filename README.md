# InflectTTS - Android App for Inflect Nano v2

<p align="center">
  <img src="docs/icon.png" width="120" alt="InflectTTS Icon" />
</p>

<p align="center">
  <strong>On-device text-to-speech using the Inflect Nano v2 model (~4M parameters)</strong>
</p>

<p align="center">
  <a href="https://github.com/owenawsong/Inflect">
    <img src="https://img.shields.io/badge/Model-Inflect%20Nano%20v2-1769E0?style=for-the-badge" alt="Model Badge" />
  </a>
  <a href="https://github.com/nikitastaf1996/iptv-player/releases/latest">
    <img src="https://img.shields.io/badge/Download-APK-4CAF50?style=for-the-badge" alt="Download Badge" />
  </a>
</p>

---

## 🎯 Overview

InflectTTS is an Android application that demonstrates **on-device text-to-speech synthesis** using the [Inflect Nano v2](https://github.com/owenawsong/Inflect) model. This compact TTS model achieves impressive quality with only **3.97 million parameters** (~16 MB).

### Key Features

- 🔊 **Local TTS Inference** - No internet required after model is loaded
- ⏱️ **Real-time Performance Metrics** - Detailed inference timing for each step
- 🎛️ **Configurable Controls** - Speed, variation, and seed parameters
- 📊 **Comprehensive Logs** - Step-by-step breakdown of the synthesis pipeline
- 📱 **Native Audio Playback** - Real-time audio output through Android's audio engine

## 🧠 Model Information

| Specification | Value |
|---------------|-------|
| **Model** | Inflect-Nano-v2 |
| **Parameters** | 3,966,721 |
| **Model Size** | 15.97 MB (FP32) |
| **Output** | 24 kHz mono WAV |
| **Voice** | Fixed English male |
| **CPU Throughput** | 10.72x real-time |

### Inference Pipeline

The TTS synthesis consists of these steps:

1. **Text Preprocessing** - Normalization and cleaning
2. **Phoneme Encoding** - Convert text to phoneme sequence
3. **Duration Prediction** - Predict how long each phoneme takes
4. **Mel Spectrogram Generation** - Generate acoustic features
5. **Waveform Synthesis** - Neural vocoder converts features to audio
6. **Post-processing** - Normalization and speed adjustment

## 🚀 Quick Start

### Prerequisites

- Node.js ≥ 22.11.0
- JDK 17
- Android SDK (API 24+)
- Android NDK (27.1.12297006)

### Installation

```bash
# Clone the repository with submodules (HF model repo is pinned as
# a git submodule at models/Inflect-Nano-v2-TorchScript/).
git clone --recurse-submodules https://github.com/nikitastaf1996/InflectTTS.git
cd InflectTTS

# If you forgot --recurse-submodules:
git submodule update --init --recursive

# Install dependencies
npm install

# Bundle JavaScript
npm run bundle:android

# Build debug APK
npm run build:apk
```

### First-run model download

The five scripted submodule `.pt` files (`inflect_enc_p`, `inflect_dec`,
`inflect_enc_q`, `inflect_flow`, `inflect_dp`, total ~20 MB) are **not
bundled** in the APK. On first launch, the app fetches them directly from
HuggingFace (`huggingface.co/nikitastaf1996/Inflect-Nano-v2-TorchScript`)
and caches them in the app's internal storage. Subsequent launches reuse
the cache.

To force a re-download, call the exposed `redownloadModel()` method on
the native `InflectTTS` module.



### Running

```bash
# Start Metro bundler
npm start

# Run on device/emulator
npm run android
```

Or install the APK directly:
```bash
adb install android/app/build/outputs/apk/debug/app-debug.apk
```

## 📁 Project Structure

```
InflectTTS/
├── .github/
│   └── workflows/
│       └── build-apk.yml    # GitHub Actions: build APK + auto-release
├── android/
│   └── app/src/main/
│       ├── java/com/inflecttts/
│       │   └── tts/
│       │       ├── TTSModule.kt        # RN native module (orchestration; real inference only, no fallback)
│       │       ├── ModelDownloader.kt  # Pulls .pt submodules from HuggingFace on first run
│       │       ├── InflectInference.kt # Loads .pt files, reconstructs SynthesizerTrn.infer()
│       │       └── TTSPackage.kt       # React Native package
│       └── assets/
│           └── index.android.bundle
├── models/
│   └── Inflect-Nano-v2-TorchScript/   # Git submodule → HF repo (pointer files only)
├── scripts/
│   └── export_onnx.py                  # Legacy ONNX export script (unused by v2.0 runtime)
├── src/
│   └── TTSBridge.ts                    # TypeScript bridge to native module
├── App.tsx                             # Main React Native component
├── package.json
└── README.md
```

## 🔧 Development

### v2.1 submodule pathway (no fallback)

The app uses the **scripted submodule** pathway described in the
[HF README](https://huggingface.co/nikitastaf1996/Inflect-Nano-v2-TorchScript):

1. **Build time** — `models/Inflect-Nano-v2-TorchScript/` is a git
   submodule pointing at the HF repo. It contains LFS pointer files
   (132 bytes each) plus the README and Python scripts for reference.
   CI initializes the submodule via `actions/checkout@v4` with
   `submodules: recursive` — git-lfs is NOT required.

2. **Runtime (first launch)** — `ModelDownloader.kt` fetches the
   actual LFS-backed `.pt` binaries from
   `https://huggingface.co/nikitastaf1996/Inflect-Nano-v2-TorchScript/resolve/main/<file>`
   and writes them to `context.filesDir/inflect_model/`. Total: ~20 MB.

3. **Runtime (subsequent launches)** — cached files are reused; no
   network access is needed.

4. **Inference** — `InflectInference.kt` loads each `.pt` as a
   `org.pytorch.Module` and runs the VITS-style pipeline:
   `enc_p → dp → flow → dec`. If the model failed to load (network
   error, corrupted file, TorchScript incompat, OOM, …), `synthesize()`
   rejects with `MODEL_NOT_LOADED` and a human-readable reason stored
   in `loadFailureReason` (surfaced via `getModelInfo()`). The legacy
   simplified synthesizer was removed in v2.1 — the app is either
   running the real Inflect v2 inference or it is erroring out with a
   clear message.

### Native Module

The `TTSModule.kt` provides:

- `initializeModel()` — download `.pt` submodules from HF (first run only),
  load them via PyTorch Android, initialize the audio engine. Emits
  `InflectTTS_ModelProgress` events during download. On load failure,
  captures a human-readable reason in `loadFailureReason`.
- `synthesize(text, speed, variation, seed)` — **real inference only**.
  Rejects with `MODEL_NOT_LOADED` if the model didn't load; rejects
  with `SYNTHESIS_ERROR` (and a cause-chain message) if inference throws.
- `redownloadModel()` — clear the cache and re-download the submodules.
- `getModelInfo()` — model metadata + `realModelReady` flag + `engine`
  field + `loadFailureReason` (null on success).

### Legacy ONNX export (optional)

The repo still ships `scripts/export_onnx.py` for users who prefer the
ONNX Runtime pathway. It is **not** used by the v2.1 runtime — the app
loads `.pt` files via PyTorch Android instead.

```bash
pip install torch onnx huggingface_hub
python scripts/export_onnx.py --model nano --download --output ./android/app/src/main/assets/
```

## 📈 Performance

The app tracks and displays:

- **Total inference time**
- **Per-step timing** for each pipeline stage
- **Average, min, and max inference times**
- **Realtime factor** (how many times faster than real-time)

Example output:
```
⏱️ Total time: 847ms
🎵 Audio: 2.3s @ 24000Hz
⚡ Realtime factor: 2.72x
💾 Memory: ~16 MB
```

## 🔄 GitHub Actions

The project includes a GitHub Actions workflow that:

1. Builds the APK on every push to `main`
2. Publishes to a "Latest Build" prerelease
3. Provides direct download URL

### Workflow File

See [`.github/workflows/build-apk.yml`](.github/workflows/build-apk.yml) for the complete workflow.

### Badge

Add this to your README to show the latest build status:

```markdown
![Build APK](https://github.com/YOUR_USERNAME/InflectTTS/actions/workflows/build-apk.yml/badge.svg)
```

## 📚 Resources

- [Inflect GitHub Repository](https://github.com/owenawsong/Inflect)
- [Inflect Nano v2 on HuggingFace](https://huggingface.co/owensong/Inflect-Nano-v2)
- [ONNX Runtime Android Documentation](https://onnxruntime.ai/docs/tutorials/android/)
- [React Native CLI Setup](https://reactnative.dev/docs/environment-setup)

## ⚠️ Limitations

- The current app uses a **simplified synthesis** that demonstrates the UI and timing features
- For production use, integrate the actual ONNX-exported Inflect model
- Audio quality depends on the underlying model implementation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
  Built with ❤️ for on-device TTS
</p>

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
# Clone the repository
git clone https://github.com/YOUR_USERNAME/InflectTTS.git
cd InflectTTS

# Install dependencies
npm install

# Bundle JavaScript
npm run bundle:android

# Build debug APK
npm run build:apk
```

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
│       └── build-apk.yml    # GitHub Actions for auto-build
├── android/
│   └── app/src/main/
│       ├── java/com/inflecttts/
│       │   └── tts/
│       │       ├── TTSModule.kt    # Native TTS inference module
│       │       └── TTSPackage.kt   # React Native package
│       └── assets/
│           └── index.android.bundle
├── scripts/
│   └── export_onnx.py       # Model export script
├── src/
│   └── TTSBridge.ts         # TypeScript bridge to native module
├── App.tsx                  # Main React Native component
├── package.json
└── README.md
```

## 🔧 Development

### Adding the ONNX Model

To add the actual ONNX model for inference:

1. Export the model:
```bash
pip install torch onnx huggingface_hub
python scripts/export_onnx.py --model nano --download --output ./android/app/src/main/assets/
```

2. Update `TTSModule.kt` to load the ONNX model using ONNX Runtime.

### Native Module

The `TTSModule.kt` provides:

- `initializeModel()` - Load the TTS model
- `synthesize(text, speed, variation, seed)` - Run inference
- `getModelInfo()` - Get model metadata

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

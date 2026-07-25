/**
 * TTS Bridge - Native Module Interface
 *
 * This module provides the TypeScript interface to the native Android TTS module.
 */

import { NativeModules, Platform } from 'react-native';

// Type definitions
export interface TTSInitResult {
  success: boolean;
  loadTime: number;
  model: string;
  parameters: number;
  sampleRate: number;
  /** True iff the real PyTorch submodules loaded successfully. */
  realModelReady?: boolean;
  /** Human-readable reason if the model failed to load (null on success). */
  loadFailureReason?: string | null;
  /** Full stack trace if the model failed to load (null on success). */
  loadFailureStacktrace?: string | null;
  /** Which inference engine is active: 'pytorch_submodules' | 'none'. */
  engine?: string;
  /** Model source URL (HuggingFace repo). */
  modelSource?: string;
}

export interface TTSTiming {
  step: string;
  time: number;
  description: string;
}

export interface TTSSynthesisResult {
  timings: TTSTiming[];
  totalTime: number;
  outputPath: string;
  sampleRate: number;
  audioLength: number;
  audioDuration: number;
  engine?: string;
}

export interface TTSModelInfo {
  name: string;
  version: string;
  parameters: number;
  size: string;
  sampleRate: number;
  outputFormat: string;
  isLoaded: boolean;
  loadTime: number;
  /** True iff the real PyTorch submodules loaded successfully. */
  realModelReady?: boolean;
  /** Human-readable reason if the model failed to load (null on success). */
  loadFailureReason?: string | null;
  /** Full stack trace if the model failed to load (null on success). */
  loadFailureStacktrace?: string | null;
  /** Which inference engine is active: 'pytorch_submodules' | 'none'. */
  engine?: string;
  /** Model source URL (HuggingFace repo). */
  modelSource?: string;
}

// Native module interface — returns the full payload from Kotlin, which
// includes more fields than the typed interfaces above (we cast as needed).
interface TTSNativeModule {
  initializeModel(): Promise<any>;
  synthesize(
    text: string,
    speed: number,
    variation: number,
    seed: number
  ): Promise<any>;
  getModelInfo(): Promise<any>;
  getDiagnostics(): Promise<any>;
  redownloadModel(): Promise<boolean>;
}

// Get native module
const { InflectTTS } = NativeModules;

// Validate native module is available
if (Platform.OS === 'android' && !InflectTTS) {
  console.warn(
    'InflectTTS native module is not available. ' +
    'TTS functionality will be simulated in JavaScript.'
  );
}

/**
 * TTS Bridge class
 *
 * Provides a unified interface for TTS synthesis, with fallback to
 * JavaScript simulation when native module is unavailable (e.g. iOS).
 *
 * IMPORTANT: the cached `modelInfo` MUST include realModelReady,
 * loadFailureReason, and loadFailureStacktrace — otherwise App.tsx
 * can't tell whether the real model loaded, and shows a spurious
 * "no reason captured — this is a bug" message even when the model
 * loaded successfully.
 */
class TTSBridge {
  private isInitialized: boolean = false;
  private modelInfo: TTSModelInfo | null = null;

  /**
   * Initialize the TTS model. Always calls the native initializeModel()
   * (no early return) so the JS side gets fresh realModelReady /
   * loadFailureReason values every time.
   */
  async initialize(): Promise<TTSInitResult> {
    try {
      if (Platform.OS === 'android' && InflectTTS) {
        const result = await (InflectTTS as TTSNativeModule).initializeModel();
        this.isInitialized = result.success === true;
        // Cache the FULL native payload — including realModelReady,
        // loadFailureReason, loadFailureStacktrace — so getModelInfo()
        // returns them too. The previous version dropped these fields,
        // which caused App.tsx to see realModelReady=undefined and show
        // a spurious "no reason captured" error.
        this.modelInfo = {
          name: result.model,
          version: '2.1',
          parameters: result.parameters,
          size: '15.97 MB',
          sampleRate: result.sampleRate,
          outputFormat: '24 kHz mono WAV',
          isLoaded: result.success === true,
          loadTime: result.loadTime,
          realModelReady: result.realModelReady,
          loadFailureReason: result.loadFailureReason,
          loadFailureStacktrace: result.loadFailureStacktrace,
          engine: result.engine,
          modelSource: result.modelSource,
        };
        return result;
      } else {
        // Simulate initialization for iOS or when native module is unavailable
        await this.simulateDelay(1500);
        this.isInitialized = true;
        this.modelInfo = {
          name: 'Inflect-Nano-v2',
          version: '2.1',
          parameters: 3966721,
          size: '15.97 MB',
          sampleRate: 24000,
          outputFormat: '24 kHz mono WAV',
          isLoaded: true,
          loadTime: 1500,
          realModelReady: false,
          loadFailureReason: 'Native module unavailable on this platform (iOS/dev mode).',
          engine: 'none',
        };
        return {
          success: true,
          loadTime: 1500,
          model: 'Inflect-Nano-v2',
          parameters: 3966721,
          sampleRate: 24000,
          realModelReady: false,
        };
      }
    } catch (error) {
      console.error('TTS initialization failed:', error);
      throw error;
    }
  }

  /**
   * Run TTS synthesis.
   */
  async synthesize(
    text: string,
    options: {
      speed?: number;
      variation?: number;
      seed?: number;
    } = {}
  ): Promise<TTSSynthesisResult> {
    const { speed = 1.0, variation = 0.667, seed = 7 } = options;

    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      if (Platform.OS === 'android' && InflectTTS) {
        return await (InflectTTS as TTSNativeModule).synthesize(
          text,
          speed,
          variation,
          seed
        );
      } else {
        // Simulate synthesis
        return await this.simulateSynthesis(text, speed, variation, seed);
      }
    } catch (error) {
      console.error('TTS synthesis failed:', error);
      throw error;
    }
  }

  /**
   * Get model information. ALWAYS re-fetches from native (no cache) so
   * the JS side sees fresh realModelReady / loadFailureReason values.
   * The previous version cached the result and dropped the new fields.
   */
  async getModelInfo(): Promise<TTSModelInfo> {
    try {
      if (Platform.OS === 'android' && InflectTTS) {
        const info = await (InflectTTS as TTSNativeModule).getModelInfo();
        // Cache the full native payload.
        this.modelInfo = {
          name: info.name,
          version: info.version,
          parameters: info.parameters,
          size: info.size,
          sampleRate: info.sampleRate,
          outputFormat: info.outputFormat,
          isLoaded: info.isLoaded,
          loadTime: info.loadTime,
          realModelReady: info.realModelReady,
          loadFailureReason: info.loadFailureReason,
          loadFailureStacktrace: info.loadFailureStacktrace,
          engine: info.engine,
          modelSource: info.modelSource,
        };
        return this.modelInfo;
      } else {
        if (!this.modelInfo) {
          this.modelInfo = {
            name: 'Inflect-Nano-v2',
            version: '2.1',
            parameters: 3966721,
            size: '15.97 MB',
            sampleRate: 24000,
            outputFormat: '24 kHz mono WAV',
            isLoaded: this.isInitialized,
            loadTime: 1500,
            realModelReady: false,
            loadFailureReason: 'Native module unavailable on this platform.',
            engine: 'none',
          };
        }
        return this.modelInfo;
      }
    } catch (error) {
      console.error('Failed to get model info:', error);
      return {
        name: 'Inflect-Nano-v2',
        version: '2.1',
        parameters: 3966721,
        size: '15.97 MB',
        sampleRate: 24000,
        outputFormat: '24 kHz mono WAV',
        isLoaded: false,
        loadTime: 0,
        realModelReady: false,
        loadFailureReason: `getModelInfo() failed: ${error}`,
        engine: 'none',
      };
    }
  }

  /**
   * Check if TTS is initialized.
   */
  isReady(): boolean {
    return this.isInitialized;
  }

  /**
   * Simulate synthesis for development/testing (non-Android only).
   */
  private async simulateSynthesis(
    text: string,
    speed: number,
    variation: number,
    seed: number
  ): Promise<TTSSynthesisResult> {
    // Simulate timing for each step
    const timings: TTSTiming[] = [
      { step: 'preprocessing', time: 10 + Math.random() * 20, description: 'Text normalization' },
      { step: 'phonemeEncoding', time: 30 + Math.random() * 30, description: 'Convert to phonemes' },
      { step: 'durationPrediction', time: 50 + Math.random() * 50, description: 'Predict durations' },
      { step: 'melGeneration', time: 100 + Math.random() * 100, description: 'Generate mel spectrogram' },
      { step: 'waveformSynthesis', time: 200 + Math.random() * 200, description: 'Neural vocoder synthesis' },
      { step: 'postProcessing', time: 20 + Math.random() * 20, description: 'Audio post-processing' },
    ];

    const totalTime = timings.reduce((sum, t) => sum + t.time, 0);
    const audioDuration = (text.length * 0.5) / speed;

    await this.simulateDelay(totalTime / 10);

    return {
      timings,
      totalTime,
      outputPath: `/data/user/0/com.inflecttts/files/inflect_output_${Date.now()}.wav`,
      sampleRate: 24000,
      audioLength: Math.round(audioDuration * 24000),
      audioDuration,
      engine: 'simulation',
    };
  }

  /**
   * Helper: Simulate async delay
   */
  private simulateDelay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Export singleton instance
export const ttsBridge = new TTSBridge();

// Export class for testing
export { TTSBridge };

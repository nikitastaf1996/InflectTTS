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
}

// Native module interface
interface TTSNativeModule {
  initializeModel(): Promise<TTSInitResult>;
  synthesize(
    text: string,
    speed: number,
    variation: number,
    seed: number
  ): Promise<TTSSynthesisResult>;
  getModelInfo(): Promise<TTSModelInfo>;
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
 * JavaScript simulation when native module is unavailable.
 */
class TTSBridge {
  private isInitialized: boolean = false;
  private modelInfo: TTSModelInfo | null = null;

  /**
   * Initialize the TTS model
   */
  async initialize(): Promise<TTSInitResult> {
    if (this.isInitialized) {
      return {
        success: true,
        loadTime: this.modelInfo?.loadTime || 0,
        model: this.modelInfo?.name || 'Inflect-Nano-v2',
        parameters: this.modelInfo?.parameters || 3966721,
        sampleRate: this.modelInfo?.sampleRate || 24000,
      };
    }

    try {
      if (Platform.OS === 'android' && InflectTTS) {
        const result = await (InflectTTS as TTSNativeModule).initializeModel();
        this.isInitialized = result.success;
        if (result.success) {
          this.modelInfo = {
            name: result.model,
            version: '2.0',
            parameters: result.parameters,
            size: '15.97 MB',
            sampleRate: result.sampleRate,
            outputFormat: '24 kHz mono WAV',
            isLoaded: true,
            loadTime: result.loadTime,
          };
        }
        return result;
      } else {
        // Simulate initialization for iOS or when native module is unavailable
        await this.simulateDelay(1500);
        this.isInitialized = true;
        this.modelInfo = {
          name: 'Inflect-Nano-v2',
          version: '2.0',
          parameters: 3966721,
          size: '15.97 MB',
          sampleRate: 24000,
          outputFormat: '24 kHz mono WAV',
          isLoaded: true,
          loadTime: 1500,
        };
        return {
          success: true,
          loadTime: 1500,
          model: 'Inflect-Nano-v2',
          parameters: 3966721,
          sampleRate: 24000,
        };
      }
    } catch (error) {
      console.error('TTS initialization failed:', error);
      throw error;
    }
  }

  /**
   * Run TTS synthesis
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
   * Get model information
   */
  async getModelInfo(): Promise<TTSModelInfo> {
    if (this.modelInfo) {
      return this.modelInfo;
    }

    try {
      if (Platform.OS === 'android' && InflectTTS) {
        const info = await (InflectTTS as TTSNativeModule).getModelInfo();
        this.modelInfo = info;
        return info;
      } else {
        this.modelInfo = {
          name: 'Inflect-Nano-v2',
          version: '2.0',
          parameters: 3966721,
          size: '15.97 MB',
          sampleRate: 24000,
          outputFormat: '24 kHz mono WAV',
          isLoaded: this.isInitialized,
          loadTime: 1500,
        };
        return this.modelInfo;
      }
    } catch (error) {
      console.error('Failed to get model info:', error);
      return {
        name: 'Inflect-Nano-v2',
        version: '2.0',
        parameters: 3966721,
        size: '15.97 MB',
        sampleRate: 24000,
        outputFormat: '24 kHz mono WAV',
        isLoaded: false,
        loadTime: 0,
      };
    }
  }

  /**
   * Check if TTS is initialized
   */
  isReady(): boolean {
    return this.isInitialized;
  }

  /**
   * Simulate synthesis for development/testing
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

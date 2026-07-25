/**
 * InflectTTS - Android App for Inflect Nano v2 TTS Model Inference
 * 
 * This app demonstrates on-device text-to-speech using the Inflect Nano v2 model
 * (~4M parameters, 16MB). It shows detailed inference timing and performance logs.
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  Alert,
  FlatList,
  StatusBar,
  PermissionsAndroid,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ttsBridge, TTSModelInfo } from './src/TTSBridge';

// Types
interface InferenceLog {
  id: string;
  timestamp: Date;
  type: 'info' | 'success' | 'error' | 'warning' | 'timing';
  message: string;
  duration?: number;
}

interface PerformanceMetrics {
  totalInferences: number;
  averageTime: number;
  minTime: number;
  maxTime: number;
  lastInferenceTime: number;
  modelLoadTime: number;
  memoryUsage: number;
}

// Default sample texts
const SAMPLE_TEXTS = [
  "Hello, this is a test of the Inflect text to speech system.",
  "The quick brown fox jumps over the lazy dog.",
  "Artificial intelligence is transforming the world.",
  "A complete local voice can fit almost anywhere.",
  "Welcome to the future of on-device speech synthesis.",
];

// Model configuration
const MODEL_CONFIG = {
  name: 'Inflect-Nano-v2',
  parameters: '3,966,721',
  size: '15.97 MB',
  sampleRate: 24000,
  outputFormat: '24 kHz mono WAV',
};

const App: React.FC = () => {
  // State
  const [inputText, setInputText] = useState<string>('');
  const [isInferring, setIsInferring] = useState<boolean>(false);
  const [logs, setLogs] = useState<InferenceLog[]>([]);
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    totalInferences: 0,
    averageTime: 0,
    minTime: Infinity,
    maxTime: 0,
    lastInferenceTime: 0,
    modelLoadTime: 0,
    memoryUsage: 0,
  });
  const [isModelLoaded, setIsModelLoaded] = useState<boolean>(false);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [modelInfo, setModelInfo] = useState<TTSModelInfo | null>(null);
  const [speed, setSpeed] = useState<number>(1.0);
  const [variation, setVariation] = useState<number>(0.667);
  const [seed, setSeed] = useState<number>(7);
  
  const logIdCounter = useRef<number>(0);
  const scrollViewRef = useRef<ScrollView>(null);

  // Generate unique log ID
  const generateLogId = (): string => {
    logIdCounter.current += 1;
    return `${Date.now()}-${logIdCounter.current}`;
  };

  // Add log entry
  const addLog = useCallback((
    type: InferenceLog['type'],
    message: string,
    duration?: number
  ) => {
    const newLog: InferenceLog = {
      id: generateLogId(),
      timestamp: new Date(),
      type,
      message,
      duration,
    };
    setLogs(prev => [...prev, newLog]);
    // Auto-scroll to bottom
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 100);
  }, []);

  // Update metrics
  const updateMetrics = useCallback((
    inferenceTime: number,
    loadTime?: number
  ) => {
    setMetrics(prev => {
      const newTotal = prev.totalInferences + 1;
      const newAverage = (prev.averageTime * prev.totalInferences + inferenceTime) / newTotal;
      return {
        ...prev,
        totalInferences: newTotal,
        averageTime: newAverage,
        minTime: Math.min(prev.minTime, inferenceTime),
        maxTime: Math.max(prev.maxTime, inferenceTime),
        lastInferenceTime: inferenceTime,
        modelLoadTime: loadTime ?? prev.modelLoadTime,
      };
    });
  }, []);

  // Load model on mount
  const loadModel = useCallback(async () => {
    addLog('info', '🚀 Starting model initialization (v2.0 submodule pathway)...');

    // Subscribe to native model-download progress events emitted by TTSModule.
    let progressSubscription: any = null;
    try {
      const { DeviceEventEmitter } = require('react-native');
      progressSubscription = DeviceEventEmitter.addListener(
        'InflectTTS_ModelProgress',
        (event: { phase: string; message: string; progress: number }) => {
          const pct = Math.max(0, Math.min(100, Math.round((event.progress || 0) * 100)));
          const icon =
            event.phase === 'download_start' ? '📥'
            : event.phase === 'downloading' ? '⬇️'
            : event.phase === 'file_done' ? '✓'
            : event.phase === 'download_done' ? '✅'
            : event.phase === 'cached' ? '💾'
            : event.phase === 'model_loaded' ? '🧠'
            : event.phase === 'model_load_failed' ? '⚠️'
            : 'ℹ️';
          addLog('info', `${icon} ${event.message} (${pct}%)`);
        }
      );
    } catch (e) {
      // DeviceEventEmitter may be unavailable in some environments.
    }

    try {
      // Request permissions on Android
      if (Platform.OS === 'android') {
        try {
          const granted = await PermissionsAndroid.request(
            PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE,
            {
              title: 'Storage Permission',
              message: 'App needs storage access to save audio files.',
              buttonPositive: 'OK',
            }
          );
          if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
            addLog('warning', '⚠️ Storage permission not granted');
          }
        } catch (err) {
          console.warn('Permission error:', err);
        }
      }

      // Initialize TTS bridge — this triggers the HuggingFace download on
      // first run, then loads the five scripted submodules via PyTorch.
      const initResult: any = await ttsBridge.initialize();

      // Get model info
      const info = await ttsBridge.getModelInfo();
      setModelInfo(info);

      setIsModelLoaded(true);
      updateMetrics(0, initResult.loadTime);

      addLog('success', `✅ Model ready in ${initResult.loadTime.toFixed(0)}ms`, initResult.loadTime);
      addLog('info', `📊 Model: ${info.name} (${info.parameters.toLocaleString()} params)`);
      addLog('info', `📊 Output: ${info.outputFormat}`);
      addLog('info', `📊 Model size: ${info.size}`);
      addLog('info', `📦 Source: ${(info as any).modelSource || 'huggingface'}`);
      if ((info as any).realModelReady) {
        addLog('success', '🧠 PyTorch submodules loaded — real Inflect v2 inference active');
      } else {
        // No fallback. Surface the exact reason and stack trace so the
        // user can debug without needing logcat access.
        const reason = (info as any).loadFailureReason
          || 'No failure reason captured — this is a bug. Call getDiagnostics() and check logcat.';
        const stack: string | undefined = (info as any).loadFailureStacktrace;
        addLog('error', `❌ Model failed to load. Synthesis is disabled.`);
        addLog('error', `   Reason: ${reason}`);
        if (stack) {
          // Log the first ~10 lines of the stack trace so they're visible
          // in the scrollable log panel without overwhelming it.
          const stackLines = stack.split('\n').slice(0, 12);
          addLog('error', `   Stack trace (first ${stackLines.length} lines):`);
          stackLines.forEach((line: string) => addLog('error', `     ${line}`));
          if (stack.split('\n').length > 12) {
            addLog('info', `     … (${stack.split('\n').length - 12} more lines — see logcat for full trace)`);
          }
        }
        addLog('info', `   Tip: call getDiagnostics() for file-level details, or redownloadModel() to retry.`);
        Alert.alert(
          'Model load failed',
          `The Inflect v2 model could not be loaded:\n\n${reason}\n\n` +
          (stack ? `Stack trace (first 5 lines):\n${stack.split('\n').slice(0, 5).join('\n')}\n\n` : '') +
          `Synthesis is disabled. Tap "Get Diagnostics" in the log panel for file-level details, ` +
          `or "Redownload model" to retry.`,
        );
      }

    } catch (error: any) {
      const code = error?.code || 'UNKNOWN';
      const msg = error?.message || String(error);
      addLog('error', `❌ Failed to load model [${code}]: ${msg}`);
    } finally {
      try { progressSubscription?.remove(); } catch (_) { /* ignore */ }
    }
  }, [addLog, updateMetrics]);

  // Run native getDiagnostics() and dump the result to the log panel.
  // Useful when the model fails to load — shows per-file sizes, which
  // modules loaded, and whether the PyTorch native lib is available.
  const runDiagnostics = useCallback(async () => {
    const { InflectTTS } = (await import('react-native')).NativeModules;
    if (!InflectTTS || !InflectTTS.getDiagnostics) {
      addLog('error', '❌ getDiagnostics() not available on this platform');
      return;
    }
    addLog('info', '🔎 Running diagnostics…');
    try {
      const diag: any = await InflectTTS.getDiagnostics();
      addLog('info', `   isInitialized: ${diag.isInitialized}`);
      addLog('info', `   realModelReady: ${diag.realModelReady}`);
      addLog('info', `   modelDir: ${diag.modelDir}`);
      addLog('info', `   modelDirExists: ${diag.modelDirExists}`);
      if (diag.loadFailureReason) {
        addLog('error', `   loadFailureReason: ${diag.loadFailureReason}`);
      }
      if (diag.files && Array.isArray(diag.files)) {
        addLog('info', `   Files:`);
        diag.files.forEach((f: any) => {
          const sizeOk = f.sizeMatches ? '✓' : '✗';
          const actualKB = f.actualSize >= 0 ? `${(f.actualSize / 1024).toFixed(0)} KB` : 'MISSING';
          const expectedKB = `${(f.expectedSize / 1024).toFixed(0)} KB`;
          addLog(f.sizeMatches ? 'info' : 'error',
            `     ${sizeOk} ${f.name}: ${actualKB} / ${expectedKB}`);
        });
      }
      if (diag.pytorchProbe) {
        addLog('info', `   PyTorch probe:`);
        addLog('info', `     classLoaded: ${diag.pytorchProbe.classLoaded}`);
        addLog('info', `     nativeLibStatus: ${diag.pytorchProbe.nativeLibStatus}`);
      }
      if (diag.inference) {
        addLog('info', `   Inference modules loaded:`);
        const m = diag.inference.modulesLoaded || {};
        addLog('info', `     encP=${m.encP} dec=${m.dec} encQ=${m.encQ} flow=${m.flow} dp=${m.dp}`);
      }
      addLog('success', '✅ Diagnostics complete — see logcat for full details');
    } catch (error: any) {
      addLog('error', `❌ Diagnostics failed: ${error?.message || error}`);
    }
  }, [addLog]);

  // Run TTS inference
  const runInference = useCallback(async () => {
    if (!inputText.trim()) {
      Alert.alert('Error', 'Please enter some text to synthesize');
      return;
    }

    if (!isModelLoaded) {
      Alert.alert('Error', 'Model not loaded. Please wait for initialization.');
      return;
    }

    setIsInferring(true);
    
    try {
      addLog('info', '━'.repeat(40));
      addLog('info', `📝 Input: "${inputText.substring(0, 50)}${inputText.length > 50 ? '...' : ''}"`);
      addLog('info', `⚙️ Speed: ${speed}x, Variation: ${variation.toFixed(3)}, Seed: ${seed}`);

      // Run synthesis
      const result = await ttsBridge.synthesize(inputText, { speed, variation, seed });
      
      // Log each timing step
      result.timings.forEach(timing => {
        addLog('timing', `   ${timing.description}: ${timing.time.toFixed(1)}ms`, timing.time);
      });
      
      // Calculate realtime factor
      const realtimeFactor = result.audioDuration / (result.totalTime / 1000);
      
      addLog('info', '━'.repeat(40));
      addLog('success', `✅ Synthesis complete!`, result.totalTime);
      addLog('info', `⏱️ Total time: ${result.totalTime.toFixed(0)}ms`);
      addLog('info', `🎵 Audio: ${result.audioDuration.toFixed(1)}s @ ${result.sampleRate}Hz`);
      addLog('info', `⚡ Realtime factor: ${realtimeFactor.toFixed(2)}x (target: 10.72x)`);
      addLog('info', `💾 Memory: ~${modelInfo?.size || '16 MB'}`);
      
      updateMetrics(result.totalTime);

      // Show completion alert
      Alert.alert(
        'Synthesis Complete! 🎉',
        `Time: ${result.totalTime.toFixed(0)}ms\nAudio: ${result.audioDuration.toFixed(1)}s\nRealtime: ${realtimeFactor.toFixed(2)}x`,
        [
          { text: 'OK', style: 'default' },
          { 
            text: 'View Details', 
            onPress: () => {
              addLog('info', '📋 Synthesis details logged above');
            }
          },
        ]
      );

    } catch (error: any) {
      // The native side rejects with code MODEL_NOT_LOADED (no model) or
      // SYNTHESIS_ERROR (inference threw). The .message field already
      // contains a human-readable cause chain built on the Kotlin side.
      const code = error?.code || 'UNKNOWN';
      const msg = error?.message || String(error);
      addLog('error', `❌ Inference failed [${code}]: ${msg}`);
      Alert.alert(
        code === 'MODEL_NOT_LOADED' ? 'Model not loaded' : 'Inference failed',
        `${msg}`,
      );
    } finally {
      setIsInferring(false);
    }
  }, [inputText, isModelLoaded, speed, variation, seed, modelInfo, addLog, updateMetrics]);

  // Load sample text
  const loadSampleText = useCallback((index: number) => {
    setInputText(SAMPLE_TEXTS[index % SAMPLE_TEXTS.length]);
    addLog('info', `📋 Loaded sample text ${index + 1}`);
  }, [addLog]);

  // Clear logs
  const clearLogs = useCallback(() => {
    setLogs([]);
    addLog('info', '🗑️ Logs cleared');
  }, [addLog]);

  // Load model on mount
  useEffect(() => {
    loadModel();
  }, []);

  // Get log color based on type
  const getLogColor = (type: InferenceLog['type']): string => {
    switch (type) {
      case 'success': return '#4CAF50';
      case 'error': return '#F44336';
      case 'warning': return '#FF9800';
      case 'timing': return '#2196F3';
      default: return '#E0E0E0';
    }
  };

  // Format timestamp
  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit',
      fractionalSecondDigits: 3 
    });
  };

  // Render log item
  const renderLogItem = ({ item }: { item: InferenceLog }) => (
    <View style={[styles.logItem, { borderLeftColor: getLogColor(item.type) }]}>
      <Text style={styles.logTimestamp}>{formatTime(item.timestamp)}</Text>
      <Text style={[styles.logMessage, { color: getLogColor(item.type) }]}>
        {item.message}
      </Text>
      {item.duration !== undefined && item.duration > 0 && (
        <Text style={styles.logDuration}>{item.duration.toFixed(1)}ms</Text>
      )}
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#1a1a2e" />
      
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>🎙️ InflectTTS</Text>
        <Text style={styles.subtitle}>
          {MODEL_CONFIG.name} • {MODEL_CONFIG.parameters} params
        </Text>
        <View style={styles.statusRow}>
          <View style={[
            styles.statusIndicator, 
            { backgroundColor: isModelLoaded ? '#4CAF50' : '#FF9800' }
          ]} />
          <Text style={styles.statusText}>
            {isModelLoaded ? 'Model Ready' : 'Loading...'}
          </Text>
        </View>
      </View>

      {/* Metrics Panel */}
      <View style={styles.metricsPanel}>
        <View style={styles.metricItem}>
          <Text style={styles.metricValue}>{metrics.totalInferences}</Text>
          <Text style={styles.metricLabel}>Inferences</Text>
        </View>
        <View style={styles.metricItem}>
          <Text style={styles.metricValue}>
            {metrics.averageTime > 0 ? metrics.averageTime.toFixed(0) : '--'}
          </Text>
          <Text style={styles.metricLabel}>Avg (ms)</Text>
        </View>
        <View style={styles.metricItem}>
          <Text style={styles.metricValue}>
            {metrics.lastInferenceTime > 0 ? metrics.lastInferenceTime.toFixed(0) : '--'}
          </Text>
          <Text style={styles.metricLabel}>Last (ms)</Text>
        </View>
        <View style={styles.metricItem}>
          <Text style={styles.metricValue}>
            {metrics.minTime < Infinity ? metrics.minTime.toFixed(0) : '--'}
          </Text>
          <Text style={styles.metricLabel}>Min (ms)</Text>
        </View>
      </View>

      {/* Input Section */}
      <View style={styles.inputSection}>
        <TextInput
          style={styles.textInput}
          placeholder="Enter text to synthesize..."
          placeholderTextColor="#888"
          value={inputText}
          onChangeText={setInputText}
          multiline
          maxLength={500}
        />
        
        {/* Sample Texts */}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.sampleButtons}>
          {SAMPLE_TEXTS.map((text, index) => (
            <TouchableOpacity
              key={index}
              style={styles.sampleButton}
              onPress={() => loadSampleText(index)}
            >
              <Text style={styles.sampleButtonText}>Sample {index + 1}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        {/* Advanced Options Toggle */}
        <TouchableOpacity 
          style={styles.advancedToggle}
          onPress={() => setShowAdvanced(!showAdvanced)}
        >
          <Text style={styles.advancedToggleText}>
            {showAdvanced ? '▼ Hide' : '▶'} Advanced Options
          </Text>
        </TouchableOpacity>

        {/* Advanced Options */}
        {showAdvanced && (
          <View style={styles.advancedOptions}>
            <View style={styles.optionRow}>
              <Text style={styles.optionLabel}>Speed: {speed.toFixed(1)}x</Text>
              <View style={styles.sliderContainer}>
                <TouchableOpacity 
                  style={styles.sliderButton}
                  onPress={() => setSpeed(Math.max(0.5, speed - 0.1))}
                >
                  <Text style={styles.sliderButtonText}>-</Text>
                </TouchableOpacity>
                <View style={styles.sliderTrack}>
                  <View style={[styles.sliderFill, { width: `${((speed - 0.5) / 1.5) * 100}%` }]} />
                </View>
                <TouchableOpacity 
                  style={styles.sliderButton}
                  onPress={() => setSpeed(Math.min(2.0, speed + 0.1))}
                >
                  <Text style={styles.sliderButtonText}>+</Text>
                </TouchableOpacity>
              </View>
            </View>
            <View style={styles.optionRow}>
              <Text style={styles.optionLabel}>Variation: {variation.toFixed(2)}</Text>
              <View style={styles.sliderContainer}>
                <TouchableOpacity 
                  style={styles.sliderButton}
                  onPress={() => setVariation(Math.max(0, variation - 0.1))}
                >
                  <Text style={styles.sliderButtonText}>-</Text>
                </TouchableOpacity>
                <View style={styles.sliderTrack}>
                  <View style={[styles.sliderFill, { width: `${variation * 100}%` }]} />
                </View>
                <TouchableOpacity 
                  style={styles.sliderButton}
                  onPress={() => setVariation(Math.min(1, variation + 0.1))}
                >
                  <Text style={styles.sliderButtonText}>+</Text>
                </TouchableOpacity>
              </View>
            </View>
            <View style={styles.optionRow}>
              <Text style={styles.optionLabel}>Seed: {seed}</Text>
              <View style={styles.seedButtons}>
                <TouchableOpacity 
                  style={styles.seedButton}
                  onPress={() => setSeed(seed - 1)}
                >
                  <Text style={styles.seedButtonText}>-</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={styles.seedButton}
                  onPress={() => setSeed(Math.floor(Math.random() * 100))}
                >
                  <Text style={styles.seedButtonText}>🎲</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  style={styles.seedButton}
                  onPress={() => setSeed(seed + 1)}
                >
                  <Text style={styles.seedButtonText}>+</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        )}

        {/* Synthesize Button */}
        <TouchableOpacity
          style={[
            styles.synthesizeButton,
            isInferring && styles.synthesizeButtonDisabled,
          ]}
          onPress={runInference}
          disabled={isInferring || !isModelLoaded}
        >
          {isInferring ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator color="#000" style={styles.loader} />
              <Text style={styles.synthesizeButtonText}>Synthesizing...</Text>
            </View>
          ) : (
            <Text style={styles.synthesizeButtonText}>
              🔊 Synthesize Speech
            </Text>
          )}
        </TouchableOpacity>

        {/* Diagnostics + Redownload row — shown when the model
            failed to load, so the user can debug without logcat. */}
        <View style={styles.diagnosticsRow}>
          <TouchableOpacity
            style={styles.diagnosticsButton}
            onPress={runDiagnostics}
          >
            <Text style={styles.diagnosticsButtonText}>🔎 Get Diagnostics</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.diagnosticsButton}
            onPress={async () => {
              const { InflectTTS } = require('react-native').NativeModules;
              if (InflectTTS?.redownloadModel) {
                addLog('info', '🔄 Re-downloading model…');
                try {
                  await InflectTTS.redownloadModel();
                  addLog('success', '✅ Re-download complete. Re-initializing…');
                  setModelInfo(null);
                  setIsModelLoaded(false);
                  loadModel();
                } catch (e: any) {
                  addLog('error', `❌ Re-download failed: ${e?.message || e}`);
                }
              }
            }}
          >
            <Text style={styles.diagnosticsButtonText}>🔄 Redownload Model</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Logs Section */}
      <View style={styles.logsSection}>
        <View style={styles.logsHeader}>
          <Text style={styles.logsTitle}>📋 Inference Logs</Text>
          <TouchableOpacity onPress={clearLogs}>
            <Text style={styles.clearButton}>Clear</Text>
          </TouchableOpacity>
        </View>
        <FlatList
          ref={scrollViewRef}
          data={logs}
          renderItem={renderLogItem}
          keyExtractor={item => item.id}
          style={styles.logsList}
          contentContainerStyle={styles.logsListContent}
          showsVerticalScrollIndicator={true}
          initialNumToRender={50}
          maxToRenderPerBatch={20}
          windowSize={10}
        />
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  header: {
    padding: 16,
    backgroundColor: '#16213e',
    borderBottomWidth: 1,
    borderBottomColor: '#0f3460',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 14,
    color: '#888',
    textAlign: 'center',
    marginTop: 4,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 8,
  },
  statusIndicator: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },
  statusText: {
    fontSize: 14,
    color: '#4CAF50',
  },
  metricsPanel: {
    flexDirection: 'row',
    backgroundColor: '#16213e',
    padding: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#0f3460',
  },
  metricItem: {
    flex: 1,
    alignItems: 'center',
  },
  metricValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#00d9ff',
  },
  metricLabel: {
    fontSize: 10,
    color: '#888',
    marginTop: 2,
  },
  inputSection: {
    padding: 16,
    backgroundColor: '#16213e',
  },
  textInput: {
    backgroundColor: '#0f3460',
    borderRadius: 12,
    padding: 12,
    color: '#fff',
    fontSize: 16,
    minHeight: 80,
    maxHeight: 150,
    textAlignVertical: 'top',
    borderWidth: 1,
    borderColor: '#1a5490',
  },
  sampleButtons: {
    marginTop: 12,
    marginBottom: 8,
  },
  sampleButton: {
    backgroundColor: '#0f3460',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 8,
    borderWidth: 1,
    borderColor: '#1a5490',
  },
  sampleButtonText: {
    color: '#00d9ff',
    fontSize: 12,
  },
  advancedToggle: {
    paddingVertical: 8,
  },
  advancedToggleText: {
    color: '#888',
    fontSize: 14,
  },
  advancedOptions: {
    backgroundColor: '#0f3460',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  optionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  optionLabel: {
    color: '#fff',
    fontSize: 14,
    flex: 1,
  },
  sliderContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 2,
  },
  sliderButton: {
    backgroundColor: '#1a5490',
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sliderButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  sliderTrack: {
    flex: 1,
    height: 6,
    backgroundColor: '#333',
    borderRadius: 3,
    marginHorizontal: 8,
    overflow: 'hidden',
  },
  sliderFill: {
    height: '100%',
    backgroundColor: '#00d9ff',
    borderRadius: 3,
  },
  seedButtons: {
    flexDirection: 'row',
  },
  seedButton: {
    backgroundColor: '#1a5490',
    width: 36,
    height: 32,
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 8,
  },
  seedButtonText: {
    color: '#fff',
    fontSize: 14,
  },
  synthesizeButton: {
    backgroundColor: '#00d9ff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginTop: 8,
  },
  synthesizeButtonDisabled: {
    backgroundColor: '#555',
  },
  synthesizeButtonText: {
    color: '#000',
    fontSize: 18,
    fontWeight: 'bold',
  },
  diagnosticsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
    gap: 8,
  },
  diagnosticsButton: {
    flex: 1,
    backgroundColor: '#2a2a2a',
    borderColor: '#00d9ff',
    borderWidth: 1,
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
  },
  diagnosticsButtonText: {
    color: '#00d9ff',
    fontSize: 13,
    fontWeight: '600',
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  loader: {
    marginRight: 8,
  },
  logsSection: {
    flex: 1,
    backgroundColor: '#0d0d1a',
    padding: 12,
  },
  logsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  logsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
  },
  clearButton: {
    color: '#ff6b6b',
    fontSize: 14,
  },
  logsList: {
    flex: 1,
  },
  logsListContent: {
    paddingBottom: 20,
  },
  logItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderLeftWidth: 3,
    marginBottom: 4,
    backgroundColor: 'rgba(255,255,255,0.02)',
    borderRadius: 4,
  },
  logTimestamp: {
    color: '#555',
    fontSize: 10,
    fontFamily: 'monospace',
    marginRight: 8,
    minWidth: 80,
  },
  logMessage: {
    flex: 1,
    fontSize: 12,
    fontFamily: 'monospace',
  },
  logDuration: {
    color: '#2196F3',
    fontSize: 11,
    fontFamily: 'monospace',
    marginLeft: 8,
  },
});

export default App;

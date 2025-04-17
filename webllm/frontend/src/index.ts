// Import styles
import './styles/main.css';

// Import necessary modules
import { debug, loadTestParameters } from './utils';
import { runBenchmark, initAutoBenchmark } from './benchmark';

/**
 * Check and display WebGPU support status in detail
 */
function checkWebGPUSupport(): void {
  const webgpuStatusElement = document.getElementById('webgpu-status');
  if (!webgpuStatusElement) return;
  
  if ('gpu' in navigator) {
    webgpuStatusElement.textContent = 'Available';
    webgpuStatusElement.style.color = 'green';
    debug('WebGPU is available in this browser');
    
    // Check for adapter to confirm hardware acceleration
    (navigator.gpu as any).requestAdapter().then((adapter: any) => {
      if (adapter) {
        const adapterInfo = adapter.getInfo?.() || {}; 
        const infoStr = JSON.stringify(adapterInfo, null, 2);
        debug(`WebGPU adapter info: ${infoStr}`);
        webgpuStatusElement.textContent = `Available (${adapterInfo.name || 'Unknown GPU'})`;
        
        // Try to request a device to verify full functionality
        adapter.requestDevice().then((device: any) => {
          if (device) {
            debug('WebGPU device created successfully - hardware acceleration enabled');
            webgpuStatusElement.textContent += ' - Hardware acceleration enabled';
          }
        }).catch((err: Error) => {
          debug(`WebGPU device request failed: ${err.message}`);
          webgpuStatusElement.textContent += ' - Hardware acceleration issue';
          webgpuStatusElement.style.color = 'orange';
        });
      } else {
        debug('WebGPU is available but no adapter found - possible software rendering');
        webgpuStatusElement.textContent = 'Available - Software rendering only';
        webgpuStatusElement.style.color = 'orange';
      }
    }).catch((err: Error) => {
      debug(`WebGPU adapter request failed: ${err.message}`);
      webgpuStatusElement.textContent = 'Detected but not functioning';
      webgpuStatusElement.style.color = 'orange';
    });
  } else {
    webgpuStatusElement.textContent = 'Not Available - Benchmark will be slow or fail';
    webgpuStatusElement.style.color = 'red';
    debug('WARNING: WebGPU is not available in this browser. Performance may be limited or benchmark may fail.');
  }
}

/**
 * Main function to initialize the benchmark
 */
async function initBenchmark(): Promise<void> {
  debug('Initializing WebLLM benchmark');
  try {
    // Check WebGPU support
    checkWebGPUSupport();
    
    // Load test parameters
    const testParameters = await loadTestParameters();
    if (!testParameters) return;
    
    // Update model status
    const modelStatus = document.getElementById('model-status');
    if (modelStatus) {
      modelStatus.textContent = 'Ready to load model from CDN';
      modelStatus.className = 'status loading';
    }
    
    // Setup benchmark button click handler
    const runBenchmarkButton = document.getElementById('run-benchmark') as HTMLButtonElement;
    if (runBenchmarkButton) {
      // Enable the button once parameters are loaded
      runBenchmarkButton.disabled = false;
      
      runBenchmarkButton.addEventListener('click', async () => {
        debug('Run benchmark button clicked');
        runBenchmarkButton.disabled = true;
        
        try {
          await runBenchmark(testParameters);
        } catch (error) {
          debug(`Error during benchmark: ${(error as Error).message}`);
        } finally {
          runBenchmarkButton.disabled = false;
        }
      });
    }
    
  } catch (error) {
    debug(`ERROR during initialization: ${(error as Error).message}`);
    console.error('Initialization error:', error);
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  // Check WebGPU support
  checkWebGPUSupport();
  
  // Set up the manual run button
  const runButton = document.getElementById('run-benchmark');
  if (runButton) {
    runButton.addEventListener('click', () => {
      // Get values from the UI
      const promptElement = document.getElementById('prompt') as HTMLTextAreaElement;
      const maxTokensElement = document.getElementById('max-tokens') as HTMLInputElement;
      const temperatureElement = document.getElementById('temperature') as HTMLInputElement;
      
      const testParams = {
        test_id: 'manual-test',
        prompt: promptElement.value,
        max_tokens: parseInt(maxTokensElement.value),
        temperature: parseFloat(temperatureElement.value),
        expected_class: 'manual',
        model_name: 'TinyLlama-1.1B-Chat-v1.0-q4f32_1-MLC-1k'
      };
      
      runBenchmark(testParams);
    });
  }
  
  // Check if we should auto-start the benchmark
  if (window.location.search.includes('testParams')) {
    initAutoBenchmark();
  }
}); 
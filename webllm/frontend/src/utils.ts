import { TestParameters } from './types';

/**
 * Add debug logging function
 */
export function debug(message: string): void {
  const logElement = document.getElementById('debug-log');
  if (!logElement) return;
  
  const timestamp = new Date().toISOString().substr(11, 8); // HH:MM:SS
  logElement.innerHTML += `[${timestamp}] ${message}<br>`;
  logElement.scrollTop = logElement.scrollHeight;
  console.log(`[${timestamp}] ${message}`);
}

/**
 * Function to get URL parameters
 */
export function getUrlParams(): { testFile: string | null; resultFile: string | null; saveEndpoint: string | null } {
  const params = new URLSearchParams(window.location.search);
  const testFile = params.get('test_file');
  const resultFile = params.get('result_file');
  const saveEndpoint = params.get('save_endpoint');
  debug(`URL params: test_file=${testFile}, result_file=${resultFile}, save_endpoint=${saveEndpoint}`);
  return { testFile, resultFile, saveEndpoint };
}

/**
 * Function to update UI status
 */
export function updateStatus(message: string, type: 'loading' | 'success' | 'error' = 'loading'): void {
  const statusElement = document.getElementById('status');
  if (!statusElement) return;
  
  statusElement.textContent = message;
  statusElement.className = `status ${type}`;
  debug(`Status: ${message} (${type})`);
}

/**
 * Function to update model loading progress
 */
export function updateModelProgress(progress: number): void {
  const progressBar = document.getElementById('model-progress');
  if (!progressBar) return;
  
  const progressPercent = Math.round(progress * 100);
  progressBar.style.width = `${progressPercent}%`;
  
  const statusElement = document.getElementById('model-status');
  if (statusElement) {
    statusElement.textContent = `Loading model: ${progressPercent}%`;
  }
  
  if (progressPercent % 10 === 0) { // Log every 10%
    debug(`Model loading progress: ${progressPercent}%`);
  }
}

/**
 * Function to load test parameters from file
 */
export async function loadTestParameters(): Promise<TestParameters | null> {
  const { testFile } = getUrlParams();
  
  if (!testFile) {
    updateStatus('No test file specified in URL', 'error');
    return null;
  }
  
  debug(`Attempting to load test file: ${testFile}`);
  
  try {
    // Try with http:// protocol (relative to server) first as it's more likely to work
    try {
      debug('Trying to load with http:// protocol (server relative)');
      const fileName = testFile.split('/').pop();
      if (fileName) {
        const response = await fetch(`${fileName}`);
        if (response.ok) {
          const testParameters = await response.json();
          debug('Successfully loaded test file from server');
          return processTestParameters(testParameters);
        }
      }
    } catch (httpError) {
      debug(`Error loading from server: ${(httpError as Error).message}`);
    }
    
    // File:// protocol will likely fail in browsers due to security restrictions
    debug('Note: file:// protocol attempts usually fail due to browser security restrictions');
    debug('Falling back to default test parameters');
    
    // If all loading attempts fail, use fallback
    return useFallbackTest();
    
  } catch (error) {
    debug(`General error loading test parameters: ${(error as Error).message}`);
    return useFallbackTest();
  }
}

/**
 * Process loaded test parameters
 */
export function processTestParameters(testParameters: TestParameters): TestParameters {
  debug(`Test parameters loaded: ${JSON.stringify(testParameters)}`);
  
  // Update UI with test parameters
  const promptElement = document.getElementById('prompt') as HTMLTextAreaElement;
  const maxTokensElement = document.getElementById('max-tokens') as HTMLInputElement;
  const temperatureElement = document.getElementById('temperature') as HTMLInputElement;
  
  if (promptElement) {
    promptElement.value = testParameters.prompt;
    promptElement.disabled = false;
  }
  
  if (maxTokensElement) {
    maxTokensElement.value = testParameters.max_tokens.toString();
    maxTokensElement.disabled = false;
  }
  
  if (temperatureElement) {
    temperatureElement.value = testParameters.temperature.toString();
    temperatureElement.disabled = false;
  }
  
  // Enable benchmark button when test is loaded
  const runBenchmarkButton = document.getElementById('run-benchmark') as HTMLButtonElement;
  if (runBenchmarkButton) {
    runBenchmarkButton.disabled = false;
  }
  
  updateStatus('Test parameters loaded successfully', 'success');
  return testParameters;
}

/**
 * Use fallback test parameters
 */
export function useFallbackTest(): TestParameters {
  const fallbackTest: TestParameters = {
    prompt: "What is the capital of France?",
    max_tokens: 256,
    temperature: 0.0,
    test_id: "fallback_test",
    model_name: "TinyLlama-1.1B-Chat-v1.0-q4f32_1-MLC-1k"
  };
  
  debug(`Using fallback test parameters: ${JSON.stringify(fallbackTest)}`);
  
  // Update UI with fallback test parameters
  const promptElement = document.getElementById('prompt') as HTMLTextAreaElement;
  const maxTokensElement = document.getElementById('max-tokens') as HTMLInputElement;
  const temperatureElement = document.getElementById('temperature') as HTMLInputElement;
  
  if (promptElement) {
    promptElement.value = fallbackTest.prompt;
    promptElement.disabled = false;
  }
  
  if (maxTokensElement) {
    maxTokensElement.value = fallbackTest.max_tokens.toString();
    maxTokensElement.disabled = false;
  }
  
  if (temperatureElement) {
    temperatureElement.value = fallbackTest.temperature.toString();
    temperatureElement.disabled = false;
  }
  
  // Enable benchmark button
  const runBenchmarkButton = document.getElementById('run-benchmark') as HTMLButtonElement;
  if (runBenchmarkButton) {
    runBenchmarkButton.disabled = false;
  }
  
  updateStatus('Using fallback test parameters', 'loading');
  return fallbackTest;
}

/**
 * Helper function to format file size
 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' bytes';
  else if (bytes < 1048576) return (bytes / 1024).toFixed(2) + ' KB';
  else return (bytes / 1048576).toFixed(2) + ' MB';
} 
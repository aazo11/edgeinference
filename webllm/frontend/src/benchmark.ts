import { CreateMLCEngine, InitProgressReport } from '@mlc-ai/web-llm';
import { TestParameters, BenchmarkResult, BatchMetrics } from './types';
import { debug, updateStatus, updateModelProgress, getUrlParams } from './utils';
import { saveResultsToFile } from './results';

// Store batch metrics globally to persist across page reloads
const BATCH_STORAGE_KEY = 'webllm_batch_metrics';

// Replace the hardcoded MODEL_ID with a function to get it from test parameters
function getModelId(testParams: TestParameters): string {
  // Use the model from test parameters if available, otherwise fall back to default
  return testParams.model_name || "TinyLlama-1.1B-Chat-v1.0-q4f32_1-MLC-1k";
}

// Function to initialize or update batch metrics storage
function initBatchMetrics(testParams: TestParameters): BatchMetrics {
  // If no batch information, return empty metrics
  if (!testParams.batch_total || !testParams.batch_index) {
    return {
      avg_tokens_per_second: 0,
      avg_latency_ms: 0,
      avg_time_to_first_token_ms: 0,
      total_tests_completed: 0,
      total_tests: testParams.batch_total || 1
    };
  }

  // Try to load existing batch metrics from localStorage
  try {
    const existingMetricsJson = localStorage.getItem(BATCH_STORAGE_KEY);
    if (existingMetricsJson) {
      const existingMetrics = JSON.parse(existingMetricsJson);
      
      // Verify this is the same batch
      if (testParams.batch_id && existingMetrics.batch_id === testParams.batch_id) {
        debug(`Loaded existing batch metrics: ${existingMetricsJson}`);
        return existingMetrics;
      }
    }
  } catch (e) {
    debug(`Error loading batch metrics: ${(e as Error).message}`);
  }

  // Initialize new batch metrics
  const newMetrics: BatchMetrics = {
    avg_tokens_per_second: 0,
    avg_latency_ms: 0,
    avg_time_to_first_token_ms: 0,
    total_tests_completed: 0,
    total_tests: testParams.batch_total
  };

  // Save to localStorage
  localStorage.setItem(BATCH_STORAGE_KEY, JSON.stringify({
    ...newMetrics,
    batch_id: testParams.batch_id
  }));

  return newMetrics;
}

// Function to update batch metrics with new test result
function updateBatchMetrics(result: BenchmarkResult): BatchMetrics {
  if (!result.batch_total || !result.batch_index) {
    return {
      avg_tokens_per_second: result.tokens_per_second,
      avg_latency_ms: result.latency_ms,
      avg_time_to_first_token_ms: result.time_to_first_token_ms || 0,
      total_tests_completed: 1,
      total_tests: 1
    };
  }

  // Try to load existing metrics
  try {
    const existingMetricsJson = localStorage.getItem(BATCH_STORAGE_KEY);
    if (existingMetricsJson) {
      const existingMetrics = JSON.parse(existingMetricsJson);
      
      // Update metrics with new result using weighted average
      const testsCompleted = existingMetrics.total_tests_completed || 0;
      const newTotalCompleted = testsCompleted + 1;
      
      // Calculate new weighted averages
      const newAvgTPS = (existingMetrics.avg_tokens_per_second * testsCompleted + result.tokens_per_second) / newTotalCompleted;
      const newAvgLatency = (existingMetrics.avg_latency_ms * testsCompleted + result.latency_ms) / newTotalCompleted;
      const newAvgTTFT = (existingMetrics.avg_time_to_first_token_ms * testsCompleted + (result.time_to_first_token_ms || 0)) / newTotalCompleted;
      
      const updatedMetrics: BatchMetrics = {
        avg_tokens_per_second: newAvgTPS,
        avg_latency_ms: newAvgLatency,
        avg_time_to_first_token_ms: newAvgTTFT,
        total_tests_completed: newTotalCompleted,
        total_tests: existingMetrics.total_tests
      };
      
      // Save updated metrics
      localStorage.setItem(BATCH_STORAGE_KEY, JSON.stringify({
        ...updatedMetrics,
        batch_id: result.batch_id
      }));
      
      debug(`Updated batch metrics: ${JSON.stringify(updatedMetrics)}`);
      return updatedMetrics;
    }
  } catch (e) {
    debug(`Error updating batch metrics: ${(e as Error).message}`);
  }
  
  // If no existing metrics, create new ones
  const newMetrics: BatchMetrics = {
    avg_tokens_per_second: result.tokens_per_second,
    avg_latency_ms: result.latency_ms,
    avg_time_to_first_token_ms: result.time_to_first_token_ms || 0,
    total_tests_completed: 1,
    total_tests: result.batch_total
  };
  
  localStorage.setItem(BATCH_STORAGE_KEY, JSON.stringify({
    ...newMetrics,
    batch_id: result.batch_id
  }));
  
  return newMetrics;
}

// Function to update batch progress UI
function updateBatchUI(batchMetrics: BatchMetrics, testParams: TestParameters) {
  // Show the batch section if we're in a batch
  const batchSection = document.getElementById('batch-section');
  if (batchSection && testParams.batch_total && testParams.batch_total > 1) {
    batchSection.style.display = 'block';
  }
  
  // Update batch progress bar
  const batchProgressBar = document.getElementById('batch-progress-bar');
  if (batchProgressBar && batchMetrics.total_tests > 0) {
    const progressPercent = (batchMetrics.total_tests_completed / batchMetrics.total_tests) * 100;
    batchProgressBar.style.width = `${progressPercent}%`;
  }
  
  // Update batch counter
  const batchCurrent = document.getElementById('batch-current');
  const batchTotal = document.getElementById('batch-total');
  
  if (batchCurrent && testParams.batch_index !== undefined) {
    batchCurrent.textContent = (testParams.batch_index + 1).toString();
  }
  
  if (batchTotal && testParams.batch_total) {
    batchTotal.textContent = testParams.batch_total.toString();
  }
  
  // Update batch metrics
  const avgTpsElement = document.getElementById('avg-tokens-per-second');
  const avgLatencyElement = document.getElementById('avg-latency');
  const avgFirstTokenElement = document.getElementById('avg-first-token-time');
  const testsCompletedElement = document.getElementById('tests-completed');
  
  if (avgTpsElement) {
    avgTpsElement.textContent = batchMetrics.avg_tokens_per_second.toFixed(2);
  }
  
  if (avgLatencyElement) {
    avgLatencyElement.textContent = `${(batchMetrics.avg_latency_ms / 1000).toFixed(2)}s`;
  }
  
  if (avgFirstTokenElement) {
    avgFirstTokenElement.textContent = `${(batchMetrics.avg_time_to_first_token_ms / 1000).toFixed(2)}s`;
  }
  
  if (testsCompletedElement) {
    testsCompletedElement.textContent = `${batchMetrics.total_tests_completed}/${batchMetrics.total_tests}`;
  }
  
  // Update batch status
  const batchStatusElement = document.getElementById('batch-status');
  if (batchStatusElement) {
    if (batchMetrics.total_tests_completed >= batchMetrics.total_tests) {
      batchStatusElement.textContent = `Batch completed: ${batchMetrics.total_tests_completed} tests run with average of ${batchMetrics.avg_tokens_per_second.toFixed(2)} tokens/sec`;
      batchStatusElement.className = 'status success';
    } else {
      batchStatusElement.textContent = `Running test ${batchMetrics.total_tests_completed + 1} of ${batchMetrics.total_tests}`;
      batchStatusElement.className = 'status loading';
    }
  }
}

// Add this function to automatically start the benchmark when the page loads
export function initAutoBenchmark() {
  debug('Auto-benchmark mode activated');
  
  // Get the test parameters from the URL query string
  const urlParams = new URLSearchParams(window.location.search);
  const testParamsJson = urlParams.get('testParams');
  
  if (!testParamsJson) {
    updateStatus('Error: No test parameters provided in URL', 'error');
    return;
  }
  
  try {
    // Parse the test parameters
    const testParams: TestParameters = JSON.parse(decodeURIComponent(testParamsJson));
    debug(`Loaded test parameters: ${JSON.stringify(testParams)}`);
    
    // Initialize batch metrics if this is part of a batch
    if (testParams.batch_total && testParams.batch_total > 1) {
      const batchMetrics = initBatchMetrics(testParams);
      updateBatchUI(batchMetrics, testParams);
      debug(`Initialized batch progress: Test ${testParams.batch_index! + 1} of ${testParams.batch_total}`);
    }
    
    // Update the UI with the test parameters
    const promptDisplayElement = document.getElementById('prompt-display');
    const modelInfoElement = document.getElementById('model-name-display');
    
    if (promptDisplayElement) promptDisplayElement.textContent = testParams.prompt;
    
    // Update the model name in the UI
    if (modelInfoElement) {
      modelInfoElement.textContent = testParams.model_name || "TinyLlama-1.1B-Chat-v1.0-q4f32_1-MLC-1k";
    }
    
    // Automatically start the benchmark
    updateStatus('Auto-starting benchmark...', 'loading');
    
    // Small delay to ensure UI updates before starting
    setTimeout(() => {
      runBenchmark(testParams)
        .then(result => {
          debug('Auto-benchmark completed');
          
          // Update batch metrics if this is part of a batch
          if (testParams.batch_total && testParams.batch_total > 1) {
            const updatedBatchMetrics = updateBatchMetrics({
              ...result,
              batch_index: testParams.batch_index,
              batch_total: testParams.batch_total,
              batch_id: testParams.batch_id
            });
            updateBatchUI(updatedBatchMetrics, testParams);
          }
          
          // Signal to any parent window or process that we're done
          if (window.opener) {
            window.opener.postMessage({ type: 'benchmark-complete', result }, '*');
          }
          
          // Add this new code to create a visible completion indicator
          const completionElement = document.createElement('div');
          completionElement.id = 'benchmark-complete';
          completionElement.setAttribute('data-result', JSON.stringify(result));
          completionElement.style.display = 'none';
          document.body.appendChild(completionElement);
          
          debug('Added completion marker to DOM');
          
          // Update final metrics display
          updateMetricsDisplay(result.tokens_per_second, result.total_tokens || 0, 
                              result.time_to_first_token_ms || 0, result.latency_ms);
          
          // Also add a visible message for debugging
          updateStatus(`Benchmark complete. Results ready for collection.`, 'success');
        })
        .catch(error => {
          debug(`Auto-benchmark failed: ${error.message}`);
          
          // Signal failure with an error element
          const errorElement = document.createElement('div');
          errorElement.id = 'benchmark-error';
          errorElement.setAttribute('data-error', error.message);
          errorElement.style.display = 'none';
          document.body.appendChild(errorElement);
        });
    }, 500);
    
  } catch (error) {
    updateStatus(`Error parsing test parameters: ${(error as Error).message}`, 'error');
  }
}

// Add this function to update the metrics display
function updateMetricsDisplay(tokensPerSecond: number, totalTokens: number, 
                             timeToFirstToken: number, totalTime: number) {
  const tpsElement = document.getElementById('tokens-per-second');
  const totalTokensElement = document.getElementById('total-tokens');
  const firstTokenTimeElement = document.getElementById('first-token-time');
  const totalTimeElement = document.getElementById('total-time');
  
  if (tpsElement) tpsElement.textContent = tokensPerSecond.toFixed(2);
  if (totalTokensElement) totalTokensElement.textContent = totalTokens.toString();
  if (firstTokenTimeElement) firstTokenTimeElement.textContent = `${(timeToFirstToken / 1000).toFixed(2)}s`;
  if (totalTimeElement) totalTimeElement.textContent = `${(totalTime / 1000).toFixed(2)}s`;
}

/**
 * Function to run inference with WebLLM
 */
export async function runBenchmark(test: TestParameters): Promise<BenchmarkResult> {
  updateStatus('Initializing WebLLM engine...', 'loading');
  
  try {
    // Get the model ID from test parameters
    const MODEL_ID = getModelId(test);
    debug(`Using model: ${MODEL_ID} from CDN`);
    
    // Create model configuration with chat options
    const chatOpts = {
      temperature: parseFloat(test.temperature.toString()),
      max_gen_len: parseInt(test.max_tokens.toString())
    };
    
    debug(`Chat options: ${JSON.stringify(chatOpts)}`);
    
    const startTime = performance.now();
    
    debug('Creating WebLLM engine with CDN model...');
    debug('This may take some time to download the model files...');
    
    // Initialize WebLLM engine with the CDN model
    const engineConfig: any = {
      // Add progress callback to monitor model loading
      progress_callback: (report: InitProgressReport) => {
        const progressMsg = `Loading progress: ${(report.progress * 100).toFixed(1)}%, phase: ${report.text}`;
        debug(progressMsg);
        updateModelProgress(report.progress);
        // Update status with more detailed information
        updateStatus(`Loading model (${(report.progress * 100).toFixed(1)}%): ${report.text}`, 'loading');
      }
    };
    
    // Use any type to avoid TypeScript errors with the WebLLM API
    debug('Calling CreateMLCEngine - this will download model files if not cached...');
    try {
      const engine: any = await CreateMLCEngine(MODEL_ID, engineConfig, chatOpts);
      
      debug('WebLLM engine created successfully');
      
      // Run inference
      updateStatus('Running inference...', 'loading');
      const resultsElement = document.getElementById('results');
      if (resultsElement) {
        resultsElement.textContent = '';
      }
      
      // Keep track of generated text
      let generatedText = '';
      let tokenCount = 0;
      let firstTokenTime: number | null = null;
      
      debug('Preparing chat message for inference...');
      
      // Create messages in the OpenAI format
      const messages = [
        { role: "system", content: "You are a helpful AI assistant." },
        { role: "user", content: test.prompt }
      ];
      
      debug(`Starting generation with prompt: "${test.prompt.substring(0, 50)}..."`);
      
      // Create a flag to track first token timing
      let receivedFirstToken = false;
      
      // Start generation with streaming
      const stream = await engine.chat.completions.create({
        messages,
        model: MODEL_ID,
        temperature: parseFloat(test.temperature.toString()),
        max_tokens: parseInt(test.max_tokens.toString()),
        stream: true
      });
      
      // Variables for live metrics updates
      let lastUpdateTime = performance.now();
      let lastTokenCount = 0;
      
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || '';
        
        if (content && !receivedFirstToken) {
          receivedFirstToken = true;
          firstTokenTime = performance.now();
          debug(`First token received after ${firstTokenTime - startTime}ms`);
          
          // Update first token metric
          const firstTokenTimeElement = document.getElementById('first-token-time');
          if (firstTokenTimeElement) {
            firstTokenTimeElement.textContent = `${((firstTokenTime - startTime) / 1000).toFixed(2)}s`;
          }
        }
        
        if (content) {
          // Append the new chunk to the generated text
          generatedText += content;
          tokenCount++;
          
          // Update token count metric
          const totalTokensElement = document.getElementById('total-tokens');
          if (totalTokensElement) {
            totalTokensElement.textContent = tokenCount.toString();
          }
          
          // Update tokens per second every 500ms
          const now = performance.now();
          if (now - lastUpdateTime > 500 && firstTokenTime !== null) {
            const elapsedSinceFirstToken = (now - firstTokenTime) / 1000;
            const currentTPS = tokenCount / elapsedSinceFirstToken;
            
            // Update tokens per second metric
            const tpsElement = document.getElementById('tokens-per-second');
            if (tpsElement) {
              tpsElement.textContent = currentTPS.toFixed(2);
            }
            
            // Update total time metric
            const totalTimeElement = document.getElementById('total-time');
            if (totalTimeElement) {
              totalTimeElement.textContent = `${((now - startTime) / 1000).toFixed(2)}s`;
            }
            
            lastUpdateTime = now;
            lastTokenCount = tokenCount;
          }
          
          if (tokenCount % 10 === 0) {
            debug(`Generated approximately ${tokenCount} chunks so far`);
          }
          
          // Update the UI with the current generated text
          if (resultsElement) {
            resultsElement.textContent = generatedText;
          }
        }
      }
      
      const endTime = performance.now();
      
      // Get accurate token count from the API if available
      debug('Getting final token count and usage information...');
      
      // Now do a non-streaming call to get accurate usage statistics
      const nonStreamResponse = await engine.chat.completions.create({
        messages,
        model: MODEL_ID,
        temperature: parseFloat(test.temperature.toString()),
        max_tokens: 1  // Just to get a quick response with usage stats
      });
      
      // Extract usage information if available
      const promptTokens = nonStreamResponse.usage?.prompt_tokens || 0;
      tokenCount = nonStreamResponse.usage?.completion_tokens || tokenCount;
      const totalTokens = nonStreamResponse.usage?.total_tokens || (promptTokens + tokenCount);
      
      debug(`Token usage - Prompt: ${promptTokens}, Completion: ${tokenCount}, Total: ${totalTokens}`);
      
      // Calculate metrics
      const totalLatencyMs = endTime - startTime;
      const timeToFirstTokenMs = firstTokenTime !== null ? (firstTokenTime - startTime) : 0;
      const tokensPerSecond = firstTokenTime !== null 
        ? tokenCount / ((endTime - firstTokenTime) / 1000)
        : 0;
      
      // Log metrics
      debug(`Benchmark completed with ${tokenCount} output tokens`);
      debug(`Total latency: ${totalLatencyMs.toFixed(2)}ms`);
      debug(`Tokens per second: ${tokensPerSecond.toFixed(2)}`);
      
      // Format the benchmark result
      const benchmarkResult: BenchmarkResult = {
        test_id: test.test_id,
        prompt: test.prompt,
        response: generatedText,
        latency_ms: totalLatencyMs,
        tokens_per_second: tokensPerSecond,
        time_to_first_token_ms: timeToFirstTokenMs,
        total_tokens: tokenCount,
        success: true,
        batch_index: test.batch_index,
        batch_total: test.batch_total,
        batch_id: test.batch_id
      };
      
      // Save results to file
      await saveResultsToFile(benchmarkResult);
      
      updateStatus(`Benchmark completed: Generated ${tokenCount} tokens in ${totalLatencyMs.toFixed(2)}ms (${tokensPerSecond.toFixed(2)} tokens/sec)`, 'success');
      
      return benchmarkResult;
    } catch (engineError) {
      debug(`ERROR during engine creation: ${(engineError as Error).message}`);
      debug(`ENGINE ERROR Stack trace: ${(engineError as Error).stack}`);
      throw engineError; // Re-throw to be caught by the outer try/catch
    }
    
  } catch (error) {
    console.error('Benchmark error:', error);
    debug(`ERROR: ${(error as Error).message}`);
    debug(`Stack trace: ${(error as Error).stack}`);
    updateStatus(`Benchmark error: ${(error as Error).message}`, 'error');
    
    // Create error result
    const errorResult: BenchmarkResult = {
      test_id: test.test_id,
      prompt: test.prompt,
      response: `Error: ${(error as Error).message}`,
      latency_ms: 0,
      tokens_per_second: 0,
      success: false,
      error: (error as Error).message,
      batch_index: test.batch_index,
      batch_total: test.batch_total,
      batch_id: test.batch_id
    };
    
    // Save error results to file
    await saveResultsToFile(errorResult);
    
    return errorResult;
  }
} 
export interface TestParameters {
  test_id: string;
  model_name: string;
  prompt: string;
  temperature: number;
  max_tokens: number;
  batch_index?: number;      // Current index in the batch (0-based)
  batch_total?: number;      // Total number of tests in the batch
  batch_id?: string;         // Unique identifier for the batch
}

export interface BatchMetrics {
  avg_tokens_per_second: number;
  avg_latency_ms: number;
  avg_time_to_first_token_ms: number;
  total_tests_completed: number;
  total_tests: number;
}

export interface BenchmarkResult {
  test_id: string;
  prompt: string;
  response: string;
  latency_ms: number;
  tokens_per_second: number;
  success: boolean;
  error?: string;
  time_to_first_token_ms?: number;
  total_tokens?: number;
  batch_index?: number;
  batch_total?: number;
  batch_id?: string;
}

export interface ProgressReport {
  progress: number;
  text: string;
} 
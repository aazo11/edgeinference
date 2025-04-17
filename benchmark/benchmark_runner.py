#!/usr/bin/env python3
import csv
import time
import json
import os
import argparse
import sys
from typing import List, Dict, Any, Callable
from enum import StrEnum

class InferenceFramework(StrEnum):
    OLLAMA = "ollama"
    WEBLLM = "webllm"
    LLAMACPP = "llamacpp"
    OPENAI = "openai"
    TRANSFORMERSJS = "transformersjs"
    VLLM = "vllm"
    SGLEN = "sglen"


# Add the parent directory to the path so we can import from the implementations
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BenchmarkTest:
    def __init__(self, id: str, prompt: str, max_tokens: int, temperature: float, 
                 expected_class: str, notes: str):
        self.id = id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.expected_class = expected_class
        self.notes = notes

class BenchmarkResult:
    def __init__(self, test: BenchmarkTest, implementation: str, response: str, 
                 latency_ms: float, tokens_per_second: float, success: bool = True, 
                 error: str = None):
        self.test_id = test.id
        self.prompt = test.prompt
        self.implementation = implementation
        self.response = response
        self.latency_ms = latency_ms
        self.tokens_per_second = tokens_per_second
        self.success = success
        self.error = error
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "prompt": self.prompt,
            "implementation": self.implementation,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "tokens_per_second": self.tokens_per_second,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp
        }

class BenchmarkRunner:
    def __init__(self, tests_file: str = "tests/simple_benchmark.csv"):
        # Make paths relative to the benchmark directory
        self.benchmark_dir = os.path.dirname(os.path.abspath(__file__))
        self.tests_file = os.path.join(self.benchmark_dir, tests_file)
        self.tests = self._load_tests(self.tests_file)
        self.results = []
    
    def _load_tests(self, tests_file: str) -> List[BenchmarkTest]:
        tests = []
        with open(tests_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test = BenchmarkTest(
                    id=row['id'],
                    prompt=row['prompt'],
                    max_tokens=int(row['max_tokens']),
                    temperature=float(row['temperature']),
                    expected_class=row['expected_class'],
                    notes=row['notes']
                )
                tests.append(test)
        return tests
    
    def run_benchmarks(self, implementation_name: str, inference_fn: Callable):
        """
        Run benchmarks against a specific implementation.
        
        Args:
            implementation_name: Name of the implementation (e.g., "ollama", "webllm")
            inference_fn: A function that takes a BenchmarkTest and returns (response, latency_ms, tokens_per_second)
        """
        for test in self.tests:
            try:
                response, latency_ms, tokens_per_second = inference_fn(test)
                result = BenchmarkResult(
                    test=test,
                    implementation=implementation_name,
                    response=response,
                    latency_ms=latency_ms,
                    tokens_per_second=tokens_per_second
                )
            except Exception as e:
                result = BenchmarkResult(
                    test=test,
                    implementation=implementation_name,
                    response="",
                    latency_ms=0,
                    tokens_per_second=0,
                    success=False,
                    error=str(e)
                )
            
            self.results.append(result)
            print(f"Test {test.id} completed for {implementation_name}")
    
    def run_benchmarks_batch(self, implementation_name: str, batch_inference_fn: Callable):
        """
        Run benchmarks in batch mode against a specific implementation.
        
        Args:
            implementation_name: Name of the implementation (e.g., "webllm")
            batch_inference_fn: A function that takes a list of BenchmarkTest objects and returns
                                a list of (response, latency_ms, tokens_per_second) tuples
        """
        try:
            batch_results = batch_inference_fn(self.tests)
            
            # Process the batch results
            for i, (response, latency_ms, tokens_per_second) in enumerate(batch_results):
                if i < len(self.tests):  # Ensure we have a test for this result
                    test = self.tests[i]
                    result = BenchmarkResult(
                        test=test,
                        implementation=implementation_name,
                        response=response,
                        latency_ms=latency_ms,
                        tokens_per_second=tokens_per_second,
                        success=True if latency_ms > 0 else False
                    )
                    self.results.append(result)
                    print(f"Test {test.id} completed for {implementation_name}")
        except Exception as e:
            # If the entire batch fails, create error results for all tests
            for test in self.tests:
                result = BenchmarkResult(
                    test=test,
                    implementation=implementation_name,
                    response="",
                    latency_ms=0,
                    tokens_per_second=0,
                    success=False,
                    error=str(e)
                )
                self.results.append(result)
            print(f"Error running batch benchmarks for {implementation_name}: {e}")
    
    def save_results(self, implementation: InferenceFramework, output_file: str | None = None):
        """Save benchmark results to a JSON file."""
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.benchmark_dir, "results", f"{implementation}_benchmark_results_{timestamp}.json")
        else:
            # If a path is provided, make sure it's under the results directory
            if not os.path.isabs(output_file):
                output_file = os.path.join(self.benchmark_dir, "results", output_file)
        
        # Ensure the results directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        
        print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Run benchmarks against different inference implementations')
    parser.add_argument('--implementation', type=InferenceFramework, choices = list(InferenceFramework), required=True, 
                        help='Implementation to benchmark')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--test-file', type=str, default='tests/simple_benchmark.csv',
                        help='Path to benchmark test file')
    
    # Add WebLLM-specific arguments
    parser.add_argument('--webllm-debug', action='store_true',
                        help='Run WebLLM in debug mode')
    parser.add_argument('--webllm-visible', action='store_true',
                        help='Run WebLLM with visible browser (improves WebGPU support)')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.test_file)
    
    if args.implementation == InferenceFramework.OLLAMA:
        from ollama.run_benchmark import run_ollama_benchmark
        runner.run_benchmarks(InferenceFramework.OLLAMA.value, run_ollama_benchmark)
    elif args.implementation == InferenceFramework.WEBLLM:
        from webllm.run_benchmark import run_webllm_benchmark, run_webllm_benchmark_batch
        # Use the batch version instead of individual test runs
        def run_batch_with_options(tests):
            return run_webllm_benchmark_batch(tests, debug=args.webllm_debug, force_visible=args.webllm_visible)
        runner.run_benchmarks_batch(InferenceFramework.WEBLLM.value, run_batch_with_options)
    elif args.implementation == InferenceFramework.LLAMACPP:
        from llama_cpp.run_benchmark import run_llamacpp_benchmark
        runner.run_benchmarks(InferenceFramework.LLAMACPP.value, run_llamacpp_benchmark)
    elif args.implementation == InferenceFramework.OPENAI:
        from open_ai.run_benchmark import run_openai_benchmark
        runner.run_benchmarks(InferenceFramework.OPENAI.value, run_openai_benchmark)
    
    runner.save_results(InferenceFramework(args.implementation), args.output)

if __name__ == "__main__":
    main() 
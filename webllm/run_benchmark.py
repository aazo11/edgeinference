#!/usr/bin/env python3
"""
Main entry point for WebLLM benchmarking.
This script is designed to be called by the benchmark_runner.py script or run directly for testing.
"""

import os
import sys
import json
import argparse
from typing import List, Tuple, Dict, Any

# Add the parent directory to the path so we can import from the benchmark_runner
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir.endswith('webllm'):
    # If running from inside the webllm directory
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

# Import the browser benchmark utility
from webllm.browser_benchmark import run_browser_benchmark, run_browser_benchmark_batch

def run_webllm_benchmark(test, debug=False, force_visible=False, batch_info=None):
    """
    Main function to be called by benchmark_runner.py for a single test.
    Takes a BenchmarkTest object and returns (response, latency_ms, tokens_per_second).
    
    Args:
        test: A BenchmarkTest object containing prompt, max_tokens, etc.
        debug: Whether to run in debug mode
        force_visible: If True, force the browser to be visible
        batch_info: Optional dictionary with batch information (batch_index, batch_total, batch_id)
        
    Returns:
        Tuple of (response, latency_ms, tokens_per_second)
    """
    print(f"Running WebLLM benchmark for test: {test.id}")
    
    # Run the benchmark and get the results
    response, latency_ms, tokens_per_second = run_browser_benchmark(
        test, 
        debug=debug,
        force_visible=force_visible,
        batch_info=batch_info
    )
    
    # Return the results in the format expected by benchmark_runner.py
    return response, latency_ms, tokens_per_second

def run_webllm_benchmark_batch(tests: List, debug=False, force_visible=False) -> List[Tuple[str, float, float]]:
    """
    Run multiple benchmark tests in a single browser session.
    Takes a list of BenchmarkTest objects and returns a list of result tuples.
    
    Args:
        tests: List of BenchmarkTest objects
        debug: Whether to run in debug mode
        force_visible: If True, force the browser to be visible
        
    Returns:
        List of tuples, each containing (response, latency_ms, tokens_per_second)
    """
    print(f"Running WebLLM batch benchmark for {len(tests)} tests")
    
    # Run all benchmarks in a single browser session
    results = run_browser_benchmark_batch(
        tests, 
        debug=debug,
        force_visible=force_visible
    )
    
    return results

# For standalone testing without using the benchmark runner
if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Run WebLLM benchmark')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--visible', action='store_true', help='Show browser window (improves WebGPU support)')
    parser.add_argument('--prompt', type=str, default="Explain the concept of quantum computing in simple terms.",
                        help='Prompt to use for the benchmark')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for generation')
    parser.add_argument('--batch', action='store_true', help='Run multiple tests in a single browser session')
    args = parser.parse_args()
    
    try:
        from benchmark.benchmark_runner import BenchmarkTest
    except ImportError:
        # Create a simple BenchmarkTest class for standalone testing
        print("Could not import BenchmarkTest from benchmark_runner, using local implementation")
        class BenchmarkTest:
            def __init__(self, id, prompt, max_tokens, temperature, expected_class, notes):
                self.id = id
                self.prompt = prompt
                self.max_tokens = max_tokens
                self.temperature = temperature
                self.expected_class = expected_class
                self.notes = notes
    
    # Print header
    print("WebLLM Benchmarking Test")
    print("=======================\n")
    
    if args.batch:
        # Create multiple test examples for batch testing
        tests = [
            BenchmarkTest(
                id="test1",
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                expected_class="explanation",
                notes="First test"
            ),
            BenchmarkTest(
                id="test2",
                prompt="What are the main advantages of quantum computing?",
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                expected_class="explanation",
                notes="Second test"
            )
        ]
        
        print(f"Running batch of {len(tests)} tests")
        print(f"Debug mode: {args.debug}, Visible mode: {args.visible}")
        print("Please wait for the benchmarks to complete...")
        
        try:
            # Run the batch benchmark
            results = run_webllm_benchmark_batch(
                tests, 
                debug=args.debug, 
                force_visible=args.visible
            )
            
            # Print the results
            print("\n=== BATCH RESULTS ===")
            for i, (response, latency_ms, tokens_per_second) in enumerate(results):
                print(f"\nTest {i+1}: {tests[i].id}")
                print(f"Latency: {latency_ms:.2f} ms")
                print(f"Tokens per second: {tokens_per_second:.2f}")
                print("\n=== RESPONSE ===")
                print(response[:200] + "..." if len(response) > 200 else response)
            
            # Create results directory in the current directory
            results_dir = os.path.join(current_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save the results to a file
            results_file = os.path.join(results_dir, "batch_test_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump([{
                    "test_id": tests[i].id,
                    "prompt": tests[i].prompt,
                    "response": response,
                    "latency_ms": latency_ms,
                    "tokens_per_second": tokens_per_second,
                    "debug_mode": args.debug,
                    "visible_mode": args.visible
                } for i, (response, latency_ms, tokens_per_second) in enumerate(results)], f, indent=2)
            
            print(f"\nResults saved to {results_file}")
            
        except Exception as e:
            print(f"Error running batch benchmark: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Create a single test using command-line arguments
        test = BenchmarkTest(
            id="test1",
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            expected_class="explanation",
            notes="Command-line test"
        )
        
        print(f"Running test with prompt: '{test.prompt}'")
        print(f"Max tokens: {test.max_tokens}, Temperature: {test.temperature}")
        print(f"Debug mode: {args.debug}, Visible mode: {args.visible}")
        print("Please wait for the benchmark to complete...")
        
        try:
            # Run the benchmark with command-line options
            response, latency_ms, tokens_per_second = run_webllm_benchmark(
                test, 
                debug=args.debug, 
                force_visible=args.visible
            )
            
            # Print the results
            print("\n=== RESULTS ===")
            print(f"Latency: {latency_ms:.2f} ms")
            print(f"Tokens per second: {tokens_per_second:.2f}")
            print("\n=== RESPONSE ===")
            print(response)
            
            # Create results directory in the current directory
            results_dir = os.path.join(current_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save the results to a file (only when running as a standalone test)
            results_file = os.path.join(results_dir, "test_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "test_id": test.id,
                    "prompt": test.prompt,
                    "response": response,
                    "latency_ms": latency_ms,
                    "tokens_per_second": tokens_per_second,
                    "debug_mode": args.debug,
                    "visible_mode": args.visible
                }, f, indent=2)
            
            print(f"\nResults saved to {results_file}")
            
        except Exception as e:
            print(f"Error running benchmark: {e}")
            import traceback
            traceback.print_exc() 
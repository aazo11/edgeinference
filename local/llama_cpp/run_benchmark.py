#!/usr/bin/env python3
import requests
import time
import sys
import os
import subprocess
import signal
import json
from typing import Tuple, Optional, List
import atexit
import shutil
import pkg_resources

# Check required dependencies
required_packages = ['requests']
try:
    for package in required_packages:
        pkg_resources.require(package)
except pkg_resources.DistributionNotFound:
    print(f"Error: Required package '{package}' not found. Please install it with:")
    print(f"  pip install -r {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')}")
    sys.exit(1)

# Add parent directory to path to import from benchmark_runner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from benchmark.benchmark_runner import BenchmarkTest

# Default model to use
DEFAULT_MODEL = "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"
# llama.cpp server API endpoint
API_BASE = "http://localhost:8080"
# Default timeout for API requests
TIMEOUT = 120  # seconds

# Global variable to track if we started the server
server_process = None

def check_llama_server_installed() -> bool:
    """
    Check if llama-server is installed and available in the system path.
    
    Returns:
        bool: True if llama-server is found, False otherwise
    """
    llama_server_path = shutil.which("llama-server")
    if llama_server_path:
        print(f"Found llama-server at: {llama_server_path}")
        return True
    else:
        print("ERROR: llama-server not found in system PATH.")
        print("Please install llama.cpp and ensure the llama-server binary is available.")
        print("Installation instructions: https://github.com/ggerganov/llama.cpp")
        return False

def ensure_server_running(model_name: str) -> bool:
    """
    Check if the llama.cpp server is running, and if not, attempt to start it.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        bool: True if server is running, False otherwise
    """
    global server_process
    
    # First check if llama-server is installed
    if not check_llama_server_installed():
        return False
    
    # Check if server is already running
    try:
        print(f"Checking if llama.cpp server is running at {API_BASE}...")
        response = requests.get(f"{API_BASE}/health", timeout=2)
        if response.status_code == 200:
            print("llama.cpp server is already running")
            return True
    except requests.exceptions.RequestException as e:
        print(f"llama.cpp server is not running: {e}")
        print("Attempting to start server...")
    
    # Only start server if we're not already managing a process
    if server_process is None:
        try:
            # Print the command we're about to run
            cmd = ["llama-server", "-hf", model_name, "--host", "0.0.0.0", "--port", "8080"]
            print(f"Starting llama-server with command: {' '.join(cmd)}")
            
            # Start server process with the specified model, but don't capture output
            # Instead, set stdout and stderr to None to allow them to print to console
            server_process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True
            )
            
            # Register cleanup function to kill server on exit
            atexit.register(stop_server)
            
            # Wait for server to start (check health endpoint)
            max_attempts = 30
            print(f"Waiting for server to start (max {max_attempts} attempts, 5 seconds between attempts)...")
            for attempt in range(max_attempts):
                try:
                    print(f"Checking server health (attempt {attempt+1}/{max_attempts})...")
                    response = requests.get(f"{API_BASE}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"llama.cpp server started successfully with model {model_name}")
                        return True
                    elif response.status_code == 503 and "Loading model" in response.text:
                        # Model is still loading, wait longer
                        print(f"Model is still loading: {response.text}")
                    else:
                        print(f"Unexpected server response: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    print(f"Server not responding yet: {e}")
                
                # Check if process is still running
                if server_process.poll() is not None:
                    exit_code = server_process.returncode
                    print(f"Server process exited unexpectedly with code {exit_code}")
                    server_process = None
                    return False
                
                # Wait before retrying
                print(f"Waiting 5 seconds before next attempt...")
                time.sleep(5)
                
            print(f"Failed to start llama.cpp server after {max_attempts} attempts")
            stop_server()  # Clean up the process
            return False
            
        except Exception as e:
            print(f"Error starting llama.cpp server: {e}")
            return False
    
    return False

def stop_server():
    """Stop the llama.cpp server if we started it"""
    global server_process
    
    if server_process is not None:
        print("Shutting down llama.cpp server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
        server_process = None

def run_inference(prompt: str, max_tokens: int, temperature: float) -> Tuple[str, float, float, float]:
    """
    Run inference using llama.cpp server API
    
    Args:
        prompt: The prompt to send
        max_tokens: Maximum tokens to generate
        temperature: Temperature parameter for generation
        
    Returns:
        Tuple containing:
        - generated text
        - latency in milliseconds
        - tokens per second
        - time to first token in milliseconds
    """
    print(f"Starting inference with {max_tokens} max tokens and temperature {temperature}")
    start_time = time.time()
    time_to_first_token = None
    
    try:
        # Use streaming to get time to first token
        print("Sending streaming request to get generation and measure time to first token...")
        response = requests.post(
            f"{API_BASE}/completion",
            json={
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "stream": True,
                "timings_per_token": True
            },
            stream=True,
            timeout=TIMEOUT
        )
        
        # Process the streaming response
        generated_text = ""
        first_token_received = False
        token_count = 0
        last_progress_time = time.time()
        
        if response.status_code == 200:
            print("Server accepted request, receiving tokens...")
            for line in response.iter_lines():
                if not line:
                    continue
                
                # Each line is a JSON object (SSE format)
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    try:
                        data = json.loads(line_str[6:])  # Skip "data: " prefix
                        
                        # Check if this is the first token
                        if not first_token_received:
                            first_token_time = time.time()
                            time_to_first_token = (first_token_time - start_time) * 1000  # in milliseconds
                            first_token_received = True
                            print(f"First token received after {time_to_first_token:.2f} ms")
                        
                        # Accumulate text
                        if 'content' in data:
                            generated_text += data['content']
                            token_count += 1
                        
                        # Print progress every 2 seconds
                        current_time = time.time()
                        if current_time - last_progress_time > 2.0:
                            print(f"Generated {token_count} tokens so far, {len(generated_text)} chars")
                            last_progress_time = current_time
                        
                        # Check if done
                        if data.get('stop', False):
                            print("Generation complete")
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON: {e}")
                        continue
            
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            print(f"Total generation time: {elapsed_ms:.2f} ms")
            
            # Calculate tokens per second based on actual measurements
            tokens_per_second = token_count / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            print(f"Measured {tokens_per_second:.2f} tokens per second (based on {token_count} tokens generated)")
            
            # If we never received a first token timestamp, use the total time
            if time_to_first_token is None:
                time_to_first_token = elapsed_ms
                print(f"No first token timestamp recorded, using total time: {time_to_first_token:.2f} ms")
            
            return generated_text, elapsed_ms, tokens_per_second, time_to_first_token
        else:
            error_msg = f"API Error: {response.status_code} - {response.text}"
            print(error_msg)
            return f"Error: {error_msg}", 0, 0, 0
    
    except Exception as e:
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"Exception during inference: {e}")
        return f"Error: {str(e)}", elapsed_ms, 0, elapsed_ms

def run_llamacpp_benchmark(test: BenchmarkTest) -> Tuple[str, float, float, float]:
    """
    Run a benchmark test using llama.cpp server
    
    Args:
        test: BenchmarkTest object containing the test parameters
    
    Returns:
        Tuple containing:
        - generated text response
        - latency in milliseconds
        - tokens per second
        - time to first token in milliseconds
    """
    # Get model name from environment or use default
    model_name = os.environ.get("LLAMACPP_MODEL", DEFAULT_MODEL)
    print("\n" + "="*80)
    print(f"Running benchmark for test ID: {test.id}")
    print(f"Model: {model_name}")
    print(f"Prompt: {test.prompt[:100]}..." if len(test.prompt) > 100 else f"Prompt: {test.prompt}")
    print(f"Max tokens: {test.max_tokens}, Temperature: {test.temperature}")
    print("="*80 + "\n")
    
    # Ensure the server is running
    if not ensure_server_running(model_name):
        error_msg = f"Error: Could not start llama.cpp server with model {model_name}"
        print(error_msg)
        return error_msg, 0, 0, 0
    
    print("Server is running, sending inference request...")
    
    # Run the inference
    response, latency_ms, tokens_per_second, time_to_first_token = run_inference(
        prompt=test.prompt,
        max_tokens=test.max_tokens,
        temperature=test.temperature
    )
    
    # Print summary of results
    print("\n" + "-"*80)
    print(f"Benchmark complete for test ID: {test.id}")
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Time to first token: {time_to_first_token:.2f} ms")
    print(f"Response length: {len(response)} characters")
    print("-"*80 + "\n")
    
    return response, latency_ms, tokens_per_second, time_to_first_token

if __name__ == "__main__":
    # This allows testing the module independently
    # Example: LLAMACPP_MODEL="path/to/model.gguf" python local/llama_cpp/run_benchmark.py
    
    # Create a simple test
    test = BenchmarkTest(
        id="test",
        prompt="What is the capital of France?",
        max_tokens=100,
        temperature=0.0,
        expected_class="factual",
        notes="Simple test"
    )
    
    # Run the test
    response, latency_ms, tokens_per_second, time_to_first_token = run_llamacpp_benchmark(test)
    
    print(f"Response: {response}")
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Time to first token: {time_to_first_token:.2f} ms")
    
    # Make sure the server is stopped when done
    stop_server() 
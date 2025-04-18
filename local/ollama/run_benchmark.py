#!/usr/bin/env python3
import requests
import time
import sys
import os
from typing import Tuple
import json


# Add parent directory to path to import from benchmark_runner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.benchmark_runner import BenchmarkTest

# Default model to use
DEFAULT_MODEL = "deepseek-r1:7b"
# Ollama API endpoint
API_BASE = "http://localhost:11434/api"

def ensure_model_available(model_name: str) -> bool:
    """
    Check if the model is available locally, and if not, pull it.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        bool: True if model is available, False otherwise
    """
    try:
        # Check if model exists
        response = requests.get(f"{API_BASE}/tags")
        models = response.json().get("models", [])
        
        # Check if our target model is in the list
        model_exists = any(model['name'] == model_name for model in models) if models else False
        
        if not model_exists:
            print(f"Model {model_name} not found locally. Pulling...")
            # Pull the model - this will take time for a 7B model
            pull_response = requests.post(
                f"{API_BASE}/pull",
                json={"name": model_name}
            )
            
            # Ollama pull API doesn't return a final response until complete
            if pull_response.status_code == 200:
                print(f"Successfully pulled {model_name}")
                return True
            else:
                print(f"Failed to pull model: {pull_response.text}")
                return False
        
        return True
    
    except Exception as e:
        print(f"Error checking/pulling model: {e}")
        return False

def run_inference(model_name: str, prompt: str, max_tokens: int, 
                  temperature: float) -> Tuple[str, float, float, float]:
    """
    Run inference using Ollama API
    
    Args:
        model_name: Name of the model to use
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
    start_time = time.time()
    time_to_first_token = None
    
    try:
        # First, try with streaming to get the time to first token
        response = requests.post(
            f"{API_BASE}/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                "stream": True
            },
            stream=True  # Enable streaming in the HTTP request
        )
        
        # Process the streaming response
        generated_text = ""
        first_token_received = False
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if not line:
                    continue
                    
                # Each line is a JSON object
                try:
                    data = json.loads(line.decode('utf-8'))
                    
                    # Check if this is the first token
                    if not first_token_received:
                        first_token_time = time.time()
                        time_to_first_token = (first_token_time - start_time) * 1000  # in milliseconds
                        first_token_received = True
                    
                    # Accumulate text
                    if 'response' in data:
                        generated_text += data['response']
                    
                    # Check if done
                    if data.get('done', False):
                        eval_count = data.get('eval_count', 0)
                        eval_duration = data.get('eval_duration', 0)
                        break
                    
                except json.JSONDecodeError:
                    continue
            
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            
            # If we never received a first token timestamp, use the total time
            if time_to_first_token is None:
                time_to_first_token = elapsed_ms
                
            # Calculate tokens per second
            eval_duration_seconds = eval_duration / 1e9 if eval_duration > 0 else elapsed_ms / 1000
            tokens_per_second = eval_count / eval_duration_seconds if eval_duration_seconds > 0 else 0
            
            return generated_text, elapsed_ms, tokens_per_second, time_to_first_token
        else:
            error_msg = f"API Error: {response.status_code} - {response.text}"
            print(error_msg)
            return f"Error: {error_msg}", elapsed_ms, 0, elapsed_ms
    
    except Exception as e:
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"Exception during inference: {e}")
        return f"Error: {str(e)}", elapsed_ms, 0, elapsed_ms

def run_ollama_benchmark(test: BenchmarkTest) -> Tuple[str, float, float, float]:
    """
    Run a benchmark test using Ollama
    
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
    model_name = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    
    # Ensure the model is available
    if not ensure_model_available(model_name):
        return f"Error: Model {model_name} not available", 0, 0, 0
    
    # Run the inference
    response, latency_ms, tokens_per_second, time_to_first_token = run_inference(
        model_name=model_name,
        prompt=test.prompt,
        max_tokens=test.max_tokens,
        temperature=test.temperature
    )
    
    return response, latency_ms, tokens_per_second, time_to_first_token

if __name__ == "__main__":
    # This allows testing the module independently
    # Example: OLLAMA_MODEL=llama2:7b python ollama/run_benchmark.py
    
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
    response, latency_ms, tokens_per_second, time_to_first_token = run_ollama_benchmark(test)
    
    print(f"Response: {response}")
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Time to first token: {time_to_first_token:.2f} ms") 
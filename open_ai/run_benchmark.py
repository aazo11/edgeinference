#!/usr/bin/env python3
import os
import time
import sys
from typing import Tuple
from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path to import from benchmark_runner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.benchmark_runner import BenchmarkTest

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# Default model to use
DEFAULT_MODEL = "gpt-3.5-turbo"

def run_inference(model_name: str, prompt: str, max_tokens: int, 
                  temperature: float) -> Tuple[str, float, float]:
    """
    Run inference using OpenAI API
    
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
    """
    # Get API key from environment variable
    api_key = os.getenv("OPEN_AI_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables. Please check your .env file.")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    start_time = time.time()
    
    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        
        # Extract the response text
        if response.choices and len(response.choices) > 0:
            generated_text = response.choices[0].message.content
            
            # Calculate tokens per second based on completion tokens
            completion_tokens = response.usage.completion_tokens
            tokens_per_second = (completion_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0
            
            return generated_text, elapsed_ms, tokens_per_second
        else:
            return "No response generated", elapsed_ms, 0
    
    except Exception as e:
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"Exception during inference: {e}")
        return f"Error: {str(e)}", elapsed_ms, 0

def run_openai_benchmark(test: BenchmarkTest) -> Tuple[str, float, float]:
    """
    Run a benchmark test using OpenAI
    
    Args:
        test: BenchmarkTest object containing the test parameters
    
    Returns:
        Tuple containing:
        - generated text response
        - latency in milliseconds
        - tokens per second
    """
    # Get model name from environment or use default
    model_name = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    
    # Run the inference
    response, latency_ms, tokens_per_second = run_inference(
        model_name=model_name,
        prompt=test.prompt,
        max_tokens=test.max_tokens,
        temperature=test.temperature
    )
    
    return response, latency_ms, tokens_per_second

if __name__ == "__main__":
    # This allows testing the module independently
    # Example: OPENAI_MODEL=gpt-3.5-turbo python openai/run_benchmark.py
    
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
    response, latency_ms, tokens_per_second = run_openai_benchmark(test)
    
    print(f"Response: {response}")
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"Tokens per second: {tokens_per_second:.2f}") 
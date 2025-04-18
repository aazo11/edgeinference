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
# Get API key from environment variable - check both possible environment variable names
api_key = os.getenv("OPEN_AI_KEY") 
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please check your .env file.")

# Default model to use
DEFAULT_MODEL = "gpt-4o-mini"

def run_inference(model_name: str, prompt: str, max_tokens: int, 
                  temperature: float) -> Tuple[str, float, float, float]:
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
        - time to first token in milliseconds
    """
   
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    start_time = time.time()
    time_to_first_token = None
    
    try:
        # Call OpenAI API with stream=True to measure time to first token
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        
        # Capture the full response
        generated_text = ""
        first_token_received = False
        
        for chunk in stream:
            if not first_token_received and chunk.choices and len(chunk.choices) > 0:
                # Record time to first token
                first_token_time = time.time()
                time_to_first_token = (first_token_time - start_time) * 1000  # in milliseconds
                first_token_received = True
                
            # Accumulate the response content
            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                generated_text += chunk.choices[0].delta.content
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        
        # If we never received a first token timestamp, use the total time
        if time_to_first_token is None:
            time_to_first_token = elapsed_ms
            
        # Make a non-streaming API call with the same prompt to get accurate token counts
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Calculate tokens per second based on completion tokens
        if hasattr(response, 'usage') and response.usage:
            completion_tokens = response.usage.completion_tokens
            tokens_per_second = (completion_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0
            return generated_text, elapsed_ms, tokens_per_second, time_to_first_token
        else:
            # Fallback if token count is not available
            return generated_text, elapsed_ms, 0, time_to_first_token
    
    except Exception as e:
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"Exception during inference: {e}")
        return f"Error: {str(e)}", elapsed_ms, 0, elapsed_ms  # Use total time as TTFT in case of error

def run_openai_benchmark(test: BenchmarkTest):
    """Run a benchmark test using the OpenAI API."""
    # Get the model from environment variable or use a default
    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    
    client = OpenAI(api_key=api_key)
    
    # Start timing
    start_time = time.time()
    first_token_time = None
    
    # Handle the API parameter change from max_tokens to max_completion_tokens
    # for newer models that require this change
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": test.prompt}],
            max_tokens=test.max_tokens,
            temperature=test.temperature,
            stream=True
        )
        
        # Process the streaming response
        full_response = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
                full_response += chunk.choices[0].delta.content
        
    except Exception as e:
        # If we get an error about max_tokens parameter, try with max_completion_tokens
        if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": test.prompt}],
                max_completion_tokens=test.max_tokens,
                temperature=test.temperature,
                stream=True
            )
            
            # Process the streaming response
            full_response = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_response += chunk.choices[0].delta.content
        else:
            # Re-raise if it's a different error
            raise
    
    # End timing
    end_time = time.time()
    
    # Calculate metrics
    latency_ms = (end_time - start_time) * 1000
    
    # Estimate tokens in response (this is approximate)
    response_tokens = len(full_response.split()) * 1.3  # rough estimate
    
    # Calculate tokens per second
    tokens_per_second = response_tokens / (end_time - (first_token_time or start_time))
    
    # Calculate time to first token
    time_to_first_token = (first_token_time - start_time) * 1000 if first_token_time else None
    
    return full_response, latency_ms, tokens_per_second, time_to_first_token

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
    response, latency_ms, tokens_per_second, time_to_first_token = run_openai_benchmark(test)
    
    print(f"Response: {response}")
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Time to first token: {time_to_first_token:.2f} ms") 
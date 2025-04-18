#!/bin/bash

# Set the model to use (default is bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF)
# If you want to use a local model file, change this to the path to your model
export LLAMACPP_MODEL="bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the project root directory
cd "$SCRIPT_DIR/../.."

# Run the benchmark test directly
python -m local.llama_cpp.run_benchmark

echo "Test completed." 
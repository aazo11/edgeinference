# WebLLM Local Model Benchmarking

This directory contains an implementation for benchmarking locally saved models using WebLLM in a browser environment.

## Setup

1. Install the required dependencies:

```bash
pip install selenium webdriver-manager
```

2. Make sure you have Chrome or Chromium installed on your system.

## How to Use

### Preparing a Local Model

This implementation requires a model in MLC format, which is the format that WebLLM uses. You can download pre-built models from the MLC Model Zoo, or convert your own models using MLC-LLM.

Model files should be in one of the following formats:
- A binary weight file (`.bin`)
- A model configuration file (`.json`)

### Running the Benchmark

#### Using the Benchmark Runner

To run the benchmark using the WebLLM implementation with the benchmark_runner.py script:

```bash
# From the project root directory
python benchmark/benchmark_runner.py --implementation webllm --test-file benchmark/tests/simple_benchmark.csv
```

#### Running a Single Test

You can also run a single test directly without using the benchmark runner:

```bash
# From the project root directory
python webllm/run_benchmark.py
```

This will run a simple test and open a browser window where you can upload a local model file.

### Code Structure

The WebLLM implementation consists of the following files:

- `run_benchmark.py`: Main entry point for running WebLLM benchmarks. Contains the function called by the benchmark runner and a standalone test.
- `browser_benchmark.py`: Utility functions for launching a browser and running benchmarks.
- `build_frontend.py`: Utility to build the WebLLM frontend.
- `benchmark.html`: The HTML page used to run the benchmark in the browser.

### How It Works

When you run the benchmark:

1. The script starts a local HTTP server to serve the benchmark.html page
2. It launches a browser (your default browser)
3. The browser loads the WebLLM engine and allows you to upload a local model file
4. After uploading the model, you can run the benchmark which will:
   - Initialize the WebLLM engine with the local model
   - Run inference on the provided test prompts
   - Measure performance metrics like latency and tokens per second
   - Return the results to the benchmark runner

## Troubleshooting

- **Model Loading Issues**: Make sure your model file is in the correct format supported by WebLLM.
- **Browser Not Starting**: Make sure your default browser is working correctly.
- **Performance Issues**: WebLLM performance depends on your hardware. For best results, use a machine with a good GPU.

## Advanced Configuration

You can modify the code in `browser_benchmark.py` to customize the behavior:
- Adjust timeouts for model loading and inference
- Change the HTTP server port

For WebLLM-specific configuration options, check the [WebLLM documentation](https://mlc.ai/web-llm/). 
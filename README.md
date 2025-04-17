# Edge Inference Benchmarks

This project benchmarks local inference solutions for running Large Language Models (LLMs):

1. **Ollama** - Running models locally via Ollama API
2. **WebLLM** - Running models in the browser using WebLLM with WebGPU acceleration
3. **Llama.cpp** - Running models in the browser using Llama.cpp port
4. **OpenAI** - Running models via OpenAI API (cloud-based)

## Project Structure

```
edge_inference_benchmarks/
├── benchmark/                # All benchmark-related code and data
│   ├── benchmark_runner.py   # Main benchmark orchestration script
│   ├── compare_results.py    # Script to compare benchmark results
│   ├── tests/                # Test data directory
│   │   └── simple_benchmark.csv  # Simple benchmark test cases
│   ├── results/              # Directory for benchmark results (gitignored)
│   └── comparison_results/   # Directory for comparison charts (gitignored)
├── requirements.txt          # Project dependencies
├── ollama/                   # Ollama implementation
│   ├── run_benchmark.py      # Ollama-specific benchmark code
│   └── requirements.txt      # Ollama-specific dependencies
├── openai/                   # OpenAI implementation
│   ├── run_benchmark.py      # OpenAI-specific benchmark code
│   ├── requirements.txt      # OpenAI-specific dependencies
│   └── .env                  # Environment file with OpenAI API key
├── webllm/                   # WebLLM implementation with WebGPU acceleration
│   ├── run_benchmark.py      # WebLLM-specific benchmark code
│   ├── requirements.txt      # WebLLM Python bridge dependencies
│   └── web/                  # Browser-based WebLLM app
│       ├── index.html        # HTML page for WebLLM benchmark
│       ├── js/               # JavaScript code
│       │   └── index.js      # Main WebLLM benchmark logic
│       ├── package.json      # NPM dependencies
│       └── webpack.config.js # Webpack configuration
└── llamacpp/                 # Llama.cpp implementation
    ├── run_benchmark.py      # Llama.cpp-specific benchmark code
    └── requirements.txt      # Llama.cpp-specific dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- Ollama installed locally (for Ollama benchmarks)
- Web browser with WebGPU support (for WebLLM and Llama.cpp benchmarks)
- OpenAI API key (for OpenAI benchmarks)
- Node.js and npm (for WebLLM benchmarks)
- Chrome or Chromium browser (for WebLLM benchmarks)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/edge_inference_benchmarks.git
cd edge_inference_benchmarks
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For Ollama benchmarks, install Ollama:
```bash
# For macOS/Linux:
curl -fsSL https://ollama.com/install.sh | sh
# For Windows: Download from https://ollama.com/download
```

4. For OpenAI benchmarks, make sure your API key is in the openai/.env file:
```
OPEN_AI_KEY=your-api-key-here
```

5. For WebLLM benchmarks, install Node.js and npm if not already installed:
```bash
# For macOS with Homebrew
brew install node

# For Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### Running Benchmarks

#### Ollama Benchmarks

1. Start the Ollama service:
```bash
ollama serve
```

2. Run the benchmark:
```bash
python benchmark/benchmark_runner.py --implementation ollama
```

You can specify a particular model by setting the environment variable:
```bash
OLLAMA_MODEL=llama2:7b python benchmark/benchmark_runner.py --implementation ollama
```

#### OpenAI Benchmarks

Run the benchmark using OpenAI's API:

```bash
python benchmark/benchmark_runner.py --implementation openai
```

You can specify a particular model by setting the environment variable:
```bash
OPENAI_MODEL=gpt-4 python benchmark/benchmark_runner.py --implementation openai
```

#### WebLLM Benchmarks

Run the benchmark using WebLLM in a browser with WebGPU acceleration:

```bash
python benchmark/benchmark_runner.py --implementation webllm
```

You can specify a particular model by setting the environment variable:
```bash
WEBLLM_MODEL=Llama-3.1-8B-Instruct-q4f32_1-MLC python benchmark/benchmark_runner.py --implementation webllm
```


You can also specify a different test file:
```bash
python benchmark/benchmark_runner.py --implementation ollama --test-file tests/custom_benchmark.csv
```

## Benchmark Results

Results will be saved as JSON files in the benchmark/results directory. You can specify an output file using the `--output` parameter:

```bash
python benchmark/benchmark_runner.py --implementation ollama --output my_benchmark_results.json
```

## Comparing Results

You can compare results from different implementations using the comparison script:

```bash
python benchmark/compare_results.py 
```

The script will automatically look for the result files in the benchmark/results directory.
You can also specify a different output directory for comparison charts:

```bash
python benchmark/compare_results.py my_benchmark_results_1.json my_benchmark_results_2.json --output-dir my_comparison
```

This will generate:
- A grouped bar chart for accuracy by test and implementation
- A bar chart for average latency by implementation
- A bar chart for average tokens per second by implementation
- A summary JSON file with the key metrics

All charts will use distinct colors for each implementation for better visual comparison.

## Adding New Tests

To add new test cases, you can:
1. Edit an existing test file like `benchmark/tests/simple_benchmark.csv`
2. Create a new test file in the `benchmark/tests` directory following the same format

The CSV format includes these columns:
- `id`: Unique identifier for the test
- `prompt`: The text prompt to send to the model
- `max_tokens`: Maximum number of tokens to generate
- `temperature`: Temperature parameter for generation (0.0-1.0)
- `expected_class`: Category of the expected response
- `notes`: Additional information about the test

## License

This project is licensed under the MIT License - see the LICENSE file for details.

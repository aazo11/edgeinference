# WebLLM Benchmark Frontend

This is a TypeScript-based frontend for the WebLLM benchmark. It allows users to upload local MLC-format models and run benchmarks in the browser.

## Setup

1. Install dependencies:

```bash
npm install
```

2. Build the project:

```bash
npm run build
```

3. Start the development server:

```bash
npm start
```

## Project Structure

- `src/`: Source code
  - `index.ts`: Main entry point
  - `benchmark.ts`: Benchmark engine
  - `results.ts`: Result handling
  - `types.ts`: TypeScript interfaces
  - `utils.ts`: Utility functions
  - `styles/`: CSS files
  - `index.html`: HTML template
- `public/`: Static assets
- `dist/`: Built files (generated)

## Usage

The benchmark can be launched from the main Python runner, which will start an HTTP server and open the browser with the right parameters, or it can be used directly by:

1. Starting the development server
2. Opening http://localhost:8000 in a browser
3. Uploading a model file
4. Running the benchmark

## Parameters

The following URL parameters are supported:

- `test_file`: Path to a JSON file containing test parameters
- `result_file`: Path where the results should be saved

## Development

To make changes to the frontend:

1. Edit the appropriate files in the `src/` directory
2. Run `npm start` to see your changes with hot reloading
3. Build with `npm run build` when ready 
#!/usr/bin/env python3
import json
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import matplotlib.cm as cm
import glob

def load_results(results_file: str) -> List[Dict[str, Any]]:
    """Load benchmark results from a JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_json_files_from_directory(directory: str) -> List[str]:
    """Get all JSON files from a directory."""
    # Use glob to find all .json files in the directory
    json_files = glob.glob(os.path.join(directory, '*.json'))
    return json_files

def compare_results(results_files: List[str] = None, results_dir: str = None, output_dir: str = None):
    """
    Compare benchmark results from different implementations.
    
    Args:
        results_files: List of JSON files containing benchmark results
        results_dir: Directory containing JSON result files (used if results_files is None)
        output_dir: Directory to save comparison charts
    """
    # Make paths relative to the benchmark directory
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If no results files or directory is specified, use the default results directory
    if results_files is None and results_dir is None:
        results_dir = os.path.join(benchmark_dir, 'results')
    
    # If results_dir is specified but not an absolute path, make it relative to benchmark_dir
    if results_dir is not None and not os.path.isabs(results_dir):
        results_dir = os.path.join(benchmark_dir, results_dir)
    
    # If using a directory, get all JSON files from it
    if results_files is None and results_dir is not None:
        if os.path.exists(results_dir):
            results_files = get_json_files_from_directory(results_dir)
            print(f"Found {len(results_files)} JSON files in {results_dir}")
        else:
            print(f"Warning: Directory {results_dir} does not exist")
            return
    
    # If there are no result files, exit
    if not results_files:
        print("No results files specified or found in directory.")
        return
    
    # If output_dir is not specified, use the default comparison_results directory
    if output_dir is None:
        output_dir = os.path.join(benchmark_dir, 'comparison_results')
    elif not os.path.isabs(output_dir):
        output_dir = os.path.join(benchmark_dir, output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Resolve relative paths for result files
    resolved_result_files = []
    for file in results_files:
        if not os.path.isabs(file):
            # Check if the file exists in the results directory
            results_path = os.path.join(benchmark_dir, 'results', file)
            if os.path.exists(results_path):
                resolved_result_files.append(results_path)
            elif os.path.exists(file):
                resolved_result_files.append(file)
            else:
                print(f"Warning: Could not find results file {file}")
                continue
        else:
            resolved_result_files.append(file)
    
    # Load all results
    all_results = []
    implementations = []
    
    for file in resolved_result_files:
        try:
            results = load_results(file)
            if results:
                all_results.append(results)
                # Extract implementation name from the first result
                impl_name = results[0]['implementation']
                implementations.append(impl_name)
        except Exception as e:
            print(f"Error loading results from {file}: {str(e)}")
    
    if not all_results:
        print("No valid results found in the provided files.")
        return
    
    # Prepare data for comparison
    comparison_data = {
        'implementation': [],
        'test_id': [],
        'prompt': [],
        'latency_ms': [],
        'tokens_per_second': [],
        'time_to_first_token': [],
        # 'success': []
    }
    
    for results in all_results:
        for result in results:
            comparison_data['implementation'].append(result['implementation'])
            comparison_data['test_id'].append(result['test_id'])
            comparison_data['prompt'].append(result['prompt'])
            comparison_data['latency_ms'].append(result['latency_ms'])
            comparison_data['tokens_per_second'].append(result['tokens_per_second'])
            # Add time_to_first_token data, default to None if not available
            time_to_first_token = result.get('time_to_first_token', None)
            comparison_data['time_to_first_token'].append(time_to_first_token)
            # comparison_data['success'].append(1 if result['success'] and result['response'] else 0)
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(comparison_data)
    
    # Calculate overall accuracy for each implementation
    # accuracy_data = df.groupby('implementation')['success'].mean()
    
    # Calculate median latency for each implementation (instead of average)
    latency_data = df.groupby('implementation')['latency_ms'].median()
    
    # Calculate median tokens per second for each implementation (instead of average)
    tps_data = df.groupby('implementation')['tokens_per_second'].median()
    
    # Calculate TTFT metrics: median and max time to first token for each implementation
    # Filter out None values
    ttft_df = df[df['time_to_first_token'].notna()]
    if not ttft_df.empty:
        # Use median instead of mean
        avg_ttft_data = ttft_df.groupby('implementation')['time_to_first_token'].median()
        max_ttft_data = ttft_df.groupby('implementation')['time_to_first_token'].max()
    else:
        avg_ttft_data = pd.Series(dtype='float64')
        max_ttft_data = pd.Series(dtype='float64')
        print("Warning: No time to first token data found in results")
    
    """
    # Generate accuracy bar chart by test_id and implementation
    plt.figure(figsize=(14, 8))
    
    # Pivot the data to get accuracy by test_id and implementation
    accuracy_by_test = df.pivot_table(
        index='test_id', 
        columns='implementation', 
        values='success',
        aggfunc='mean'
    )
    
    # Create a colormap for the implementations
    colors = cm.get_cmap('tab10', len(implementations))
    
    # Plot grouped bar chart for accuracy with distinct colors
    ax = accuracy_by_test.plot(
        kind='bar', 
        figsize=(14, 8), 
        color=[colors(i) for i in range(len(implementations))]
    )
    
    plt.title('Accuracy by Test and Implementation', fontsize=16)
    plt.xlabel('Test ID', fontsize=14)
    plt.ylabel('Accuracy (1 = Success, 0 = Failure)', fontsize=14)
    plt.xticks(rotation=0)
    plt.ylim(0, 1.1)  # Set y-axis limits
    plt.legend(title='Implementation')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)
    
    # Save the chart
    accuracy_chart_path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.tight_layout()
    plt.savefig(accuracy_chart_path)
    plt.close()
    """
    
    # Create a colormap for the implementations
    colors = cm.get_cmap('tab10', len(implementations))
    
    # Generate latency bar chart with distinct colors
    plt.figure(figsize=(12, 6))
    
    # Create color-mapped bars for each implementation
    bars = plt.bar(
        latency_data.index, 
        latency_data.values, 
        alpha=0.8,
        color=[colors(i) for i in range(len(latency_data))]
    )
    
    plt.title('Median Latency by Implementation', fontsize=16)
    plt.xlabel('Implementation', fontsize=14)
    plt.ylabel('Latency (ms)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}',
                ha='center', va='bottom', rotation=0)
    
    # Save the chart
    latency_chart_path = os.path.join(output_dir, 'latency_comparison.png')
    plt.tight_layout()
    plt.savefig(latency_chart_path)
    plt.close()
    
    # Generate tokens per second bar chart with distinct colors
    plt.figure(figsize=(12, 6))
    
    # Create color-mapped bars for each implementation
    bars = plt.bar(
        tps_data.index, 
        tps_data.values, 
        alpha=0.8,
        color=[colors(i) for i in range(len(tps_data))]
    )
    
    plt.title('Median Tokens per Second by Implementation', fontsize=16)
    plt.xlabel('Implementation', fontsize=14)
    plt.ylabel('Tokens per Second', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}',
                ha='center', va='bottom', rotation=0)
    
    # Save the chart
    tps_chart_path = os.path.join(output_dir, 'tps_comparison.png')
    plt.tight_layout()
    plt.savefig(tps_chart_path)
    plt.close()
    
    # Generate time to first token charts if data is available
    ttft_chart_path = None
    max_ttft_chart_path = None
    
    if not avg_ttft_data.empty:
        # Generate average TTFT bar chart
        plt.figure(figsize=(12, 6))
        
        # Create color-mapped bars for each implementation
        bars = plt.bar(
            avg_ttft_data.index, 
            avg_ttft_data.values, 
            alpha=0.8,
            color=[colors(i) for i in range(len(avg_ttft_data))]
        )
        
        plt.title('Median Time to First Token by Implementation', fontsize=16)
        plt.xlabel('Implementation', fontsize=14)
        plt.ylabel('Time (ms)', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.1f}',
                    ha='center', va='bottom', rotation=0)
        
        # Save the chart
        ttft_chart_path = os.path.join(output_dir, 'ttft_comparison.png')
        plt.tight_layout()
        plt.savefig(ttft_chart_path)
        plt.close()
        
        # Generate max TTFT bar chart
        plt.figure(figsize=(12, 6))
        
        # Create color-mapped bars for each implementation
        bars = plt.bar(
            max_ttft_data.index, 
            max_ttft_data.values, 
            alpha=0.8,
            color=[colors(i) for i in range(len(max_ttft_data))]
        )
        
        plt.title('Maximum Time to First Token by Implementation', fontsize=16)
        plt.xlabel('Implementation', fontsize=14)
        plt.ylabel('Time (ms)', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.1f}',
                    ha='center', va='bottom', rotation=0)
        
        # Save the chart
        max_ttft_chart_path = os.path.join(output_dir, 'max_ttft_comparison.png')
        plt.tight_layout()
        plt.savefig(max_ttft_chart_path)
        plt.close()
    
    # Generate summary report
    summary = {
        # 'accuracy': accuracy_data.to_dict(),
        'median_latency_ms': latency_data.to_dict(),
        'median_tokens_per_second': tps_data.to_dict()
    }
    
    # Add TTFT data to summary if available
    if not avg_ttft_data.empty:
        summary['median_time_to_first_token'] = avg_ttft_data.to_dict()
        summary['max_time_to_first_token'] = max_ttft_data.to_dict()
    
    # Save summary to JSON
    summary_path = os.path.join(output_dir, 'comparison_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n===== Benchmark Comparison Summary =====")
    
    """
    print("\nAccuracy:")
    for impl, acc in accuracy_data.items():
        print(f"  {impl}: {acc:.2%}")
    """
    
    print("\nMedian Latency (ms):")
    for impl, lat in latency_data.items():
        print(f"  {impl}: {lat:.2f} ms")
    
    print("\nMedian Tokens per Second:")
    for impl, tps in tps_data.items():
        print(f"  {impl}: {tps:.2f}")
    
    # Print TTFT data if available
    if not avg_ttft_data.empty:
        print("\nMedian Time to First Token (ms):")
        for impl, ttft in avg_ttft_data.items():
            print(f"  {impl}: {ttft:.2f} ms")
        
        print("\nMaximum Time to First Token (ms):")
        for impl, ttft in max_ttft_data.items():
            print(f"  {impl}: {ttft:.2f} ms")
    
    print(f"\nCharts saved to: {output_dir}")
    # print(f"  - {os.path.basename(accuracy_chart_path)}")
    print(f"  - {os.path.basename(latency_chart_path)}")
    print(f"  - {os.path.basename(tps_chart_path)}")
    
    if ttft_chart_path:
        print(f"  - {os.path.basename(ttft_chart_path)}")
    if max_ttft_chart_path:
        print(f"  - {os.path.basename(max_ttft_chart_path)}")
    
    print(f"Summary saved to: {os.path.basename(summary_path)}")

def main():
    parser = argparse.ArgumentParser(description='Compare benchmark results from different implementations')
    
    # Create mutually exclusive group for results files or directory
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--files', type=str, nargs='+', 
                      help='JSON files containing benchmark results')
    group.add_argument('--dir', type=str,
                      help='Directory containing JSON result files (default: benchmark/results)')
    
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save comparison charts')
    args = parser.parse_args()
    
    # If neither files nor dir are specified, use default directory
    if args.files is None and args.dir is None:
        benchmark_dir = os.path.dirname(os.path.abspath(__file__))
        default_results_dir = os.path.join(benchmark_dir, 'results')
        args.dir = default_results_dir
    
    compare_results(results_files=args.files, results_dir=args.dir, output_dir=args.output_dir)

if __name__ == "__main__":
    main() 
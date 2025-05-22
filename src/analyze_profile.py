import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from pathlib import Path

def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def generate_comparative_charts(profile_data: List[Dict], output_dir: str):
    """Generate comparative visualizations across different inputs."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data structures
    all_steps = set()
    input_step_times = {}
    input_token_usage = {}
    
    # Process profile data
    for entry in profile_data:
        input_name = entry['input']
        profile = entry['profile_data']
        
        # Collect step times
        step_times = {step: sum(times) for step, times in profile.items()}
        input_step_times[input_name] = step_times
        all_steps.update(step_times.keys())
    
    # Generate comparative bar chart
    plt.figure(figsize=(15, 8))
    x = np.arange(len(input_step_times))
    width = 0.8 / len(all_steps)
    
    for i, step in enumerate(all_steps):
        step_values = [times.get(step, 0) for times in input_step_times.values()]
        plt.bar(x + i * width, step_values, width, label=step)
    
    plt.xlabel('Input Topics')
    plt.ylabel('Time (seconds)')
    plt.title('Comparative Step Times Across Inputs')
    plt.xticks(x + width * len(all_steps) / 2, list(input_step_times.keys()), rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparative_step_times.png'))
    plt.close()
    
    # Generate box plot for step time distributions
    plt.figure(figsize=(15, 8))
    step_data = []
    step_labels = []
    
    for step in all_steps:
        step_times = [times.get(step, 0) for times in input_step_times.values()]
        if any(step_times):  # Only include steps with non-zero times
            step_data.append(step_times)
            step_labels.append(step)
    
    plt.boxplot(step_data, labels=step_labels)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Time (seconds)')
    plt.title('Distribution of Step Times Across Inputs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_time_distributions.png'))
    plt.close()

def analyze_token_usage(token_data: List[Dict], output_dir: str):
    """Analyze token usage patterns across different inputs."""
    
    # Process token usage data
    input_token_totals = {}
    step_token_totals = {}
    
    for entry in token_data:
        input_name = entry['input']
        token_usage = entry['token_usage']
        
        # Calculate total tokens per input
        total_tokens = sum(sum(data['total_tokens']) for data in token_usage.values())
        input_token_totals[input_name] = total_tokens
        
        # Aggregate tokens by step
        for step, data in token_usage.items():
            if step not in step_token_totals:
                step_token_totals[step] = []
            step_token_totals[step].append(sum(data['total_tokens']))
    
    # Generate token usage comparison chart
    plt.figure(figsize=(12, 6))
    plt.bar(input_token_totals.keys(), input_token_totals.values())
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Total Tokens')
    plt.title('Total Token Usage by Input')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_token_usage.png'))
    plt.close()
    
    # Generate step token usage box plot
    plt.figure(figsize=(15, 8))
    step_data = list(step_token_totals.values())
    plt.boxplot(step_data, labels=list(step_token_totals.keys()))
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Tokens')
    plt.title('Token Usage Distribution by Step')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_token_distributions.png'))
    plt.close()

def main():
    # Configure input/output paths
    profile_dir = "profile_data"  # Default directory
    profile_file = os.path.join(profile_dir, "profile_data.jsonl")
    token_file = os.path.join(profile_dir, "token_usage.jsonl")
    output_dir = os.path.join(profile_dir, "analysis")
    
    # Load data
    profile_data = load_jsonl_data(profile_file)
    token_data = load_jsonl_data(token_file)
    
    # Generate visualizations
    generate_comparative_charts(profile_data, output_dir)
    analyze_token_usage(token_data, output_dir)
    
    print(f"Analysis complete. Visualizations saved to {output_dir}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of inputs analyzed: {len(profile_data)}")
    
    # Calculate and print average execution times
    total_times = []
    for entry in profile_data:
        if 'total_execution_time' in entry['profile_data']:
            total_times.append(entry['profile_data']['total_execution_time'][0])
    
    if total_times:
        print(f"Average execution time: {np.mean(total_times):.2f} seconds")
        print(f"Min execution time: {min(total_times):.2f} seconds")
        print(f"Max execution time: {max(total_times):.2f} seconds")

if __name__ == "__main__":
    main() 
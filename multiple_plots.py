#!/usr/bin/env python3
"""
Script to plot all .npy files by data type.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def plot_results(x_start=None, x_end=None, max_episodes=None):
    
    results_base_dir = "results/"
    
    # Find all .npy files in all Results_*/raw_data/ directories
    npy_files = glob.glob(os.path.join(results_base_dir, "Results_*", "raw_data", "*.npy"))
    
    # Group files by data type
    data_groups = {}
    
    for npy_file in npy_files:
        filename = os.path.basename(npy_file)
        # Extract data type (everything before the last underscore)
        data_type = filename.rsplit('_', 1)[0].replace('.npy', '')
        
        # Extract parameter value from directory name
        dir_name = os.path.basename(os.path.dirname(os.path.dirname(npy_file)))
        param_value = dir_name.replace('Results_', '')
        
        if data_type not in data_groups:
            data_groups[data_type] = {}
                
        # Load the data
        try:
            data = np.load(npy_file)
            data_groups[data_type][param_value] = data
            print(f"Loaded {filename} from {dir_name}")
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
    
    # Create a plot for each data type
    for data_type, param_data in data_groups.items():
        plt.figure(figsize=(10, 6))
        
        # Determine the x-axis limits
        all_data_lengths = [len(data) for data in param_data.values()]
        max_length = max(all_data_lengths) if all_data_lengths else 0
        
        # Set default values for x_start and x_end
        start_idx = x_start if x_start is not None else 0
        
        if max_episodes is not None:
            end_idx = min(start_idx + max_episodes, max_length)
        elif x_end is not None:
            end_idx = min(x_end, max_length)
        else:
            end_idx = max_length
        
        # Ensure valid range
        start_idx = max(0, start_idx)
        end_idx = max(start_idx + 1, end_idx)
        
        print(f"Plotting {data_type} from episode {start_idx} to {end_idx-1}")
        
        # Plot each parameter value
        for param_value, data in sorted(param_data.items()):
            # Slice the data according to the specified range
            data_slice = data[start_idx:end_idx]
            x_values = range(start_idx, start_idx + len(data_slice))
            plt.plot(x_values, data_slice, label=f'α_rew_model = {param_value}', linewidth=2)
        
        plt.xlim(start_idx, end_idx - 1)
        
        # Set x-axis ticks every 50 episodes
        x_ticks = range(start_idx, end_idx, 500)
        plt.xticks(x_ticks, [])  # Pass empty list to hide labels, show only ticks
        plt.xticks(x_ticks)
        
        plt.title(f'{data_type} (Episodes {start_idx}-{end_idx-1})')
        plt.xlabel('Episodes')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Create output directory for comparison plots
        output_dir = "comparison_plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Clean filename for saving (no range info in filename)
        safe_filename = data_type.replace(' ', '_').replace('/', '_')
        save_path = os.path.join(output_dir, f'{safe_filename}_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
        
        # Save statistics to text file
        stats_path = os.path.join(output_dir, f'{safe_filename}_statistics.txt')
        with open(stats_path, 'w') as f:
            f.write(f"Statistics for {data_type}\n")
            f.write(f"Episode range: {start_idx} to {end_idx-1}\n")
            f.write("=" * 50 + "\n\n")
            
            for param_value, data in sorted(param_data.items()):
                data_slice = data[start_idx:end_idx]
                
                # Calculate statistics
                min_val = np.min(data_slice)
                max_val = np.max(data_slice)
                mean_val = np.mean(data_slice)
                std_val = np.std(data_slice)
                median_val = np.median(data_slice)
                total_sum = np.sum(data_slice)
                
                f.write(f"Parameter: α_rew_model = {param_value}\n")
                f.write(f"  Minimum value: {min_val:.4f}\n")
                f.write(f"  Maximum value: {max_val:.4f}\n")
                f.write(f"  Mean value: {mean_val:.4f}\n")
                f.write(f"  Standard deviation: {std_val:.4f}\n")
                f.write(f"  Median value: {median_val:.4f}\n")
                f.write(f"  Total sum: {total_sum:.4f}\n")
                f.write(f"  Data points: {len(data_slice)}\n")
                f.write("-" * 30 + "\n")
        
        print(f"Saved statistics: {stats_path}")
        
        # Show the plot
        plt.show()

if __name__ == "__main__":
    # Examples of usage:
    
    # Plot all episodes (default behavior)
    plot_results()
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import datetime
from logger import get_logger


WINDOW_SIZE = 10

# Initialize logger for this module
logger = get_logger("data")

def create_output_directories_tree(s_name):
    
    print("\n" + "="*50)
    print("CREATING SIM DIRECTORIES TREE")
    print("="*50)
    
    # Ensure the results directory exists
    sim_name = str(s_name)
    sim_dir = os.path.join(os.getcwd(), "simulations",sim_name)
    if os.path.exists(sim_dir):
        print(f"Directory{sim_dir} already exists")
        while True:
            response = input("Do you want to create a new directory? (y/n): ").lower().strip()
            if response in ['y','yes']:
                new_sim_name = input("Please insert the new folder name: ").strip()
                if new_sim_name:
                    sim_name = new_sim_name
                    sim_dir = os.path.join(os.getcwd(), "simulations", sim_name)
                    print(f"Created output directory: {sim_name}")
                    break
                else:
                    print("No valid name!")
            elif response in ['n', 'no']:
                print(f"Using the existing directory name: {sim_name}")
                break
            else:
                print("Please insert 'y' or 'n'")
                
    directories_tree = [
        sim_dir,
        os.path.join(sim_dir,"Q_Tables"),
        os.path.join(sim_dir,"plots"),
        os.path.join(sim_dir,"raw_data"),
        os.path.join(sim_dir,"statistics"),
    ]
    
    for dir in directories_tree:    
        try:
            os.makedirs(dir, exist_ok=True)
            print(f"Created folder (if not exsists): {dir}")
        except Exception as e:
            print(f"Failed to create directory {dir}: {e}")
            raise
    
    return sim_dir

def store_QTable(QTable, folder_path, seed = 0):
    
    final_folder_path = folder_path + "/Q_Tables"
    
    # Save the Q-table to a CSV file
    with open(final_folder_path + "/Q_table_" + str(seed) + ".csv", "w", newline="") as file:
        writer = csv.writer(file)
        for row in QTable:
            row = [np.round(float(x), 7) for x in row]
            writer.writerow(row)
            
            
def plot_data(data,output_dir, title="Data Trend"):
    
    # Plot data
    x_values = np.arange(0, len(data))
        
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, data, color='green')
    plt.xlabel('Episodes')
    plt.ylabel(title)    
    plt.title(title)
    
    # mostra i numeri sull'asse x solo ogni 500 episodi
    total_episodes = len(data)
    major_ticks = np.arange(0, total_episodes, 500)  # Numeri ogni 500 episodi

    ax = plt.gca()  # Get current axis
    ax.set_xticks(major_ticks)  # Imposta le tacche principali
    ax.set_xticklabels([''] * len(major_ticks))  # Rimuove i numeri dalle tacche principali

    # Imposta la griglia orizzontale (per l'asse y)
    ax.yaxis.grid(True, which='major', alpha=0.8)
    ax.yaxis.grid(True, which='minor', alpha=0.4)
    
    # Imposta la griglia verticale (per l'asse x)
    # ax.grid(which='minor', alpha=0.8, axis='x')  # Griglia più visibile per le tacche secondarie
    ax.grid(which='major', alpha=0.8, axis='x')  # Griglia più marcata per le tacche principali
    
    # Limiti dell'asse x esattamente sui dati
    plt.xlim(0, len(data) - 1)
    
    path_to_save_plots = os.path.join(output_dir, "plots")
    
    file_name = title + ".jpg"
    save_path = os.path.join(path_to_save_plots, file_name)
    plt.savefig(save_path)
    plt.close()
    
def store_raw_data(data,output_dir,title="Data Trend"):
    
    path_to_save_raw_data = os.path.join(output_dir, "raw_data")
    
    vector_filename = title + ".npy"
    vector_save_path = os.path.join(path_to_save_raw_data, vector_filename)
    np.save(vector_save_path, data)            
    
def save_statistics(data, output_dir, title="Data Statistics", nr_seeds=0):
    
    data = np.array(data)
    
    statistics_dir = os.path.join(output_dir, "statistics")

    # Calculate descriptive statistics
    stats = {
        'nr_seeds': nr_seeds,
        'count': len(data),
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'var': np.var(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'sum': np.sum(data),
        'range': np.max(data) - np.min(data)
    }
    
    # Add IQR (Interquartile Range)
    stats['iqr'] = stats['q75'] - stats['q25']
    
    # Save statistics to text file
    stats_filename = f"{title}_statistics.txt"
    stats_path = os.path.join(statistics_dir, stats_filename)
    
    with open(stats_path, 'w') as f:
        f.write(f"Descriptive Statistics for: {title}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of Seeds:      {stats['nr_seeds']}\n")
        f.write(f"Data Count:           {stats['count']}\n")
        f.write(f"Minimum Value:        {stats['min']:.6f}\n")
        f.write(f"Maximum Value:        {stats['max']:.6f}\n")
        f.write(f"Range:                {stats['range']:.6f}\n")
        f.write(f"Mean:                 {stats['mean']:.6f}\n")
        f.write(f"Median:               {stats['median']:.6f}\n")
        f.write(f"Standard Deviation:   {stats['std']:.6f}\n")
        f.write(f"Variance:             {stats['var']:.6f}\n")
        f.write(f"First Quartile (Q1):  {stats['q25']:.6f}\n")
        f.write(f"Third Quartile (Q3):  {stats['q75']:.6f}\n")
        f.write(f"Interquartile Range:  {stats['iqr']:.6f}\n")
        f.write(f"Sum:                  {stats['sum']:.6f}\n")
        
        # Special analysis for Student Competence
        if "Student Competence" in title:
            f.write("\n" + "=" * 50 + "\n")
            f.write("STUDENT COMPETENCE VALUE DISTRIBUTION\n")
            f.write("=" * 50 + "\n\n")
            
            # Calculate value frequencies and percentages
            unique_values, counts = np.unique(data, return_counts=True)
            total_count = len(data)
            
            f.write(f"Total data points: {total_count}\n")
            f.write(f"Unique values found: {len(unique_values)}\n\n")
            f.write("Value Distribution:\n")
            f.write("-" * 30 + "\n")
            
            for value, count in zip(unique_values, counts):
                percentage = (count / total_count) * 100
                f.write(f"Value {value:.3f}: {count:4d} occurrences ({percentage:6.2f}%)\n")
            
            f.write("\n" + "-" * 30 + "\n")
            f.write("Summary by competence ranges:\n")
            
            # Group by competence ranges for better interpretation
            ranges = [
                (0.0, 0.2, "Very Low (0.0-0.2)"),
                (0.2, 0.4, "Low (0.2-0.4)"),
                (0.4, 0.6, "Medium (0.4-0.6)"),
                (0.6, 0.8, "High (0.6-0.8)"),
                (0.8, 1.0, "Very High (0.8-1.0)")
            ]
            
            for min_val, max_val, label in ranges:
                mask = (data >= min_val) & (data <= max_val)
                count_in_range = np.sum(mask)
                percentage_in_range = (count_in_range / total_count) * 100
                f.write(f"{label}: {count_in_range:4d} occurrences ({percentage_in_range:6.2f}%)\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    logger.info("Saved statistics to: %s", stats_path)
    return stats

def analyze_data(data,output_dir, title="Data",nr_seeds = 0):
    
    plot_data(data,output_dir, title)
    store_raw_data(data,output_dir, title)
    save_statistics(data,output_dir, title, nr_seeds)
    
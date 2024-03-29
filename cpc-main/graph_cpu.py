import math
import re
from datetime import datetime

import matplotlib.pyplot as plt


# extract specific column from csv file without first row.
def extract_index_column(log_path, index):
    with open(log_path, "r") as file:
        log_path = file.read()
    rows = log_path.split("\n")
    values = []
    for row_index, row in enumerate(rows):
        row = row.strip()
        if row_index == 0:
            continue  # Skip the first row (header)
        if row:
            columns = row.split(",")
            if len(columns) > index:
                values.append(columns[index])

    return values



# convert data from pyjoules log to data vectors
def extract_cpu_data_from_log(log_path):
    with open(log_path, "r") as file:
        log_path = file.read()
        
    timestamps = []
    package_0_energy = []
    package_1_energy = []
    dram_0_energy = []
    dram_1_energy = []
    lines = log_path.strip().split("\n")

    for line in lines:
        parts = line.split("; ")
        entry = {}
        for part in parts:
            key, value = part.split(" : ")
            if key.strip() in ['package_0', 'package_1', 'dram_0', 'dram_1']:
                entry[key.strip()] = float(value.strip())
            elif key.strip() == 'begin timestamp':
                entry[key.strip()] = value.strip()  # Keep timestamp as string if needed for relative calculation
            # Skip 'tag' and 'duration' for numeric conversion, handle them differently if needed
        
        timestamps.append(float(entry["begin timestamp"]))  # Convert timestamp to float here
        package_0_energy.append(entry.get("package_0", 0))  # Use .get to avoid KeyError if key doesn't exist
        package_1_energy.append(entry.get("package_1", 0))
        dram_0_energy.append(entry.get("dram_0", 0))
        dram_1_energy.append(entry.get("dram_1", 0))

    relative_timestamps = [t - timestamps[0] for t in timestamps]
    
    # import pdb; pdb.set_trace();
    #print(relative_timestamps)
    
    return relative_timestamps, package_0_energy, package_1_energy, dram_0_energy, dram_1_energy

def plot_cpu_energy(log_path):
    
    base_name = os.path.basename(log_path)
    name_without_extension = os.path.splitext(base_name)[0]
    
    # Initialize a counter for the filename
    counter = 0
    plot_filename = f"{name_without_extension}_energy_plot.png"
    
    # Define the directory where the plots will be saved
    save_directory = "graphs"
    
    # Construct the full path to check for existence
    save_path = os.path.join(save_directory, plot_filename)
    
    # Check if the file exists and increment the counter until an unused name is found
    while os.path.exists(save_path):
        counter += 1
        plot_filename = f"{name_without_extension}_energy_plot_{counter}.png"
        save_path = os.path.join(save_directory, plot_filename)
    
    timestamps, package_0, package_1, dram_0, dram_1 = extract_cpu_data_from_log(log_path)

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, package_0, label='Package 0 Energy')
    plt.plot(timestamps, package_1, label='Package 1 Energy')
    plt.plot(timestamps, dram_0, label='DRAM 0 Energy')
    plt.plot(timestamps, dram_1, label='DRAM 1 Energy')

    plt.xlabel('Time (s)')
    plt.ylabel('Energy (uJ)')
    plt.title(f'CPU Energy/Time, {save_path}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

import argparse
import os


# Create graphs folder if it doesn't exist
if not os.path.exists("graphs"):
    os.makedirs("graphs")

parser = argparse.ArgumentParser(description="Measure and plot CPU energy consumption for NLP models")
parser.add_argument("log_path", help="Path to the log file containing CPU energy measurements")
args = parser.parse_args()

plot_cpu_energy(args.log_path)

#filtered_data = [x for x in data if abs(x - np.mean(data)) < 4 * np.std(data)]


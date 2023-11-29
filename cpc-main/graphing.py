import math
import re
from datetime import datetime

import matplotlib.pyplot as plt


# extract specific column from csv file without first row.
def extract_index_column(csv_path, index):
    with open(csv_path, 'r') as file:
        csv_data = file.read()
    rows = csv_data.split('\n')
    values = []
    for row_index, row in enumerate(rows):
        row = row.strip()
        if row_index == 0:
            continue  # Skip the first row (header)
        if row:
            columns = row.split(',')
            if len(columns) > index:
                values.append(columns[index])

    return values


# convert data from pyjoules log to data vectors
def extract_data_from_pyjoules(csv_path):
    with open(csv_path, 'r') as file:
        csv_data = file.read()
    durations = []
    gpu_energy_uJs = []
    cpu_energy_uJs = []
    wall_cpu_energy_uJs = []
    timestamps = []
    fixed_timestamps = []
    lines = csv_data.strip().split('\n')

    for line in lines:
        parts = line.split('; ')
        entry = {}
        for part in parts:
            key, value = part.split(' : ')
            entry[key.strip()] = value.strip()

        timestamp, duration = float(entry['begin timestamp']), float(entry['duration'])
        cpu_energy_uJ, gpu_energy_uJ = float(entry['core_0']), float(entry['nvidia_gpu_0'])
        wall_cpu_energy_uJ = float(entry['package_0'])
        timestamps.append(timestamp)
        durations.append(duration)
        gpu_energy_uJs.append(gpu_energy_uJ)
        cpu_energy_uJs.append(cpu_energy_uJ)
        wall_cpu_energy_uJs.append(wall_cpu_energy_uJ)

    for item in timestamps:
        fixed_timestamps.append(item - timestamps[0])

    return fixed_timestamps, timestamps, durations, wall_cpu_energy_uJs, cpu_energy_uJs, gpu_energy_uJs


def print_plot(invidia_smi_path, pyjoules_smi_path):
    # calculating carbon emission

    # ---------data from nvidia-smi-----------

    # power data
    power_data_nvidia = extract_index_column(invidia_smi_path, 2)
    power_data_nvidia = [float(re.findall(r'\d+\.\d+', item)[0]) for item in power_data_nvidia]
    # time data - change format to seconds and init first measure to be  0
    fixed_time_nvidia = [0]
    timestamps_nvidia = []

    time_data_nvidia = extract_index_column(invidia_smi_path, 1)
    timestamp1 = datetime.strptime(time_data_nvidia[0], " %Y/%m/%d %H:%M:%S.%f")
    timestamps_nvidia.append(timestamp1.timestamp())
    for i in range(1, len(time_data_nvidia)):
        timestamp_temp = datetime.strptime(time_data_nvidia[i], " %Y/%m/%d %H:%M:%S.%f")
        fixed_time_nvidia.append((timestamp_temp - timestamp1).total_seconds())
        timestamps_nvidia.append(timestamp_temp.timestamp())

    #plot_1
    plt.plot(fixed_time_nvidia, power_data_nvidia)
    plt.xlabel('Time [sec]')
    plt.ylabel('power consumption [W]')
    plt.title('Power Consumption At Inference (BERT model)')
    plt.xticks(fontsize=8)
    # plt.savefig("graphs/" + 'Power_Consumption_At_Inference_BERT_model_gpu_nvidia-smi.png')
    plt.savefig("graphs/nvs/" + 'pc_inf_bert_gpu.png')
    # plt.show()

    # ----------- data from pyjoules --------------

    fixed_timestamps_pyjoules, timestamps_pyjoules, durations_pyjoules, wall_cpu_energy_uJs_pyjoules, cpu_energy_uJs_pyjoules, gpu_energy_uJs_pyjoules = extract_data_from_pyjoules(
        pyjoules_smi_path)

    # plot_2
    plt.plot(fixed_timestamps_pyjoules, gpu_energy_uJs_pyjoules, color='blue')
    plt.xlabel('Time [sec]')
    plt.ylabel('Energy consumption [uJ]')
    plt.title('Energy Consumption At Inference per Query (BERT model - GPU)')
    plt.xticks(fontsize=8)
    # plt.savefig("graphs/" + 'Energy_Consumption_At_Inference_per_query_BERT_model_gpu_pyjoules.png')
    plt.savefig("graphs/pyj/" + 'ec_inf_pq_bert_gpu.png')
    # plt.show()

    # plot_3
    plt.plot(fixed_timestamps_pyjoules[2:], cpu_energy_uJs_pyjoules[2:], color='blue', label='Line 1')
    plt.plot(fixed_timestamps_pyjoules[2:], wall_cpu_energy_uJs_pyjoules[2:], color='red', label='Line 2')
    plt.plot(fixed_timestamps_pyjoules[2:], gpu_energy_uJs_pyjoules[2:], color='green', label='Line 3')
    plt.xlabel('Time [sec]')
    plt.ylabel('Energy consumption [uJ]')
    plt.title('Energy Consumption At Inference per Query (BERT model - CPU (BLUE) & WALL CPU (RED) & GPU (GREEN)')
    plt.xticks(fontsize=8)
    # plt.savefig("graphs/" + 'Energy_Consumption_At_Inference_per_query_BERT_model_cpu_&_wall_cpu_&_GPU_pyjoules.png')
    plt.savefig("graphs/pyj/" + 'ec_inf_pq_bert_cpugpu.png')
    # plt.show()

    power_data_pyjoules = []
    for i in range(len(durations_pyjoules)):
        power_data_pyjoules.append((gpu_energy_uJs_pyjoules[i] * math.pow(10, -6)) / durations_pyjoules[i])

    plt.plot(fixed_timestamps_pyjoules, power_data_pyjoules, color='blue')
    plt.xlabel('Time [sec]')
    plt.ylabel('power consumption [W]')
    plt.title('Power Consumption At Inference per Query (BERT model - GPU)')
    plt.xticks(fontsize=8)
    # plt.savefig("graphs/" + 'Power_Consumption_At_Inference_per_query_BERT_model_gpu_pyjoules.png')
    plt.savefig("graphs/pyj/" + 'pc_inf_pq_bert_gpu.png')
    # plt.show()

    # ------------ combined power consumption ------------
    plt.plot(timestamps_nvidia, power_data_nvidia, color='blue', label='Line 1')
    plt.plot(timestamps_pyjoules, power_data_pyjoules, color='red', label='Line 2')

    plt.xlabel('Time [sec]')
    plt.ylabel('power consumption [W]')
    plt.title('Power Consumption At Inference per Query (BERT model - GPU)')
    plt.xticks(fontsize=8)
    # plt.savefig("graphs/" + 'Power_Consumption_At_Inference_pyjoules_&_nvidia_data_gpu_only.png')
    plt.savefig("graphs/" + 'pc_inf_bert_gpu.png')
    # plt.show()

    # ---------- check energy values ---------------
    start_time = timestamps_pyjoules[0]
    stop_time = timestamps_pyjoules[len(timestamps_pyjoules) - 1]
    sum_energy = 0
    for i in range(len(timestamps_nvidia)):
        if start_time < timestamps_nvidia[i] < stop_time:
            sum_energy += power_data_nvidia[i] * 1

    print("energy_value_of_nvidia_sim", sum_energy, "J")


import argparse 
import os

# if graphs doesnt exist, create folder
if not os.path.exists("graphs"):
    os.makedirs("graphs")
    
parser = argparse.ArgumentParser(
    prog='calc_emission',
    description='measure power consumption of GPU & CPU for NLP models',
    epilog='Text')

# user specifies:
parser.add_argument('-device')  # GPU/CPU,
parser.add_argument('-task_type')  # text-generation/fill-mask,
parser.add_argument('-interface')  # how to run the task
parser.add_argument('--test',
                    action='store_true')  # whether they're in test mode
args = parser.parse_args()

# everything is at logs/tests/nvout_{device}_{task_type}_{interface}.log or pyj_ot_{device}_{interface}_{task_type}.log
print_plot(f"logs/tests/nvout_{args.device}_{args.task_type}_{args.interface}.log",
           f"logs/tests/pyj_out_{args.device}_{args.interface}_{args.task_type}.log")
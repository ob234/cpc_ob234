#/bin/bash
#
#
# Log for 60 seconds into output.log
#s
# This file takes three arguments, assign them to device, task_type and iterface below
device=$1
task_type=$2
interface=$3
# generate filename as nvout_device_task_type_interface.log below as a variable
nvout=nvout_$device$task_type$interface.log
timestamp=$(date +%s)
nvidia-smi > nvidia_spec$timestamp.log
nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv -l 1 -f $nvout

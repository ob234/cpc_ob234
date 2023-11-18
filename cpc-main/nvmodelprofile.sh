#/bin/bash
#
#
# Log for 60 seconds into output.log
#s

timestamp=$(date +%s)
nvidia-smi > nvidia_spec$timestamp.log
nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv -l 1 -f "nvidia_output_$timestamp.log"

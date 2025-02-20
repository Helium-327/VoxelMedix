#!/bin/bash

# # Directory containing the YAML files
# config_dir="/root/data/code/VoxelMedixsrc/configs/2025_1_15"

# # Loop through all YAML files in the directory
# for config_file in "$config_dir"/*.yaml; do
#     echo "Running with config file: $config_file"
#     python /root/data/code/VoxelMedixsrc/main.py --config "$config_file"
    
#     # Check if the previous command succeeded
#     if [ $? -ne 0 ]; then
#         echo "Error: Failed to run with config file $config_file"
#         continue
#     fi
# done

# echo "All configurations processed successfully."

# Directory containing the YAML files
config_dir="/root/data/code/VoxelMedix/src/configs/2025_2_20"

# Create a log directory to store error logs
log_dir="/root/data/code/VoxelMedixlogs"

mkdir -p $log_dir

# Loop through all YAML files in the directory
for config_file in "$config_dir"/*.yaml; do
    echo "Running with config file: $config_file"
    
    # Extract the base name of the config file (without extension) for the log file name
    config_name=$(basename "$config_file" .yaml)
    
    # Run the Python script and redirect both stdout and stderr to a log file
    python /root/data/code/VoxelMedix/src/main_dw.py --config "$config_file" 
    # Check if the previous command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: Failed to run with config file $config_file"
        # Save the error message to a separate error log file
        echo "Error: Failed to run with config file $config_file" >> "$log_dir/${config_name}_error.log"
        cat "$log_dir/${config_name}.log" >> "$log_dir/${config_name}_error.log"
        continue
    fi
done

echo "All configurations processed successfully."

shutdown -h now
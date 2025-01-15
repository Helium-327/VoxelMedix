#!/bin/bash

# Directory containing the YAML files
config_dir="/root/workspace/VoxelMedix/src/configs/2025_1_15"

# Loop through all YAML files in the directory
for config_file in "$config_dir"/*.yaml; do
    echo "Running with config file: $config_file"
    python /root/workspace/VoxelMedix/src/main.py --config "$config_file"
    
    # Check if the previous command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: Failed to run with config file $config_file"
        continue
    fi
done

echo "All configurations processed successfully."
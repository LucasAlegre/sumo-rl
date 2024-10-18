#!/bin/bash

# 检查是否提供了目标文件夹路径
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 target_directory"
    exit 1
fi

TARGET_DIR="$1"

# 检查目标文件夹是否存在
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

# 重命名文件夹
counter=1
for dir in "$TARGET_DIR"/PPO_sumo_env_*; do
    if [[ -d "$dir" ]]; then
        new_name="$TARGET_DIR/PPO_sumo_env_$counter"
        mv "$dir" "$new_name"
        echo "Renamed directory: $dir -> $new_name"
        ((counter++))
    fi
done

# 重命名basic-variant-state文件
counter=1
for file in "$TARGET_DIR"/basic-variant-state-*.json; do
    if [[ -f "$file" ]]; then
        new_name="$TARGET_DIR/basic-variant-state_$counter.json"
        mv "$file" "$new_name"
        echo "Renamed file: $file -> $new_name"
        ((counter++))
    fi
done

# 重命名experiment_state文件
counter=1
for file in "$TARGET_DIR"/experiment_state-*.json; do
    if [[ -f "$file" ]]; then
        new_name="$TARGET_DIR/experiment_state_$counter.json"
        mv "$file" "$new_name"
        echo "Renamed file: $file -> $new_name"
        ((counter++))
    fi
done

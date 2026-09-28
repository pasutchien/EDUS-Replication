#!/bin/bash
set -e

# EDIT THIS: only the drives you actually need
train_list=("2013_05_28_drive_0003_sync"
            "2013_05_28_drive_0007_sync"
            "2013_05_28_drive_0010_sync")
cam_list=("00" "01")

root_dir=KITTI-360
data_2d_dir=data_2d_raw
base_url=https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_2d_raw

mkdir -p "$root_dir/$data_2d_dir"
cd "$root_dir"

for sequence in "${train_list[@]}"; do
    for camera in "${cam_list[@]}"; do
        zip_file=${sequence}_image_${camera}.zip
        echo ">>> $zip_file"
        curl -L -C - -o "$zip_file" "$base_url/$zip_file"   # -C - resumes if interrupted
        unzip -q -n -d "$data_2d_dir" "$zip_file"           # -n = don't re-extract existing
        rm "$zip_file"                                       # free space immediately
    done
done

zip_file=data_timestamps_perspective.zip
curl -L -C - -o "$zip_file" "$base_url/$zip_file"
unzip -q -n -d "$data_2d_dir" "$zip_file"
rm "$zip_file"

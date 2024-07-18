'''
This script demonstrates how to visualize demonstration data in the dataset.
User define:
    DATASET_NAME: the name of the dataset
Videos will be saved in the video folder under the VLA project folder.
'''

import os
import sys

# Add VLA_DIR to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), '../')))

# Add LIBERO to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), '../external/LIBERO')))

from libero.libero import benchmark, get_libero_path
import yaml
import torch
import matplotlib.pyplot as plt
from PIL import Image
from easydict import EasyDict
from transformers import AutoModelForVision2Seq, AutoProcessor
from utils.LIBERO_utils import get_task_names, extract_task_info
import cv2

# Check dataset path
BENCHMARK_PATH = get_libero_path("benchmark_root")
DATASET_BASE_PATH = get_libero_path("datasets")
VLA_BASE_PATH = os.path.abspath(os.path.join(os.path.dirname('__file__'), '../'))
print("Default benchmark root path: ", BENCHMARK_PATH)
print("Default dataset root path: ", DATASET_BASE_PATH)

# Select a dataset
DATASET_NAME = "libero_10"
FILTER_KEY = None  # Set filter key if needed, e.g., "valid" for validation
VERBOSE = True
dataset_path_demo = os.path.join(DATASET_BASE_PATH, DATASET_NAME)
print(f"Dataset path: {dataset_path_demo}")

# Video folder
video_folder = os.path.join(VLA_BASE_PATH, "videos")

# Load dataset
# use a dictionary to store demonstration data for each task
demonstration_data = {}
# get all task names in the dataset
task_names_demo = get_task_names(dataset_path_demo)
# get demonstration data for each task

for task_name_demo in task_names_demo:
    print(f"Loading demonstration data for task: {task_name_demo}")
    [language_instruction, actions_batch, images_batch] = extract_task_info(dataset_path_demo, task_name_demo, filter_key=FILTER_KEY, verbose=VERBOSE)
    demonstration_data[task_name_demo] = [language_instruction, actions_batch, images_batch]
    
print(images_batch.shape) # 50 (50 demos)
print(images_batch[0].shape) # (103, 128, 128, 3) (103 frames, 128x128 pixels, 3 channels)

# create folders with language instructions under video_folder
for task_name_demo in task_names_demo:
    task_folder = os.path.join(video_folder, task_name_demo)
    os.makedirs(task_folder, exist_ok=True)
    print(f"Created folder: {task_folder}")

# save videos using cv2
for task_name_demo in task_names_demo:
    task_folder = os.path.join(video_folder, task_name_demo)
    language_instruction, actions_batch, images_batch = demonstration_data[task_name_demo]
    for i in range(len(images_batch)):
        frames = images_batch[i]
        video_path = os.path.join(task_folder, f"{i}.mp4")
        print(f"Saving video: {video_path}")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (128, 128))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
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
import numpy as np

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
VIDEO_FOLDER = os.path.join(VLA_BASE_PATH, "demo_record/videos")
ACTION_FOLDER = os.path.join(VLA_BASE_PATH, "demo_record/actions")

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
    
# print(images_batch.shape) # 50 (50 demos)
# print(images_batch[0].shape) # (103, 128, 128, 3) (103 frames, 128x128 pixels, 3 channels)

# create folders with language instructions under VIDEO_FOLDER
for task_name_demo in task_names_demo:
    task_folder_video = os.path.join(VIDEO_FOLDER, task_name_demo)
    task_folder_action = os.path.join(ACTION_FOLDER, task_name_demo)
    os.makedirs(task_folder_video, exist_ok=True)
    os.makedirs(task_folder_action, exist_ok=True)
    print(f"Created folder: {task_folder_video}")
    print(f"Created folder: {task_folder_action}")

# save videos using cv2
for task_name_demo in task_names_demo:
    task_folder_video = os.path.join(VIDEO_FOLDER, task_name_demo)
    task_folder_action = os.path.join(ACTION_FOLDER, task_name_demo)
    language_instruction, actions_batch, images_batch = demonstration_data[task_name_demo]
    for i in range(len(images_batch)):
        frames = images_batch[i]
        video_path = os.path.join(task_folder_video, f"{i}.mp4")
        action_path = os.path.join(task_folder_action, f"{i}.npy")
        print(f"Saving video: {video_path}")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (128, 128))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Saving action: {action_path}")
        print(actions_batch[i].shape)
        np.save(action_path, actions_batch[i])
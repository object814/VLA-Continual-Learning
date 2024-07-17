import os
import h5py
import json
import numpy as np
import cv2

def get_task_names(dataset_path):
    '''
    Get the names of the tasks in the dataset.
    Input: dataset_path (str) - path to the dataset (e.g. LIBERO_OBJECT)
    Output: task_descriptions (list) - list of task descriptions
    '''
    task_names = []
    for f in os.listdir(dataset_path):
        if f.endswith('.hdf5'):
            task_name = f.split('.')[0]
            if task_name not in task_names:
                task_names.append(task_name)
    return task_names

def extract_task_info(dataset_path, task_name, filter_key=None, verbose=False):
    '''
    Get the task information from the dataset, including language instruction, actions, and images.
    Input:  dataset_path (str) - path to the dataset (e.g. LIBERO_OBJECT)
            task_description (str) - task description (e.g. "pick_and_place")
    Output: language_instruction (str) - language instruction for the task
            actions_batch (list) - list of actions for each demonstration episode for the task
            images_batch (list) - list of images for each demonstration episode for the task
    '''
    file_path = os.path.join(dataset_path, f"{task_name}.hdf5")
    with h5py.File(file_path, "r") as f:
        if filter_key is not None:
            print(f"NOTE: using filter key {filter_key}")
            demos = sorted([elem.decode("utf-8") for elem in np.array(f[f"mask/{filter_key}"])])
        else:
            demos = sorted(list(f["data"].keys()))
            
            # extract filter key information
            if "mask" in f:
                all_filter_keys = {}
                for fk in f["mask"]:
                    fk_demos = sorted(
                        [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(fk)])]
                    )
                    all_filter_keys[fk] = fk_demos
        
        inds = np.argsort([int(elem.split('_')[1]) for elem in demos])
        demos = [demos[i] for i in inds]
        
        traj_lengths = []
        action_min = np.inf
        action_max = -np.inf
        actions_batch = []
        images_batch = []
        
        for ep in demos:
            traj_lengths.append(f[f"data/{ep}/actions"].shape[0])
            action_min = min(action_min, np.min(f[f"data/{ep}/actions"][()]))
            action_max = max(action_max, np.max(f[f"data/{ep}/actions"][()]))
            
            # Extract actions and observations
            episode_actions = []
            episode_images = []
            num_samples = f[f"data/{ep}"].attrs["num_samples"]
            for i in range(num_samples):
                ee_states = f[f"data/{ep}/obs/ee_states"][i]
                gripper_states = f[f"data/{ep}/obs/gripper_states"][i]
                ee_states = ee_states.tolist()
                gripper_states = gripper_states.tolist()
                episode_actions.append(np.array([ee_states, gripper_states]))
                
                obs_img = f[f"data/{ep}/obs/agentview_rgb"][i]
                episode_images.append(obs_img)
            
            episode_actions = np.array(episode_actions)
            
            actions_batch.append(episode_actions)
            images_batch.append(episode_images)
        
        traj_lengths = np.array(traj_lengths)
        
        problem_info = json.loads(f["data"].attrs["problem_info"])
        language_instruction = "".join(problem_info["language_instruction"])
        
        print("")
        print(f"Task: {task_name}")
        print(f"Total transitions: {np.sum(traj_lengths)}")
        print(f"Total trajectories: {traj_lengths.shape[0]}")
        print(f"Trajectory length mean: {np.mean(traj_lengths)}")
        print(f"Trajectory length std: {np.std(traj_lengths)}")
        print(f"Trajectory length min: {np.min(traj_lengths)}")
        print(f"Trajectory length max: {np.max(traj_lengths)}")
        print(f"Action min: {action_min}")
        print(f"Action max: {action_max}")
        print(f"Language instruction: {language_instruction.strip()}")

        if "mask" in f and filter_key is None:
            all_filter_keys = {}
            for fk in f["mask"]:
                fk_demos = sorted([elem.decode("utf-8") for elem in np.array(f[f"mask/{fk}"])])
                all_filter_keys[fk] = fk_demos
            
            print("==== Filter Keys ====")
            for fk in all_filter_keys:
                print(f"Filter key {fk} with {len(all_filter_keys[fk])} demos")
        else:
            print("No filter keys")
        
        if verbose:
            if "mask" in f and filter_key is None:
                print("==== Filter Key Contents ====")
                for fk in all_filter_keys:
                    print(f"Filter key {fk} with {len(all_filter_keys[fk])} demos: {all_filter_keys[fk]}")
            print("")
        
        env_meta = json.loads(f["data"].attrs["env_args"])
        print("==== Env Meta ====")
        print(json.dumps(env_meta, indent=4))
        print("")
        
        return language_instruction.strip(), actions_batch, images_batch
import os
import h5py
import json
import numpy as np
import cv2
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import torch

def safe_device(x, device="cpu"):
    if device == "cpu":
        return x.cpu()
    elif "cuda" in device:
        if torch.cuda.is_available():
            return x.to(device)
        else:
            return x.cpu()

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
                
                # Ensure that states are numpy arrays of the same length
                ee_states = np.array(ee_states).flatten()
                gripper_states = np.array(gripper_states).flatten()
                
                # Stack ee_states and gripper_states horizontally
                episode_actions.append(np.concatenate([ee_states, gripper_states]))

                obs_img = f[f"data/{ep}/obs/agentview_rgb"][i]
                episode_images.append(obs_img)

            episode_actions = np.array(episode_actions)
            episode_images = np.array(episode_images)
            
            actions_batch.append(episode_actions)
            images_batch.append(episode_images)

        # Convert lists to numpy arrays with dtype=object if they contain ragged sequences
        actions_batch = np.array(actions_batch, dtype=object)
        images_batch = np.array(images_batch, dtype=object)
        
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
    
def extract_env_obs(obs, device_id):
    '''
    Extract RGB image from the environment observation.
    Input: 
        obs (dict) - environment observation
        device_id (int) - device ID
    Output: rgb_img (np.ndarray) - RGB image
    '''
    # Modality OpenVLA need (copied from LIBERO)
    modality = {
        "rgb": ["agentview_rgb"],
        "depth": [],
        "low_dim": ["gripper_states", "joint_states"]
    }
    obs_key_mapping = {
        "agentview_rgb": "agentview_image",
        "eye_in_hand_rgb": "robot0_eye_in_hand_image",
        "gripper_states": "robot0_gripper_qpos",
        "joint_states": "robot0_joint_pos"
    }
    # LIBERO obs modalities
    # obs:
    # modality:
    #     rgb: ["agentview_rgb", "eye_in_hand_rgb"]
    #     depth: []
    #     low_dim: ["gripper_states", "joint_states"]
    # LIBERO obs_key_mapping
    # obs_key_mapping:
    #     agentview_rgb: agentview_image
    #     eye_in_hand_rgb: robot0_eye_in_hand_image
    #     gripper_states: robot0_gripper_qpos
    #     joint_states: robot0_joint_pos
    
    env_num = len(obs)
    data = {}
    all_obs_keys = []
    for modality_name, modality_list in modality.items():
        for obs_name in modality_list:
            data[obs_name] = []
        all_obs_keys += modality_list

    for k in range(env_num):
        for obs_name in all_obs_keys:
            data[obs_name].append(
                ObsUtils.process_obs(
                    torch.from_numpy(obs[k][obs_key_mapping[obs_name]]),
                    obs_key=obs_name,
                ).float()
            )

    for key in data:
        data[key] = torch.stack(data[key])

    data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=device_id))
    return data
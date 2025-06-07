import os
from PIL import Image
import torch
import torchvision.transforms.functional as TF

def read_image(image_path):
    """Reads an image from the given path and returns a PIL Image."""
    return Image.open(image_path).convert("RGB")

def get_scared_file_pairs(root_path, training=True):
    """
    Gets all the data (left, right images, and optionally ground truth) from datasets
    based on whether it's for training or testing.

    Args:
        root_path (str): The base directory where the dataset folders are located.
        training (bool): If True, processes datasets up to 'dataset_5' and returns
                         left and right images. If False, processes 'dataset_6'
                         and 'dataset_7' and returns left images, right images,
                         and ground truth depth maps.

    Returns:
        tuple: A tuple containing lists of image paths.
               If training is True: (left_images, right_images)
               If training is False: (left_images, right_images, ground_truth_images)
    """
    left_images = []
    right_images = []
    ground_truth_images = [] # This list will be populated only when training is False

    # Determine which datasets to process based on the 'training' flag
    if training:
        # For training, process datasets from 'dataset_1' to 'dataset_5'
        dataset_folders_to_process = [f'dataset_{i}' for i in range(1, 6)]
    else:
        # For testing, process 'dataset_6' and 'dataset_7'
        dataset_folders_to_process = ['dataset_6', 'dataset_7']

    # Iterate through the specified dataset folders
    for dataset_folder_name in sorted(os.listdir(root_path)):
        # Only process folders that are in our target list
        if dataset_folder_name not in dataset_folders_to_process:
            continue

        dataset_path = os.path.join(root_path, dataset_folder_name)
        print(f"Processing dataset: {dataset_folder_name} at {dataset_path}")

        if not os.path.isdir(dataset_path):
            print(f'Warning: Dataset path not found or not a directory: {dataset_path}. Skipping.')
            continue

        # Iterate through keyframe folders within each dataset
        for keyframe_folder in sorted(os.listdir(dataset_path)):
            keyframe_path = os.path.join(dataset_path, keyframe_folder)
            print(f"  Processing keyframe: {keyframe_folder} at {keyframe_path}")

            if not os.path.isdir(keyframe_path):
                print(f'  Warning: Keyframe path not found or not a directory: {keyframe_path}. Skipping.')
                continue

            # Path for left and right images (always 'rectified_video_frame')
            rectified_video_frame_path = os.path.join(keyframe_path, 'rectified_video_frame')
            if not os.path.isdir(rectified_video_frame_path):
                print(f'  Warning: rectified_video_frame path not found: {rectified_video_frame_path}. Skipping keyframe images.')
                continue

            left_dir = os.path.join(rectified_video_frame_path, 'left')
            right_dir = os.path.join(rectified_video_frame_path, 'right')

            # Populate left and right images lists
            if os.path.isdir(left_dir):
                for filename in sorted(os.listdir(left_dir)):
                    left_path = os.path.join(left_dir, filename)
                    left_images.append(left_path)
            else:
                print(f'  Warning: Left images directory not found: {left_dir}')

            if os.path.isdir(right_dir):
                for filename in sorted(os.listdir(right_dir)):
                    right_path = os.path.join(right_dir, filename)
                    right_images.append(right_path)
            else:
                print(f'  Warning: Right images directory not found: {right_dir}')

            # If not in training mode, also collect ground truth depth maps
            if not training:
                rectified_depth_maps_path = os.path.join(keyframe_path, 'rectified_depth_maps')
                left_dir_gt = os.path.join(rectified_depth_maps_path, 'left')
                if os.path.isdir(left_dir_gt):
                    for filename in sorted(os.listdir(left_dir_gt)):
                        depth_map_path = os.path.join(rectified_depth_maps_path, filename)
                        ground_truth_images.append(depth_map_path)
                else:
                    print(f'  Warning: Ground truth directory not found: {rectified_depth_maps_path}. Skipping ground truth for this keyframe.')

    # Return based on the training flag
    if training:
        return left_images, right_images
    else:
        return left_images, right_images, ground_truth_images
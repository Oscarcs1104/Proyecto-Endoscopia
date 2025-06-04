import os
from PIL import Image
import torch
import torchvision.transforms.functional as TF

def read_image(image_path):
    """Reads an image from the given path and returns a PIL Image."""
    return Image.open(image_path).convert("RGB")

def get_scared_file_pairs(root_path):
    """
    Gets all the data, the right and left image from all dataset. the drive
    """
    left_images = []
    right_images = []

    for dataset_folder in sorted(os.listdir(root_path)):
        print(dataset_folder)
        dataset_path = os.path.join(root_path, dataset_folder)
        print(dataset_path)
        #if not os.path.isdir(dataset_path):
        #  print('Not found')
        #  continue
        for keyframe_folder in sorted(os.listdir(dataset_path)):
            print("keyframe " +keyframe_folder)
            keyframe_path = os.path.join(dataset_path, keyframe_folder, 'rectified_video_frame')
            #print(keyframe_path)
            #if not os.path.isdir(keyframe_path):
            #  print('Not found')
            #  continue

            left_dir = os.path.join(keyframe_path, 'left')
            right_dir = os.path.join(keyframe_path, 'right')
            print(left_dir)

            for filename in sorted(os.listdir(left_dir)):
                  left_path = os.path.join(left_dir, filename)
                  left_images.append(left_path)
            for filename in sorted(os.listdir(right_dir)):
                  right_path = os.path.join(right_dir, filename)
                  right_images.append(right_path)

    return left_images, right_images
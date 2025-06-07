
import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, random_split


from core.utils.utils_dataloader import get_scared_file_pairs
from core.utils.utils_dataloader import read_image

class RandomFlip:
    """Random horizontal or vertical flip"""
    def __init__(self, horizontal=True, vertical=True):
        self.horizontal = horizontal
        self.vertical = vertical

    def __call__(self, img):
        if self.horizontal and random.random() < 0.5:
            img = F.hflip(img)
        if self.vertical and random.random() < 0.5:
            img = F.vflip(img)
        return img

class SCARED_IGEV(Dataset):
    def __init__(self, root_path, training=True):
        """
        Args:
            root_path (str): Root path to the SCARED dataset.
            training (bool): Whether the dataset is for training or evaluation.
        """
        if self.training:
            self.left_paths, self.right_paths = get_scared_file_pairs(root_path, training=training)
            self.gt_paths = None # No ground truth needed for training
        else:
            self.left_paths, self.right_paths, self.gt_paths = get_scared_file_pairs(root_path, training=training)


        self.transform_base  = transforms.Compose([
            transforms.Resize((256, 320)),
            transforms.ToTensor()
        ])
        self.transform_train  = transforms.Compose([
            RandomFlip(horizontal=True, vertical=True),
            transforms.ColorJitter(
                brightness=(0.8, 1.4),
                contrast=(0.8, 1.4),
                saturation=(0.8, 1.4),
                hue=(-0.1, 0.1)
            )
        ])
        # Define a transform for ground truth images (usually just ToTensor)
        self.transform_gt = transforms.Compose([
            transforms.Resize((256, 320)), # Ensure ground truth matches image size
            transforms.ToTensor()
        ])
        """
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]
        )
        """

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_img = read_image(self.left_paths[idx])
        right_img = read_image(self.right_paths[idx])

        # Apply base transforms
        left_tensor = self.transform_base(left_img)
        right_tensor = self.transform_base(right_img)

        if self.training:

            left_aug = left_tensor.clone()
            left_aug = self.transform_train(left_aug)
            return left_aug, left_tensor, right_tensor

        else:

            gt_img = read_image(self.gt_paths[idx])
            gt_tensor = self.transform_gt(gt_img)

            left_aug = left_tensor.clone()
            return left_tensor, right_tensor, gt_tensor


def get_scared_dataloader(args, train=True):
    dataset = SCARED_IGEV(args.data_path, training=train)

    if train:
        total_size = len(dataset)
        train_size = int(0.70 * total_size)
        val_size = total_size - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size_evaluation,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )
        return train_loader, val_loader, total_size
    else:
        # For testing, return a single DataLoader for the entire dataset
        test_loader = DataLoader(
            dataset,
            batch_size=args.batch_size_evaluation, # Use evaluation batch size
            shuffle=False, # Do not shuffle test data
            num_workers=4,
            drop_last=False # Do not drop last batch for evaluation
        )
        return test_loader, len(dataset) # Return the test loader and total size

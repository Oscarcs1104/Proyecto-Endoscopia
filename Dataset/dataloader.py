
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
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

class StereoColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        # Igual firma que ColorJitter
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue

    def __call__(self, left, right):
        _, brightness_factor, contrast_factor, saturation_factor, hue_factor = transforms.ColorJitter.get_params(
            self.brightness, 
            self.contrast,
            self.saturation, 
            self.hue)
        
        left  = F.adjust_brightness(left, brightness_factor)
        right = F.adjust_brightness(right, brightness_factor)

        left  = F.adjust_contrast(left, contrast_factor)
        right = F.adjust_contrast(right, contrast_factor)

        left  = F.adjust_saturation(left, saturation_factor)
        right = F.adjust_saturation(right, saturation_factor)

        left  = F.adjust_hue(left, hue_factor)
        right = F.adjust_hue(right, hue_factor)

        return left, right

class SCARED_IGEV(Dataset):
    def __init__(self, root_path, training=True):
        """
        Args:
            root_path (str): Root path to the SCARED dataset.
            training (bool): Whether the dataset is for training or evaluation.
        """

        self.training = training
        
        if self.training:
            self.left_paths, self.right_paths = get_scared_file_pairs(root_path, training=training)
            self.gt_paths = None # No ground truth needed for training
        else:
            self.left_paths, self.right_paths, self.gt_paths = get_scared_file_pairs(root_path, training=training)


        self.transform_base  = transforms.Compose([
            transforms.Resize((256, 320)),
            transforms.ToTensor()
        ])

        """
        self.transform_train  = transforms.Compose([
            #RandomFlip(horizontal=True, vertical=True),
            transforms.ColorJitter(
                brightness=(0.7, 1.4),
                contrast=(0.7, 1.4),
                saturation=(0.7, 1.4),
                hue=(-0.1, 0.1)
            )
        ])
        """
        """
        self.stereo_jitter = StereoColorJitter(
            brightness=(0.9, 1.0),
            contrast=(0.9, 1.0),
            saturation=(0.9, 1.0),
            hue=(0, 0)
        )
        """
        
        self.transform_gt = transforms.Compose([
            transforms.Resize((256, 320)), # Ensure ground truth matches image size
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_img = read_image(self.left_paths[idx])
        right_img = read_image(self.right_paths[idx])

        # Apply base transforms
        left_tensor = self.transform_base(left_img)
        right_tensor = self.transform_base(right_img)

        if self.training:
            #left_tensor, right_tensor = self.stereo_jitter(left_tensor, right_tensor)
            return left_tensor, right_tensor

        else:
            gt_img = read_image(self.gt_paths[idx])
            gt_tensor = self.transform_gt(gt_img)

            return left_tensor, right_tensor, gt_tensor


def get_scared_dataloader(args, train=True):
    dataset = SCARED_IGEV(args.data_path, training=train)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    return dataloader


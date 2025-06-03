from core.utils.frame_utils import get_scared_file_pairs
from core.utils.frame_utils import read_image

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
        self.training = training
        self.left_paths, self.right_paths = get_scared_file_pairs(root_path)


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

            left_aug = left_tensor.clone()
            return left_aug, left_tensor, right_tensor


def get_scared_dataloader(args, train=True):

    dataset = SCARED_IGEV(args.data_path, training=train)

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
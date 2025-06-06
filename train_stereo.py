import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from core.igev_stereo_lora import IGEVStereoLoraModel
from Dataset.dataloader import get_scared_dataloader



def photometric_loss(img1, img2, mask=None, alpha=0.85, eps=1e-8):
    """
    Photometric loss combining SSIM and L1.

    Args:
        img1, img2: [B, C, H, W] (RGB images)
        mask: [B, 1, H, W] (optional, weights for pixels)
        alpha: Weight for SSIM vs L1 (0.85 means 85% SSIM, 15% L1)
        eps: Small value to avoid division by zero
    Returns:
        Scalar loss value
    """
    # Ensure images are in [0,1] range for SSIM
    if img1.max() > 1 or img2.max() > 1:
        img1 = img1 / 255.0
        img2 = img2 / 255.0


    ssim_val = ssim(img1, img2, data_range=1.0, size_average=False) 

    ssim_loss = 1 - ssim( img1, img2, data_range=1.0, 
    size_average=True)

    # Calculate L1 (average over channels)
    l1_loss = F.l1_loss(img1, img2, reduction='none').mean(dim=1, keepdim=True)  # [B, 1, H, W]

    # Combine losses
    combined_loss = alpha * ssim_loss + (1 - alpha) * l1_loss  # [B, 1, H, W]
    #    combined_loss = alpha * ssim_loss.view(-1, 1, 1, 1) + (1 - alpha) * l1_loss  # [B, 1, H, W]

    # Apply mask if provided
    if mask is not None:
        valid_pixels = mask.sum() + eps
        loss = (combined_loss * mask).sum() / valid_pixels
    else:
        loss = combined_loss.mean()

    return loss


def stereo_warp(img, disp):
    """
    Warps an image (right view) to the left view using horizontal disparity.

    Args:
        img: Tensor [B, C, H, W] (input image to warp)
        disp: Tensor [B, 1, H, W], horizontal disparity (positive values)
    Returns:
        Warped image: [B, C, H, W]
        Valid mask:   [B, 1, H, W] (1 where warping was valid)
    """
    B, C, H, W = img.size()
    device = img.device

    # Create mesh grid
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device=device),
                                  torch.arange(H, device=device),
                                  indexing='xy')
    grid_x = grid_x.unsqueeze(0).expand(B, H, W)  # [B, H, W]
    grid_y = grid_y.unsqueeze(0).expand(B, H, W)  # [B, H, W]

    # Subtract disparity (horizontal shift)
    x_warp = grid_x - disp.squeeze(1)  # disp was [B,1,H,W], now [B,H,W]
    y_warp = grid_y

    # Normalize to [-1, 1] for grid_sample
    x_norm = 2.0 * x_warp / (W - 1) - 1.0
    y_norm = 2.0 * y_warp / (H - 1) - 1.0

    grid = torch.stack((x_norm, y_norm), dim=3)  # [B, H, W, 2]

    # Warp the image
    warped_img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # Mask for valid pixels (where the warped coordinates are within image bounds)
    valid_mask = (x_warp >= 0) & (x_warp <= W - 1) & (y_warp >= 0) & (y_warp <= H - 1)
    valid_mask = valid_mask.float().unsqueeze(1)  # [B, 1, H, W]

    return warped_img, valid_mask

def train_stereo_model_lora(args):
    model_with_lora = IGEVStereoLoraModel(args)
    optimizer = optim.AdamW(model_with_lora.parameters(), lr=1e-4)
    train_loader, val_loader, total_size = get_scared_dataloader(args, train=True)
    num_epochs = args.train_iters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_with_lora.to(device)

    for epoch in range(num_epochs):
        model_with_lora.train()
        for img_left, img_right in train_loader:
            img_left = img_left.to(device)
            img_right = img_right.to(device)
            
            # 1. Forward
            disp_list = model_with_lora(img_left, img_right)  # [B, 1, H, W]
            disp_left = disp_list[0]

            # 2. Warping
            img_right_warped, mask = stereo_warp(img_left, disp_left)
            
            # 3. Pérdida fotométrica
            loss = photometric_loss(img_right, img_right_warped, mask=mask)
            
            # 4. Backprop + optim
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Loss: {loss.item():.4f}")


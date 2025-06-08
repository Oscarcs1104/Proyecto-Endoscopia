import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
import torch.optim as optim

from core.igev_stereo_lora import IGEVStereoLoraModel
from Dataset.dataloader import get_scared_dataloader
from torch.optim.lr_scheduler import StepLR
import wandb
from metrics import DepthEvaluationMetrics
from loguru import logger
from torch.cuda.amp import autocast, GradScaler
from core.utils.args import Args
from tqdm import tqdm
import matplotlib
import numpy as np

def disparity_to_depth(disparity, focal_length, baseline):

    focal_lengths = torch.tensor(focal_length, dtype=disparity.dtype, device=disparity.device).view(-1, 1, 1, 1)
    depth = (focal_lengths * baseline) / (disparity + 1e-8)
    depth[disparity <= 0] = 0

    return depth

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

    ssim_loss = 1 - ssim( img1, img2, data_range=1.0, size_average=True)
    l1_loss = F.l1_loss(img1, img2, reduction='none').mean(dim=1, keepdim=True)  # [B, 1, H, W]

    # Combine losses
    combined_loss = alpha * ssim_loss + (1 - alpha) * l1_loss  # [B, 1, H, W]

    # Apply mask if provided
    if mask is not None:
        valid_pixels = mask.sum() + eps
        loss = (combined_loss * mask).sum() / valid_pixels
    else:
        loss = combined_loss.mean()

    return loss

def smoothness_loss(disparity, img):
    """Encourages disparity smoothness except at image edges."""
    grad_disp_x = torch.abs(disparity[:, :, :, :-1] - disparity[:, :, :, 1:])
    grad_disp_y = torch.abs(disparity[:, :, :-1, :] - disparity[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    # Reduce penalty at image edges
    smooth_x = grad_disp_x * torch.exp(-grad_img_x)
    smooth_y = grad_disp_y * torch.exp(-grad_img_y)
    return smooth_x.mean() + smooth_y.mean()

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
    B, _, H, W = img.size()
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
    start_epoch = 0
    accum_steps = 2  # Puedes ajustar este valor

    model_with_lora = IGEVStereoLoraModel(args)
    optimizer = optim.AdamW(model_with_lora.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.7)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda"
    model_with_lora.to(device)
    best_rmse = float('inf')

    wandb.init(project="stereo-lora", config=vars(args))
    scaler = GradScaler()
    train_loader = get_scared_dataloader(args, train=True)
    val_loader = get_scared_dataloader(args, train=False)

    logger.info(f"Total images for training phase {len(train_loader)}")

    for epoch_idx in range(start_epoch, args.epochs):
            model_with_lora.train()
            progress_bar = tqdm(enumerate(train_loader), 
                        total=len(train_loader),
                        desc=f"Epoch {epoch_idx + 1}")
            
            for batch_idx, batch in progress_bar:
                img_left, img_right = [x.cuda(non_blocking=True) for x in batch]

                #with autocast(dtype=torch.float32): #Try full precision first
                disp_list = model_with_lora(img_left, img_right)  # [B, 1, H, W]
                img_right_warped, mask = stereo_warp(img_right, disp_list) #img_right_warped == synthetic left
                loss_photo = photometric_loss(img_right, img_right_warped, mask=mask)
                smooth_loss = smoothness_loss(disp_list, img_left)

                loss = loss_photo + 0.3 * smooth_loss  
                loss = loss / accum_steps  # Normaliza la pérdida

                #scaler.scale(loss).backward()
                loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                """
                if (batch_idx + 1) % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                """
                lr = optimizer.param_groups[0]['lr']

                total_norm = 0.0
                for p in model_with_lora.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)  # L2 norm of this parameter's gradient
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                wandb.log({"gradient norm": total_norm,
                        "epoch": epoch_idx+1,
                        "loss": loss.item()*accum_steps,
                        "lr": lr,})
                #print(f"Epoch [{epoch_idx+1}/{args.epochs}], Step [{batch_idx+1}], Loss: {loss.item()*accum_steps:.4f}, lr: {lr:.6f}")

                if batch_idx % 20 == 0:
                    # --- Wandb logging de imágenes ---
                    left_np = img_left[0].detach().cpu().numpy().transpose(1,2,0)
                    right_np = img_right[0].detach().cpu().numpy().transpose(1,2,0)
                    warp_np = img_right_warped[0].detach().cpu().numpy().transpose(1,2,0)
                    disp_np = disp_list[0,0].detach().cpu().numpy()

                    # Chequeo de valores válidos
                    if not np.isfinite(disp_np).all() or (np.max(disp_np) - np.min(disp_np) < 1e-6):
                        disp_norm = np.zeros_like(disp_np)
                    else:
                        disp_norm = (disp_np - np.min(disp_np)) / (np.max(disp_np) - np.min(disp_np) + 1e-8)

                    disp_color = matplotlib.cm.get_cmap('Spectral')(disp_norm)[:, :, :3]  # (H, W, 3)
                    disp_color = (disp_color * 255).astype(np.uint8)

                    wandb.log({
                        "Warp": wandb.Image(warp_np, caption="Warped"),
                        "Disparidad Color": wandb.Image(disp_color, caption="Disparidad Color"),
                        "Imagen Izquierda": wandb.Image(left_np, caption="Imagen Izquierda"),
                        "Imagen Derecha": wandb.Image(right_np, caption="Imagen Derecha")
                    })

            scheduler.step()

            eval_metrics = evaluate_stereo_lora(args, model=model_with_lora, val_loader=val_loader)
            rmse_metric = eval_metrics["RMSE"]
            if rmse_metric < best_rmse:
                print(f"Validation RMSE improved from {best_rmse:.4f} to {rmse_metric:.4f}. Saving checkpoint.")
                torch.save(model_with_lora.state_dict(), f"checkpoints/model_epoch_{epoch_idx+1}.pth")

    wandb.finish()

def evaluate_stereo_lora(args, model, val_loader):

    metrics_calculator = DepthEvaluationMetrics()

    model.eval()
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")

    with torch.no_grad():
        for idx, batch in progress_bar:
            img_left, img_right, gt = [x.cuda(non_blocking=True) for x in batch]

            disp_pred = model(img_left, img_right)
            disp2depth = disparity_to_depth(disp_pred, focal_length=1035, baseline=4.143) #baseline in mm, focal_length in px

            metrics_calculator.update(pred_depth=disp2depth, target_depth=gt)

        final_metrics = metrics_calculator.compute_metrics()
        print("\n--- Validation Metrics ---")
        for metric, value in final_metrics.items():
            print(f"{metric}: {value:.4f}")

    metrics_calculator.reset()
    return final_metrics

if __name__ == "__main__":
    args = Args()
    train_stereo_model_lora(args)

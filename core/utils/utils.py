import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
import cv2

# Creadas por nosotros

def load_image(path, normalize=True):
    """
    Load an image from path and convert to PyTorch tensor.
    
    Args:
        path: Path to image file
        normalize: Whether to normalize to [0,1] range
    Returns:
        Tensor [1, C, H, W] in RGB format
    """
    img = cv2.imread(path)  # [H, W, C] in BGR uint8
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
    img = torch.from_numpy(img).float()  # [H, W, C]
    
    if normalize:
        img = img / 255.0  # normalize to [0,1]
    
    img = img.permute(2, 0, 1)  # [C, H, W]
    img = img.unsqueeze(0)  # [1, C, H, W]
    return img


def load_disparity_png(path, scale=1.0, channel_to_use=0):
    """
    Load a disparity PNG file (with alpha channel) and convert to PyTorch tensor.
    
    Args:
        path: Path to the disparity PNG file
        scale: Scaling factor to apply to disparity values
        channel_to_use: Which channel to extract (default 0 for first channel)
                       Set to -1 to use the maximum across all channels
    
    Returns:
        Tensor [1, 1, H, W] with disparity values
    """
    # Load image with all channels (including alpha)
    disp_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if disp_img is None:
        raise FileNotFoundError(f"Disparity image not found at {path}")
    
    # Convert to float32 tensor
    disp_tensor = torch.from_numpy(disp_img.astype(np.float32))
    
    # Handle multi-channel case (like RGBA)
    if len(disp_tensor.shape) == 3:
        if channel_to_use == -1:
            # Use maximum across all channels
            disp_tensor = disp_tensor.max(dim=2)[0]
        else:
            # Use specified channel
            disp_tensor = disp_tensor[:, :, channel_to_use]
    
    # Apply scaling and add batch/channel dimensions
    disp_tensor = disp_tensor * scale
    disp_tensor = disp_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    return disp_tensor



# Creadas por IGEV++

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]

    # print("$$$55555", img.shape, coords.shape)
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1

    # print("######88888", xgrid)
    assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem

    grid = torch.cat([xgrid, ygrid], dim=-1)
    # print("###37777", grid.shape)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def gauss_blur(input, N=5, std=1):
    B, D, H, W = input.shape
    x, y = torch.meshgrid(torch.arange(N).float() - N//2, torch.arange(N).float() - N//2)
    unnormalized_gaussian = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * std ** 2))
    weights = unnormalized_gaussian / unnormalized_gaussian.sum().clamp(min=1e-4)
    weights = weights.view(1,1,N,N).to(input)
    output = F.conv2d(input.reshape(B*D,1,H,W), weights, padding=N//2)
    return output.view(B, D, H, W)
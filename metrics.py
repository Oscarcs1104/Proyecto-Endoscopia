import torch
import numpy as np

class DepthEvaluationMetrics:
    """
    Calculates standard evaluation metrics for depth estimation:
    - Absolute Relative Error (AbsRel)
    - Squared Relative Error (SqRel)
    - Root Mean Squared Error (RMSE)
    - Root Mean Squared Error on Logarithmic Scale (RMSE_log)
    - Threshold Accuracy (δ1, δ2, δ3)
    """
    def __init__(self):
        """Initializes the accumulators for each metric."""
        self.abs_rel = 0.0
        self.sq_rel = 0.0
        self.rmse = 0.0
        self.rmse_log = 0.0
        self.delta1 = 0.0
        self.delta2 = 0.0
        self.delta3 = 0.0
        self.valid_pixels = 0

    def update(self, pred_depth, target_depth, mask=None):
        """
        Updates the metric accumulators with a batch of predictions.

        Args:
            pred_depth (torch.Tensor): Predicted depth map (shape [B, H, W]).
            target_depth (torch.Tensor): Ground truth depth map (shape [B, H, W]).
            mask (torch.Tensor, optional): Mask for valid pixels (shape [B, 1, H, W]).
                                           If None, pixels where target_depth > 0 are considered valid.
                                           Defaults to None.
        """
        # Ensure tensors are on CPU and detach gradients
        pred_depth = pred_depth.detach().cpu()
        target_depth = target_depth.detach().cpu()

        # Create a mask for valid pixels (target_depth > 0)
        # You might need to adjust the threshold based on your data
        valid_mask = target_depth > 1e-3
        if mask is not None:
            # Combine provided mask with target depth validity mask
            valid_mask = valid_mask & (mask.squeeze(1) > 0) # Assuming mask is [B, 1, H, W]

        # Apply mask to both predicted and target depths
        pred_depth = pred_depth[valid_mask]
        target_depth = target_depth[valid_mask]

        if pred_depth.numel() == 0:
            return # Skip if no valid pixels in this batch

        # Calculate ratios
        ratio = pred_depth / target_depth

        # Calculate Absolute Relative Error (AbsRel)
        abs_rel = torch.abs(ratio - 1)

        # Calculate Squared Relative Error (SqRel)
        sq_rel = (ratio - 1) ** 2

        # Calculate Root Mean Squared Error (RMSE)
        rmse = torch.sqrt((pred_depth - target_depth) ** 2)

        # Calculate Root Mean Squared Error on Logarithmic Scale (RMSE_log)
        log_pred = torch.log(pred_depth)
        log_target = torch.log(target_depth)
        rmse_log = torch.sqrt((log_pred - log_target) ** 2)

        # Calculate Threshold Accuracy (δ1, δ2, δ3)
        delta1 = torch.min(ratio, 1 / ratio) < 1.25
        delta2 = torch.min(ratio, 1 / ratio) < 1.25**2
        delta3 = torch.min(ratio, 1 / ratio) < 1.25**3

        # Accumulate sums and valid pixel count
        self.abs_rel += abs_rel.sum().item()
        self.sq_rel += sq_rel.sum().item()
        self.rmse += rmse.sum().item()
        self.rmse_log += rmse_log.sum().item()
        self.delta1 += delta1.sum().item()
        self.delta2 += delta2.sum().item()
        self.delta3 += delta3.sum().item()
        self.valid_pixels += pred_depth.numel()

    def compute_metrics(self):
        """
        Computes and returns the average metrics over all processed batches.

        Returns:
            dict: A dictionary containing the average metric values.
        """
        if self.valid_pixels == 0:
            return {
                'AbsRel': 0.0,
                'SqRel': 0.0,
                'RMSE': 0.0,
                'RMSE_log': 0.0,
                'δ1': 0.0,
                'δ2': 0.0,
                'δ3': 0.0,
            }

        avg_abs_rel = self.abs_rel / self.valid_pixels
        avg_sq_rel = self.sq_rel / self.valid_pixels
        avg_rmse = self.rmse / self.valid_pixels
        avg_rmse_log = self.rmse_log / self.valid_pixels
        avg_delta1 = (self.delta1 / self.valid_pixels) * 100.0
        avg_delta2 = (self.delta2 / self.valid_pixels) * 100.0
        avg_delta3 = (self.delta3 / self.valid_pixels) * 100.0

        return {
            'AbsRel': avg_abs_rel,
            'SqRel': avg_sq_rel,
            'RMSE': avg_rmse,
            'RMSE_log': avg_rmse_log,
            'δ1': avg_delta1,
            'δ2': avg_delta2,
            'δ3': avg_delta3,
        }

    def reset(self):
        """Resets the metric accumulators to zero."""
        self.abs_rel = 0.0
        self.sq_rel = 0.0
        self.rmse = 0.0
        self.rmse_log = 0.0
        self.delta1 = 0.0
        self.delta2 = 0.0
        self.delta3 = 0.0
        self.valid_pixels = 0

# metrics_calculator = DepthEvaluationMetrics(), metrics_calculator.reset()
# # Compute and print final metrics
# final_metrics = metrics_calculator.compute_metrics()
# print("\n--- Validation Metrics ---")
# for metric, value in final_metrics.items():
#     print(f"{metric}: {value:.4f}")

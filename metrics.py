import torch
import numpy as np

import torch
import math

class DepthEvaluationMetrics:
    """
    Calculates standard evaluation metrics for depth estimation:
    - Absolute Relative Error (AbsRel)
    - Squared Relative Error (SqRel)
    - Root Mean Squared Error (RMSE)
    - Root Mean Squared Error on Logarithmic Scale (RMSE_log)
    - Threshold Accuracy (δ1, δ2, δ3) in percent
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all accumulators."""
        self.abs_rel_sum   = 0.0
        self.sq_rel_sum    = 0.0
        self.mse_sum       = 0.0
        self.msle_sum      = 0.0
        self.delta1_count  = 0
        self.delta2_count  = 0
        self.delta3_count  = 0
        self.valid_pixels  = 0

    def update(self, pred_depth, target_depth, mask=None):
        pd = pred_depth.detach().cpu().squeeze(1)
        td = target_depth.detach().cpu().squeeze(1)

        # 1) máscara de píxeles reales válidos
        valid = td > 1e-6
        if mask is not None:
            valid &= (mask.detach().cpu().squeeze(1) > 0)
        # 2) excluir predicciones inválidas (profundidad cero o negativa)
        #valid &= (td > 1e-6)

        pd = pd[valid]
        td = td[valid]
        n  = pd.numel()
        if n == 0:
            return

        eps = 1e-6

        # AbsRel
        abs_rel_map = torch.abs(pd - td) / (td + eps)
        self.abs_rel_sum += abs_rel_map.sum().item()

        # SqRel
        sq_rel_map = (pd - td)**2 / (td + eps)
        self.sq_rel_sum += sq_rel_map.sum().item()

        # RMSE
        mse_map = (pd - td)**2
        self.mse_sum += mse_map.sum().item()

        # RMSE_log
        msle_map = (torch.log(pd + eps) - torch.log(td + eps))**2
        self.msle_sum += msle_map.sum().item()

        # δ1, δ2, δ3
        ratio = pd / (td + eps)
        max_r = torch.max(ratio, 1.0/ratio)
        self.delta1_count += (max_r < 1.25).sum().item()
        self.delta2_count += (max_r < 1.25**2).sum().item()
        self.delta3_count += (max_r < 1.25**3).sum().item()

        # píxeles contados
        self.valid_pixels += n


    def compute_metrics(self):
        """Returns a dict of final averaged metrics."""
        if self.valid_pixels == 0:
            return {k: 0.0 for k in
                ["AbsRel","SqRel","RMSE","RMSE_log","δ1","δ2","δ3"]}

        n = self.valid_pixels
        abs_rel   = self.abs_rel_sum   / n
        sq_rel    = self.sq_rel_sum    / n
        rmse      = math.sqrt(self.mse_sum  / n)
        rmse_log  = math.sqrt(self.msle_sum / n)
        δ1        = 100.0 * self.delta1_count / n
        δ2        = 100.0 * self.delta2_count / n
        δ3        = 100.0 * self.delta3_count / n

        return {
            "AbsRel":   abs_rel,
            "SqRel":    sq_rel,
            "RMSE":     rmse,
            "RMSE_log": rmse_log,
            "δ1":       δ1,
            "δ2":       δ2,
            "δ3":       δ3,
        }

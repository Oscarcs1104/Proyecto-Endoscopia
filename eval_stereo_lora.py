import torch
from core.igev_stereo_lora import IGEVStereoLoraModel
from Dataset.dataloader import get_scared_dataloader
from core.utils.args import Args
from train_stereo import evaluate_stereo_lora
import wandb
import imageio
import plotly.graph_objects as go
from metrics import DepthEvaluationMetrics


if __name__ == "__main__":
    args = Args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Inicializa wandb
    wandb.init(project="stereo-lora-eval", config=vars(args))

    # Lista de checkpoints a evaluar
    checkpoints = [
        "checkpoints/model_epoch_5.pth"
    ]

    # DataLoader de validación
    val_loader = get_scared_dataloader(args, train=False)

    for ckpt in checkpoints:
        print(f"\nEvaluando checkpoint: {ckpt}")
        model = IGEVStereoLoraModel(args)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)

        # Evaluación con impresión de rangos de depth
        model.eval()
        from train_stereo import disparity_to_depth
        import matplotlib
        import numpy as np

        metrics_calculator = DepthEvaluationMetrics()

        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                img_left, img_right, gt = [x.to(device, non_blocking=True) for x in batch]

                disp_pred = model(img_left, img_right)
                disp2depth = disparity_to_depth(disp_pred, focal_length=1035.20, baseline=0.0041434)

                # Calcula min/max del depth predicho
                depth_pred_min = disp2depth.min().item()
                depth_pred_max = disp2depth.max().item()

                # Calcula min/max del ground truth
                depth_gt_min = gt.min().item()
                depth_gt_max = gt.max().item()

                metrics_calculator.update(pred_depth=disp2depth.squeeze(0), target_depth=gt.squeeze(0))
                print(f"[Batch {idx}] Depth predicho: min={depth_pred_min:.4f}, max={depth_pred_max:.4f} | GT: min={depth_gt_min:.4f}, max={depth_gt_max:.4f}")

                # Guarda el primer depth del batch como PNG (normalizado a 0-255)
                if idx < 5:  # Cambia el número para guardar más o menos imágenes
                    depth_img = disp2depth[0][0].detach().cpu().numpy()
                    depth_img_norm = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-8)
                    depth_img_uint8 = (depth_img_norm * 255).astype(np.uint8)

                    filename_base = f"depth_pred_ckpt{ckpt.split('_')[-1].split('.')[0]}_batch{idx}"
                    imageio.imwrite(f"{filename_base}.png", depth_img_uint8)

                    # Guarda min/max en un archivo txt
                    with open(f"{filename_base}_minmax.txt", "w") as f:
                        f.write(f"{depth_pred_min},{depth_pred_max}\n")

                    Z = depth_img_uint8.astype(np.float32) / 255.0
                    Z = Z * (depth_pred_max - depth_pred_min) + depth_pred_min  # desnormaliza a escala real

                    H, W = Z.shape
                    x = np.linspace(0, W-1, W)
                    y = np.linspace(0, H-1, H)
                    X, Y = np.meshgrid(x, y)

                    fig = go.Figure(data=[
                        go.Surface(
                            x=X, y=Y, z=Z,
                            colorscale='Viridis',
                            cmin=np.nanmin(Z), cmax=np.nanmax(Z),
                            colorbar=dict(title='Depth'),
                            showscale=True
                        )
                    ])
                    # ...dentro del if idx < 5: ...
                    fig.update_layout(
                        title=f'3D Surface Plot of {filename_base}',
                        scene=dict(
                            xaxis_title='Pixel X',
                            yaxis_title='Pixel Y',
                            zaxis_title='Depth (mm)',
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
                        ),
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    fig.write_html(f"{filename_base}.html")
                    print(f"Visualización guardada en {filename_base}.html")
            final_metrics = metrics_calculator.compute_metrics()
            print("\n--- Validation Metrics ---")
            for metric, value in final_metrics.items():
                print(f"{metric}: {value:.4f}")

        metrics_calculator.reset()
    wandb.finish()
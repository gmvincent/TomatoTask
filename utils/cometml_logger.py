import comet_ml
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import torch
import torch.distributed as dist
import numpy as np
from sklearn.manifold import TSNE
import datetime
from collections import Counter
from dataclasses import dataclass
from typing import Optional, List

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter
from mpl_toolkits.axes_grid1 import ImageGrid

from pytorch_grad_cam import GradCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from sklearn.metrics import confusion_matrix

from thop import profile, clever_format

from utils.metrics import initialize_metrics, log_metrics

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold")

@dataclass
class TaskInfo:
    task_idx: int              # position in args.task / args.classes / model output list
    batch_idx: int             # index into a dataloader batch tuple -> batch[batch_idx]
    type: str                  # "classification" | "segmentation" | "regression"
    classes: Optional[list]    # class names (classification/segmentation) or int placeholders (regression)

# Initialize CometML Experiment
def create_experiment(args):
    now = datetime.datetime.now()
    
    if isinstance(args.task, list):
        task = "multi"
    elif args.task == "classification":
        task = "clf"
    elif args.task == "segmentation":
        task = "seg"
    elif args.task == "regression":
        task = "reg"
    else:
        raise ValueError(f"Unrecognized args.task '{args.task}' and args.task is not a list.")
    
    experiment_name = f"{args.model_name}_{task}_{args.num_tasks}task_{args.dataset_name}_{now.strftime('%y%m%d%H%M')}"
    
    experiment = Experiment(
        api_key="6XqmAhuJUkx6wPhz0sdCRXwRz",
        project_name=args.cometml_project, 
        workspace="gmvincent",
    )

    experiment.set_name(experiment_name)

    return experiment

def log_experiment(
    args, experiment, model, dataloader, metrics, loss, epoch, y_true, y_pred, mode="train",
):

    # Ensure only rank 0 logs to CometML
    if args.ddp and dist.get_rank() != 0:
        return  
    
    log_metrics(experiment, metrics, loss, epoch, mode)
    
    # log plots
    if (epoch >= args.epochs - 1) or (epoch % args.print_freq == 0):
        plot_confusion_matrix(args, experiment, y_true.cpu(), y_pred.cpu(), epoch, mode) 
        plot_segmentation_outputs(args, experiment, model, dataloader, epoch, mode) 
        plot_residual_histogram(args, experiment, y_true, y_pred, epoch, mode)
        pred_v_actual(args, experiment, y_true, y_pred, epoch, mode) 
        
        cam_model = model.module if hasattr(model, "module") else model
        if any(m in args.model_name.lower() for m in ["vit", "swin"]):
            plot_attention_maps(args, experiment, cam_model, dataloader, epoch, mode)
        else:
            plot_cam(args, experiment, cam_model, dataloader, epoch, mode)
        
    if (epoch >= args.epochs - 1) and (mode == "test"):
        # Log Experiment Specific Args
        for arg, value in vars(args).items():
            experiment.log_parameter(arg, value)

def log_model_weights(args, experiment, model):
    # Log model weights
    log_model(experiment, model, "final_model")
    
    # Calculate and log the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    experiment.log_parameter("num_parameters", total_params)
    
    # Calculate and log the model size (in MB)
    param_size = sum(p.element_size() * p.numel() for p in model.parameters())
    model_size_mb = param_size / (1024 ** 2)
    experiment.log_parameter("model_size_MB", model_size_mb)
    
    # Calculate and log the number of FLOPs
    #flops, params = profile(model, inputs=(torch.zeros(1, args.input_channels, 64, 64)), verbose=False)
    #flops, params = clever_format([flops, params], "%.3f")

    #experiment.log_parameter("num_flops", flops)

def resolve_tasks(args) -> List[TaskInfo]:
    """
    Normalize legacy single-task and multi-task configs into one list of
    TaskInfo objects, so every plotting function loops over `tasks` and
    filters by `.type` instead of special-casing args.num_tasks.
 
    Assumes model(inputs) and args.classes are ordered identically to
    args.task (enforced by your dataset_tasks sanity check in main).
    """
    if args.num_tasks in ["1", "tomato"]:
        return [TaskInfo(task_idx=0, batch_idx=1, type=args.task, classes=args.classes)]
 
    tasks = [args.task] if isinstance(args.task, str) else list(args.task)
    return [
        TaskInfo(task_idx=i, batch_idx=i + 1, type=t, classes=args.classes[i])
        for i, t in enumerate(tasks)
    ]

def plot_distribution(args, experiment, dataloader, mode):
    tasks = [t for t in resolve_tasks(args) if t.type in ("classification", "segmentation", "regression")]
    single_task = args.num_tasks in ["1", "tomato"]
    if not tasks:
        return
    
    per_task_labels = {t.task_idx: [] for t in tasks}
    for batch in dataloader:
        for t in tasks:
            label = batch[t.batch_idx]
            if label is not None and len(label) > 0:
                per_task_labels[t.task_idx].append(label.numpy())
    
    for t in tasks:
        labels = per_task_labels[t.task_idx]
        if not labels:
            print(f"No labels found for task {t.task_idx} ({t.type}) in {mode} dataloader")
            continue
        
        labels = np.concatenate([l.flatten() for l in labels])
        
        label_counts = Counter(labels)
        freqs = [label_counts.get(i, 0) for i in range(len(t.classes))]

        # Bar plot with class names as x-axis ticks
        fig, ax = plt.subplots(figsize=(14, 11))
        
        ax.bar(range(len(t.classes)), freqs, color="orchid")
        ax.set_xticks(range(len(t.classes)))
        ax.set_xticklabels(t.classes, rotation=90, ha="right")
        ax.set_xlabel('')
        ax.set_ylabel('Frequency')
        
        # Log the plot to CometML
        suffix = "" if single_task else f"_task{t.task_idx}"
        experiment.log_figure(figure_name=f"{mode}/data_distribution{suffix}", figure=plt.gcf())
        plt.close(fig)

def plot_confusion_matrix(args, experiment, y_true, y_pred, step, mode):
    tasks = [t for t in resolve_tasks(args) if t.type in ("classification", "segmentation")]
    single_task = args.num_tasks in ["1", "tomato"]
    
    for t in tasks:
        true_t = y_true if single_task else y_true[t.task_idx]
        pred_t = y_pred if single_task else y_pred[t.task_idx]
        
        if t.type == "segmentation":
            true_t = true_t.flatten()
            pred_t = pred_t.flatten()
            
        cmat = confusion_matrix(true_t, pred_t, normalize="true", labels=range(len(t.classes)))
        
        fig, ax = plt.subplots(figsize=(20, 18))
        sns.heatmap(
                    cmat, 
                    annot=True, 
                    fmt=".2f", 
                    cmap="Blues", 
                    square=True, 
                    cbar=False,
                    xticklabels=t.classes,
                    yticklabels=t.classes,
                    ax=ax,
                    )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        
        # Log the plot to CometML
        suffix = "" if single_task else f"_task{t.task_idx}"
        experiment.log_figure(figure_name=f"{mode}/cm{suffix}", figure=plt.gcf(), step=step)
        plt.close(fig)

# TODO: setup GradCam for segmentation
def plot_cam(args, experiment, model, dataloader, step, mode, num_images=6):
    tasks = [t for t in resolve_tasks(args) if t.type == "classification"]
    single_task = args.num_tasks in ["1", "tomato"]
    
    model.eval()
    
    batch = next(iter(dataloader))
    inputs = batch[0]
    
    indices = np.random.choice(inputs.shape[0], size=num_images, replace=False)
    inputs = inputs[indices].to(args.device).float()
    for t in tasks:
        labels = batch[t.batch_idx][indices].to(args.device).long()
        
        wrapped_model = model if single_task else MultiTaskWrapper(model, task_idx=t.task_idx)
        cam = GradCAM(model=wrapped_model, target_layers=args.target_layers)        
                
        targets = [ClassifierOutputTarget(label.item()) for label in labels]
        grayscale_cam = cam(input_tensor=inputs, targets=targets)
        
        fig, axes = plt.subplots(num_images, 2, figsize=(8, 4 * num_images))
        if num_images == 1:
            axes = [axes]  # make sure it's iterable

        for i in range(num_images):
            rgb_img = inputs[i].cpu().permute(1, 2, 0).numpy()
            if args.input_channels > 3: rgb_img = rgb_img[:,:,:3]
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)  # normalize
            visualization = show_cam_on_image(rgb_img, grayscale_cam[i], use_rgb=True)

            axes[i][0].imshow(rgb_img)
            for spine in axes[i][0].spines.values():
                spine.set_visible(False)
            axes[i][0].set_xticks([])
            axes[i][0].set_yticks([])
            
            label_idx = int(labels[i].item())
            label_text = t.classes[label_idx] if isinstance(t.classes[0], str) else str(label_idx)
            axes[i][0].set_ylabel(label_text, fontsize=14, rotation=90, labelpad=40, va='center')
            
            axes[i][1].imshow(visualization)
            axes[i][1].axis("off")
            
            if i == 0:
                axes[i][0].set_title("Input Image")
                axes[i][1].set_title("CAM")
                
        plt.tight_layout()
        suffix = "" if single_task else f"_task{t.task_idx}"
        experiment.log_figure(figure_name=f"{mode}/cam{suffix}", figure=fig, step=step)
        plt.close(fig)

def get_attention_map(model, img, single_task=True):
    attention_maps = []

    def hook_fn(module, input, output):
        # Torchvision ViT returns (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) == 2:
            attn_weights = output[1]
        else:
            attn_weights = output
        if attn_weights is not None:
            attention_maps.append(attn_weights.detach().cpu())

    def forward_with_weights(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return original_forward(*args, **kwargs)
    
    def my_forward_wrapper(attn_obj):
        def my_forward(x):
            reshape_back = False

            # Swin blocks may receive [B, H, W, C]
            if x.dim() == 4:
                B, H, W, C = x.shape
                x = x.view(B, H * W, C)
                reshape_back = True
            else:
                B, N, C = x.shape
                H = W = int(N**0.5)  # fallback if not 4D

            qkv = attn_obj.qkv(x).reshape(B, -1, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            head_dim = attn_obj.qkv.weight.shape[1] // attn_obj.num_heads
            scale = head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            
            attn = attn.softmax(dim=-1)
            attn_obj.attn_map = attn  # Save attention

            #attn = attn_obj.attn_dropout(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
            x = attn_obj.proj(x)
            #x = attn_obj.proj_drop(x)

            if reshape_back:
                x = x.view(B, H, W, C)

            return x
        return my_forward
    
    model_to_check = model.model if not single_task else model
    is_swin = False

    # Torchvision ViT
    if hasattr(model_to_check, "encoder") and hasattr(model_to_check.encoder, "layers"):
        last_block = model_to_check.encoder.layers[-1]
        sa_module = last_block.self_attention
        original_forward = sa_module.forward
        
        sa_module.forward = forward_with_weights
        hook_handle = sa_module.register_forward_hook(hook_fn)
    elif hasattr(model_to_check, "blocks"):
        last_block = model_to_check.blocks[-1]
        sa_module = last_block.self_attention
        original_forward = sa_module.forward
        
        sa_module.forward = forward_with_weights
        hook_handle = sa_module.register_forward_hook(hook_fn)
    
    # Torchvision Swin
    elif hasattr(model_to_check, "features"):
        last_block = model_to_check.features[-1][1]
        sa_module = last_block.attn
        
        original_forward = sa_module.forward
        sa_module.forward = my_forward_wrapper(sa_module)
        hook_handle = None
        is_swin = True
    else:
        raise ValueError("Unsupported model structure.")
    
    # Forward pass
    with torch.no_grad():
        _ = model(img)

    # Remove hook
    if hook_handle:
        hook_handle.remove()
    if original_forward:
        sa_module.forward = original_forward

    if not attention_maps and not is_swin:
        raise RuntimeError("No attention maps captured.")


    # Propagate CLS attention
    if is_swin: 
        attn = sa_module.attn_map.detach().cpu()
        attn = attn.mean(dim=1)        # [B*windows, tokens, tokens]
        attn = attn.mean(dim=-1)       # [B*windows, tokens]
        attn = attn.mean(dim=0)        # [tokens]
        num_tokens = attn.size(0)
        size = int(num_tokens ** 0.5)

        if size * size == num_tokens:
            mask = attn.reshape(size, size).numpy()
        else:
            print(f"[WARNING] Swin attention map size ({num_tokens}) is not square — returning flat attention.")
            mask = attn.numpy()
    else:    
        attn = attention_maps[-1]  # [B, num_heads, num_tokens, num_tokens]
        if attn.dim() == 4:
            attn = attn.mean(dim=1)[0] # Average over heads, get first item in batch

        # Propagate CLS attention
        identity = torch.eye(attn.size(-1))
        a = (attn + identity) / 2
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ identity
        for _ in range(1, 1):  # Propagation depth = 1
            result = a @ result

        mask = result[0, 1:]  # CLS token to patches
        num_patches = mask.size(0)
        size = int(num_patches ** 0.5)
        mask = mask.reshape(size, size).numpy()
            
    mask = mask / np.max(mask)

    return mask

def plot_attention_maps(args, experiment, model, dataloader, step, mode, num_images=6):
    tasks = [t for t in resolve_tasks(args) if t.type == "classification"]
    single_task = args.num_tasks in ["1", "tomato"]
        
    model.eval()
    
    batch = next(iter(dataloader))
    inputs = batch[0].to(args.device).float()
    
    indices = np.random.choice(inputs.shape[0], size=num_images, replace=False)
    inputs = inputs[indices]
    for t in tasks:
        labels = batch[t.batch_idx][indices].to(args.device).long()
        
        wrapped_model = model if single_task else MultiTaskWrapper(model, task_idx=t.task_idx)
        
        fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
        for i in range(num_images):
            rgb_img = inputs[i].detach().cpu().permute(1, 2, 0).numpy()
            if args.input_channels > 3: rgb_img = rgb_img[:,:,:3]
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
            rgb_img = Image.fromarray((rgb_img * 255).astype(np.uint8))
            
            input_img = inputs[i].unsqueeze(0).to(args.device)
            mask = get_attention_map(wrapped_model, input_img)
            mask_resized = np.array(
                Image.fromarray((mask * 255).astype(np.uint8)).resize(rgb_img.size, resample=Image.BILINEAR)
            ) / 255.0
            colored_mask = cm.get_cmap('turbo')(mask_resized)[..., :3]  # drop alpha
            overlay = (0.55 * np.array(rgb_img) / 255.0 + 0.45 * colored_mask).clip(0, 1)
            
            axes[0][0].set_title("Input Image")
            axes[i][0].imshow(rgb_img)
            axes[i][0].spines['top'].set_visible(False)
            axes[i][0].spines['right'].set_visible(False)
            axes[i][0].spines['bottom'].set_visible(False)
            axes[i][0].spines['left'].set_visible(False)
            axes[i][0].set_xticks([])
            axes[i][0].set_yticks([])

            label_idx = int(labels[i].item())
            label_text = t.classes[label_idx] if isinstance(t.classes[0], str) else str(label_idx)
            axes[i][0].set_ylabel(label_text, fontsize=14, rotation=90, labelpad=40, va='center')

            axes[i, 1].imshow(mask, cmap="turbo")
            axes[0, 1].set_title("Attention Map")
            axes[i, 1].axis("off")
            
            axes[i, 2].imshow(overlay)
            axes[0, 2].set_title("Overlay")
            axes[i, 2].axis("off")

        plt.tight_layout()
        suffix = "" if single_task else f"_task{t.task_idx}"
        experiment.log_figure(figure_name=f"{mode}/attention_maps{suffix}", figure=fig, step=step)
        plt.close(fig)    

def plot_segmentation_outputs(args, experiment, model, dataloader, step, mode, num_images=6):
    tasks = resolve_tasks(args)
    seg_tasks = [t for t in tasks if t.type == "segmentation"]
    if not seg_tasks:
        return
    
    model.eval()
       
    def _prep_image(img_tensor):
        img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
        if img.shape[-1] > 3:
            img = img[:, :, :3]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img
    
    def _plot_one_task(inputs, seg_mask, seg_classes, cls_labels, cls_names, model_out_idx):
        fig = plt.figure(figsize=(9, 4 * num_images), dpi=300)
        grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(num_images, 3),
            cbar_mode="single",
            cbar_pad=0.1,
            cbar_size="1%",
        )
        
        inputs = inputs.to(args.device).float()
        seg_mask = seg_mask.detach().cpu().numpy()
        
        with torch.no_grad():
            outputs = model(inputs)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[model_out_idx]
            pred_mask = outputs.detach().cpu().argmax(dim=1).numpy()
        
        # use a fixed colormap
        num_colors = len(seg_classes) + 1  # + 1 for null pixels
        cmap = plt.get_cmap("viridis", num_colors)
        
        last_im = None
        for i in range(num_images):
            rgb_img = _prep_image(inputs[i])
            gt_i = seg_mask[i].copy()
            pred_i = pred_mask[i].copy()
        
            # adds row of all colors to avoid colormap distortion
            gt_i[0, 0:num_colors] = list(range(num_colors))
            pred_i[0, 0:num_colors] = list(range(num_colors))

            ax_img, ax_gt, ax_pred = (grid[i * 3], grid[i * 3 + 1], grid[i * 3 + 2])

            ax_img.imshow(rgb_img)
            ax_gt.imshow(gt_i, cmap=cmap, vmin=0, vmax=num_colors - 1)
            last_im = ax_pred.imshow(pred_i, cmap=cmap, vmin=0, vmax=num_colors - 1)

            for ax in (ax_img, ax_gt, ax_pred):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)    
            
            if cls_labels is not None:
                label_idx = int(cls_labels[i].item())
                label_text = (
                    cls_names[label_idx] if isinstance(cls_names[0], str) else str(label_idx)
                )
                ax_img.set_ylabel(label_text, fontsize=14, rotation=90, labelpad=40, va="center")

            if i == 0:
                ax_img.set_title("Input Image", fontsize=10)
                ax_gt.set_title("Ground Truth", fontsize=10)
                ax_pred.set_title("Prediction", fontsize=10)
 
        cbar = grid.cbar_axes[0].colorbar(last_im, ticks=list(range(num_colors)))
        cbar.ax.set_yticklabels(list(seg_classes) + ["unlabeled"])    
        
        return fig
    
    batch = next(iter(dataloader))
    inputs = batch[0].to(args.device).float()
    
    indices = np.random.choice(inputs.shape[0], size=num_images, replace=False)
    inputs = inputs[indices]
    
    cls_task = [task for task in tasks if task.type == "classification"]
    cls_task = cls_task[0] if len(cls_task) == 1 else None
    
    for t in seg_tasks:
        seg_mask = batch[t.batch_idx][indices]
            
        cls_labels = batch[cls_task.batch_idx][indices] if cls_task is not None else None
        cls_names = cls_task.classes if cls_task is not None else None
        
        fig = _plot_one_task(
            inputs, seg_mask, t.classes, cls_labels, cls_names, model_out_idx=t.task_idx
        )
        
        plt.tight_layout()
        suffix = "" if len(seg_tasks) == 1 else f"_task{t.task_idx}"
        experiment.log_figure(figure_name=f"{mode}/segmentation{suffix}", figure=fig, step=step)
        plt.close(fig)
    
# prediction vs true plots
def pred_v_actual(args, experiment, y_true, y_pred, step, mode):
    tasks = [t for t in resolve_tasks(args) if t.type == "regression"]
    single_task = args.num_tasks in ["1", "tomato"]
    
    for t in tasks:
        true_t = y_true.detach().cpu() if single_task else y_true[t.task_idx].detach().cpu()
        pred_t = y_pred.detach().cpu() if single_task else y_pred[t.task_idx].detach().cpu()
        
        coefficients = np.polyfit(true_t, pred_t, 1)
        line = np.poly1d(coefficients)

        fig, ax = plt.subplots()
        
        ax.scatter(true_t, pred_t, label="Predictions", color="blue")
        ax.plot(true_t, line(true_t), "r--")
        ax.plot([min(t.classes), max(t.classes)], [min(t.classes), max(t.classes)], color="gray", linestyle=":", label="Ideal Fit")

        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        
        ax.set_xlim([min(t.classes), max(t.classes)])
        ax.set_ylim([min(t.classes), max(t.classes)])
        
        plt.legend()
        plt.grid(True)
        
        # Log the plot to CometML
        suffix = "" if single_task else f"_task{t.task_idx}"
        experiment.log_figure(figure_name=f"{mode}/pred_vs_true{suffix}", figure=plt.gcf(), step=step)
        plt.close(fig)
    

def plot_residual_histogram(args, experiment, y_true, y_pred, step, mode):
    tasks = [t for t in resolve_tasks(args) if t.type == "regression"]
    single_task = args.num_tasks in ["1", "tomato"]
    
    for t in tasks:
        true_t = y_true.detach().cpu() if single_task else y_true[t.task_idx].detach().cpu()
        pred_t = y_pred.detach().cpu() if single_task else y_pred[t.task_idx].detach().cpu()

        residuals = true_t - pred_t
        
        fig, _ = plt.subplots()
        plt.hist(residuals, bins=30, edgecolor="k", alpha=0.7, color="blue")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Residuals Distribution")

        # Log the plot to CometML
        suffix = "" if single_task else f"_task{t.task_idx}"
        experiment.log_figure(figure_name=f"{mode}/residual_histogram{suffix}", figure=plt.gcf(), step=step)
        plt.close(fig)

#TODO: plot true positives, false positives, false negatives

class MultiTaskWrapper(torch.nn.Module):
    def __init__(self, model, task_idx=0):
        super().__init__()
        self.model = model
        self.task_idx = task_idx

    def forward(self, x):
        return self.model(x, task_idx=self.task_idx)
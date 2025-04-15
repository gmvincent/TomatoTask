import comet_ml
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import os
import torch
import torch.distributed as dist
import numpy as np
from sklearn.manifold import TSNE
import datetime

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from sklearn.metrics import confusion_matrix

from thop import profile, clever_format

from utils.metrics import initialize_metrics, log_metrics, log_metrics_ddp

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold")

# Initialize CometML Experiment
def create_experiment(args):
    now = datetime.datetime.now()
    
    experiment_name = f"{args.model_name}_{args.num_tasks}tasks_{args.dataset_name}_{now.strftime('%y%m%d%H%M')}"

    experiment = Experiment(
        api_key="6XqmAhuJUkx6wPhz0sdCRXwRz",
        project_name=args.cometml_project, 
        workspace="gmvincent",
    )

    experiment.set_name(experiment_name)

    return experiment

def log_experiment(
    args, experiment, metrics, loss, epoch, y_true, y_pred, mode="train",
):

    # Ensure only rank 0 logs to CometML
    if args.ddp and dist.get_rank() != 0:
        return  
    
    log_metrics(experiment, metrics, loss, epoch, mode)
    
    # log plots
    if (epoch >= args.epochs - 1) or (epoch % args.print_freq == 0):
        y_true = torch.as_tensor(y_true).cpu().detach().numpy()
        y_pred = torch.as_tensor(y_pred).cpu().detach().numpy()

        plot_confusion_matrix(args, experiment, y_true, y_pred, epoch, mode)
        
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
    flops, params = profile(model, inputs=(torch.zeros(1, args.input_channels, 64, 64)), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    experiment.log_parameter("num_flops", flops)

def plot_distribution(args, experiment, dataloader, classes, mode):
    if args.num_tasks in ["1", "tomato"]:
        fig, ax = plt.subplots()
        
        labels = []
        for input, label in dataloader:
            if label is not None and len(label) > 0:
                labels.append(label.numpy())
        if labels:  # Only concatenate if labels contain data
            labels = np.concatenate(labels)
        else:
            print(f"No labels found in {mode} dataloader")
            return
        ax.hist(labels, bins=10, color="orchid")
        
        ax.set_ylabel('Frequency')
        
        # Log the plot to CometML
        experiment.log_figure(
            figure_name=f"{mode}/data_distribution", figure=plt.gcf()
        )
        plt.close(fig)
    else:
        for idx, task_labels in enumerate(args.classes):
            fig, ax = plt.subplots()
        
            labels = []
            for batch in dataloader:
                label = batch[idx]
                
                if label is not None and len(label) > 0:
                    labels.append(label.numpy())
            if labels:  # Only concatenate if labels contain data
                labels = np.concatenate(labels)
            else:
                print(f"No labels found in {mode} dataloader")
                return
            ax.hist(labels, bins=10, color="orchid")
            
            ax.set_ylabel('Frequency')
            
            # Log the plot to CometML
            experiment.log_figure(
                figure_name=f"{mode}/data_distribution_task{idx}", figure=plt.gcf()
            )
            plt.close(fig)

def plot_confusion_matrix(args, experiment, y_true, y_pred, step, mode):
    if args.num_tasks in ["1", "tomato"]:
        cm = confusion_matrix(y_true, y_pred, normalize="true", labels=args.classes)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt=".2f", 
                    cmap="Blues", 
                    square=True, 
                    cbar=False
                    )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        
        # Log the plot to CometML
        experiment.log_figure(figure_name=f"{mode}/cm", figure=plt.gcf(), step=step)
        plt.close(fig)
    else:
        for idx, task_labels in enumerate(args.classes):
            cm = confusion_matrix(y_true[idx], y_pred[idx], normalize="true", labels=task_labels)
        
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                        cm, 
                        annot=True, 
                        fmt=".2f", 
                        cmap="Blues", 
                        square=True, 
                        cbar=False
                        )
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            
            # Log the plot to CometML
            experiment.log_figure(figure_name=f"{mode}/cm_task{idx}", figure=plt.gcf(), step=step)
            plt.close(fig)

def plot_cam(args, experiment, model, dataloader, step, mode, num_images=6):
    if args.num_tasks in ["1", "tomato"]:
        fig, axes = plt.subplots(num_images, 1, figsize=(8, 4 * num_images))
        
        cam = GradCAM(model=model, target_layers=args.target_layers, use_cuda=args.device.type == "cuda")
        data_iter = iter(dataloader)
        inputs, labels = next(data_iter)
        
        inputs = inputs[:num_images].to(args.device).float()
        labels = labels[:num_images].to(args.device).float()
        
        targets = [ClassifierOutputTarget(label.item()) for label in labels]
        grayscale_cam = cam(input_tensor=inputs, targets=targets)
        
        if num_images == 1:
            axes = [axes]  # make sure it's iterable

        for i in range(num_images):
            rgb_img = inputs[i].cpu().permute(1, 2, 0).numpy()
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)  # normalize
            visualization = show_cam_on_image(rgb_img, grayscale_cam[i], use_rgb=True)

            axes[i].imshow(visualization)
            axes[i].axis("off")
            label_idx = int(labels[i].item())
            label_text = args.classes[label_idx] if isinstance(args.classes[0], str) else str(label_idx)
            axes[i].set_ylabel(label_text, fontsize=10, rotation=0, labelpad=40, va='center')

        plt.tight_layout()
        experiment.log_figure(figure_name=f"{mode}/cam", figure=fig, step=step)
        plt.close(fig)
    else:
        # TODO: set up multi-task cam support
        pass

        
#TODO: plot true positives, false positives, false negatives

#TODO: multi-task metrics and multi-task plots    
#TODO: plot segmentation masks (for canopy and lesion segmentation)
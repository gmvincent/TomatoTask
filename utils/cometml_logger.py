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

def plot_distribution(experiment, dataloader, classes, mode):
    
    fig, ax = plt.subplots()
    
    labels = []
    for input, label, _ in dataloader:
        if label is not None and len(label) > 0:
            labels.append(label.numpy())
    if labels:  # Only concatenate if labels contain data
        labels = np.concatenate(labels)
    else:
        print(f"No labels found in {mode} dataloader")
        return
    ax.hist(labels, bins=10, color="orchid")
    
    ax.set_xlim([min(classes), max(classes)])
    ax.set_xticks(np.arange(1, 10, 1.0).tolist())
    ax.set_xlabel('Severity Scores')
    ax.set_ylabel('Frequency')
    
    # Log the plot to CometML
    experiment.log_figure(
        figure_name=f"{mode}/data_distribution", figure=plt.gcf()
    )
    plt.close(fig)
    
#TODO: multi-task metrics and multi-task plots    
#TODO: plot confusion matrices
#TODO: plot segmentation masks (for canopy and lesion segmentation)
#TODO: plot class activation maps
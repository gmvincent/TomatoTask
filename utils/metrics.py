import torch
import torchmetrics
import torchmetrics.regression
import torch.distributed as dist

import numpy as np

class PredictionTime(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_time", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, start_time, end_time):
        self.total_time += torch.tensor(end_time - start_time)
        self.count += 1

    def compute(self):
        return self.total_time / self.count

    def reset(self):
        self.total_time = torch.tensor(0.0)
        self.count = torch.tensor(0)


def initialize_metrics(args):
    metrics = {
        "Accuracy": torchmetrics.Accuracy(num_classes=args.num_classes, task="multiclass"),
        #"F1": torchmetrics.F1Score(average="none", num_classes=args.num_classes, task="multiclass"),
        "Recall": torchmetrics.Recall(average=None, num_classes=args.num_classes, task="multiclass"),
        "Precision": torchmetrics.Precision(average=None, num_classes=args.num_classes, task="multiclass"),
        "F1_macro": torchmetrics.F1Score(average="macro", num_classes=args.num_classes, task="multiclass"),
        "MCC": torchmetrics.MatthewsCorrCoef(num_classes=args.num_classes, task="multiclass"),
        "PredictionTime": PredictionTime(),
    }

    [metric.to(args.device) for name, metric in metrics.items()]

    train_metrics = torchmetrics.MetricCollection(metrics)
    val_metrics, test_metrics = train_metrics.clone(), train_metrics.clone()

    return train_metrics, val_metrics, test_metrics


def log_metrics(experiment, metrics, loss, step, mode="train"):

    for name, value in metrics.items():
        val = value.compute()
        experiment.log_metric(
            f"{mode}/{name}", val.cpu().detach().numpy().tolist(), step=step
        )

    if loss is not None:
        experiment.log_metric(f"{mode}/loss", loss, step=step)
    else:
        experiment.log_metric(f"{mode}/loss", 0, step=step)  


def reduce_tensor(tensor, world_size):
    """
    Averages a tensor across all processes in DDP.
    """
    if world_size > 1:
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= world_size
    return tensor

def log_metrics_ddp(args, experiment, metrics, loss, step, mode="train"):       
    # Reduce metrics across processes
    world_size = dist.get_world_size() if args.ddp else 1

    loss_tensor = torch.tensor(loss, device=args.device)
    loss_reduced = reduce_tensor(loss_tensor, world_size).item()
    
    if args.ddp and metrics is not None:
        metrics.sync()

    dist.barrier() 
    
    log_metrics(experiment, metrics, loss_reduced, step, mode)
    
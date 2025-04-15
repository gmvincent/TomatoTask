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
    def build_metrics(num_classes):
        metrics = {
            "Accuracy": torchmetrics.Accuracy(num_classes=args.num_classes, task="multiclass"),
            #"F1": torchmetrics.F1Score(average="none", num_classes=args.num_classes, task="multiclass"),
            "Recall_macro": torchmetrics.Recall(average="macro", num_classes=args.num_classes, task="multiclass"), # also called Sensitivity
            "Precision_macro": torchmetrics.Precision(average="macro", num_classes=args.num_classes, task="multiclass"),
            "Specificity_macro": torchmetrics.Specificity(average="macro", num_classes=args.num_classes, task="multiclass"),
            "F1_macro": torchmetrics.F1Score(average="macro", num_classes=args.num_classes, task="multiclass"),
            "MCC": torchmetrics.MatthewsCorrCoef(num_classes=args.num_classes, task="multiclass"),
            "PredictionTime": PredictionTime(),
        }

        for metric in metrics.values():
            metric.to(args.device)
        return torchmetrics.MetricCollection(metrics)
        
    if args.num_tasks in [1, "1", "tomato"]:
        train_metrics = build_metrics(args.num_classes)
        val_metrics, test_metrics = train_metrics.clone(), train_metrics.clone()
        return train_metrics, val_metrics, test_metrics
    else:
        # Multi-task setup
        train_metrics = []
        for num_classes in args.classes:  # args.classes should be a list of class counts per task
            train_metrics.append(build_metrics(num_classes))
        val_metrics = [tm.clone() for tm in train_metrics]
        test_metrics = [tm.clone() for tm in train_metrics]
        return train_metrics, val_metrics, test_metrics
    


def log_metrics(experiment, metrics, loss, step, mode="train"):
    if isinstance(metrics, list):  # Multi-task
        for task_idx, task_metrics in enumerate(metrics):
            for name, value in task_metrics.items():
                val = value.compute()
                experiment.log_metric(
                    f"{mode}/{name}_task{task_idx}", val.cpu().detach().numpy().tolist(), step=step
                )
    else:  # Single-task
        for name, value in metrics.items():
            val = value.compute()
            experiment.log_metric(
                f"{mode}/{name}", val.cpu().detach().numpy().tolist(), step=step
            )

    # Log loss (shared across tasks or single)
    experiment.log_metric(f"{mode}/loss", loss if loss is not None else 0, step=step)


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
    
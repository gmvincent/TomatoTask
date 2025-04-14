import os 
import time
import torch

def train_model(
    args,
    model,
    optimizer,
    scheduler,
    criterion,
    train_dataloader,
    epoch,
    train_metrics=None,
    return_preds=False,
):
    model.train()

    y_pred = []
    y_true = []
    
    running_loss = 0
    if train_metrics is None:
        running_correct = 0

    for batch, data in enumerate(train_dataloader):
        instances, labels, _ = data
        instances = instances.to(args.device).float()  
        labels = labels.to(args.device).float()      

        optimizer.zero_grad()

        start_time = time.time()
        output = model(instances)
        end_time = time.time()
        
        loss = criterion(output.squeeze(), labels)
        
        loss.backward(retain_graph=True)
        optimizer.step()

        predictions = output.squeeze()
        running_loss += loss.item() * len(labels.cpu())

        # Update metrics
        if train_metrics is not None:
            for name, metric in train_metrics.items():
                if name == "PredictionTime":
                    metric.update(start_time, end_time)
                else:
                    metric.update(predictions.cpu(), labels.cpu())
        else:
            running_correct += (predictions == labels).sum().item()

        # Store Predictions
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(predictions.detach().cpu().numpy())
        
    
    # Train outputs
    epoch_loss = running_loss / len(train_dataloader.dataset)

    # compute metrics at the end of this epoch
    if train_metrics is not None:
        metrics_dict = train_metrics.compute()
        if args.ddp:
            torch.distributed.all_reduce(metrics_dict)
        epoch_acc = metrics_dict["Accuracy"].item()
    else:
        epoch_acc = running_correct / len(train_dataloader.dataset)

    if return_preds:
        return epoch_loss, epoch_acc, y_true, y_pred
    else:
        return epoch_loss, epoch_acc

def test_model(
    args,
    model,
    optimizer,
    scheduler,
    criterion,
    test_dataloader,
    epoch,
    test_metrics=None,
    return_preds=False,
    task="val",
):

    model.eval()

    y_pred = []
    y_true = []

    running_loss = 0
    if test_metrics is None:
        running_correct = 0

    with torch.no_grad():
        for batch, data in enumerate(test_dataloader):
            instances, labels, _ = data
            instances = instances.to(args.device).float()  
            labels = labels.to(args.device).float()     

            start_time = time.time()
            output = model(instances)
            end_time = time.time()
            
            loss = criterion(output.squeeze(), labels)

            predictions = output.squeeze()
            running_loss += loss.item() * len(labels.cpu())

            # Update metrics
            if test_metrics is not None:
                for name, metric in test_metrics.items():
                    if name == "PredictionTime":
                        metric.update(start_time, end_time)
                    else:
                        metric.update(predictions.cpu(), labels.cpu())
            else:
                running_correct += (predictions == labels).sum().item()

            # Store Predictions
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(predictions.detach().cpu().numpy())

    # Test outputs
    epoch_loss = running_loss / len(test_dataloader.dataset)

    # compute metrics at the end of this epoch
    if test_metrics is not None:
        metrics_dict = test_metrics.compute()
        if args.ddp:
            torch.distributed.all_reduce(metrics_dict)
        epoch_acc = metrics_dict["Accuracy"].item()
    else:
        epoch_acc = running_correct / len(test_dataloader.dataset)

    if return_preds:
        return epoch_loss, epoch_acc, y_true, y_pred
    else:
        return epoch_loss, epoch_acc

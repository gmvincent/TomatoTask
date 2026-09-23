import time
import torch

def train_model(
    args,
    model,
    optimizer,
    criterion,
    train_dataloader,
    epoch,
    train_metrics=None,
    return_preds=False,
):
    model.train()

    y_pred, y_true  = [], []
    running_loss, running_samples = 0, 0
    if train_metrics is None:
        running_correct = 0

    for batch, data in enumerate(train_dataloader):
        instances, labels = data
        instances = instances.to(args.device).float()  
        if args.task == "regression":
            labels = labels.to(args.device).float()
        else:
            labels = labels.to(args.device).long()

        batch_size = instances.size(0)
        
        optimizer.zero_grad()

        start_time = time.time()
        output = model(instances)
        end_time = time.time()
        
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        
        if args.task == "regression":
            preds = output.detach() 
        else:
            preds = output.argmax(dim=1)
        running_loss += loss.item() * batch_size
        running_samples += batch_size

        # Update metrics
        if train_metrics is not None:
            for name, metric in train_metrics.items():
                if name == "PredictionTime":
                    metric.update(start_time, end_time, batch_size=batch_size)
                elif name == "Dice":
                    metric.update(preds, labels)
                else:
                    metric.update(output, labels)
        else:
            running_correct += (preds == labels).sum().item()

        # Store Predictions
        y_true.append(labels)
        y_pred.append(preds)
    
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred).detach()

    # Train outputs
    epoch_loss = (running_loss / running_samples)

    # compute metrics at the end of this epoch
    if train_metrics is not None:
        metrics_dict = train_metrics.compute()
        epoch_acc = metrics_dict["Accuracy"].item() if args.task != "regression" else metrics_dict["MAE"].item()
    else:
        epoch_acc = (running_correct / running_samples) if args.task != "regression" else float("nan")

    if return_preds:
        return epoch_loss, epoch_acc, y_true, y_pred
    else:
        return epoch_loss, epoch_acc

def test_model(
    args,
    model,
    optimizer,
    criterion,
    test_dataloader,
    epoch,
    test_metrics=None,
    return_preds=False,
):
    model.eval()

    y_pred, y_true  = [], []
    running_loss, running_samples = 0, 0
    if test_metrics is None:
        running_correct = 0

    with torch.no_grad():
        for batch, data in enumerate(test_dataloader):
            instances, labels = data
            instances = instances.to(args.device).float()  
            if args.task == "regression":
                labels = labels.to(args.device).float()
            else:
                labels = labels.to(args.device).long()
            
            batch_size = instances.size(0)     

            start_time = time.time()
            output = model(instances)
            end_time = time.time()
            
            loss = criterion(output, labels)

            if args.task == "regression":
                preds = output.detach() 
            else:
                preds = output.argmax(dim=1)
            running_loss += loss.item() * batch_size
            running_samples += batch_size

            # Update metrics
            if test_metrics is not None:
                for name, metric in test_metrics.items():
                    if name == "PredictionTime":
                        metric.update(start_time, end_time, batch_size=batch_size)
                    elif name == "Dice":
                        metric.update(preds, labels)
                    else:
                        metric.update(output, labels)
            else:
                running_correct += (preds == labels).sum().item()

            # Store Predictions
            y_true.append(labels)
            y_pred.append(preds)

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred).detach()
    
    # Test outputs
    epoch_loss = (running_loss / running_samples)

    # compute metrics at the end of this epoch
    if test_metrics is not None:
        metrics_dict = test_metrics.compute()
        epoch_acc = metrics_dict["Accuracy"].item() if args.task != "regression" else metrics_dict["MAE"].item()
    else:
        epoch_acc = (running_correct / running_samples) if args.task != "regression" else float("nan")

    if return_preds:
        return epoch_loss, epoch_acc, y_true, y_pred
    else:
        return epoch_loss, epoch_acc

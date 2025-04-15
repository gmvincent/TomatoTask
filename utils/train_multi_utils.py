import os 
import time
import torch
import numpy as np

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

    if args.num_tasks == "2":
        multi_task = 2
    elif args.num_tasks == "3":
        multi_task = 3
    elif args.num_tasks == "2_tomato":
        multi_task = 2
        
    y_pred = [[] for _ in range(multi_task)]
    y_true = [[] for _ in range(multi_task)]
    
    predictions = [[] for _ in range(multi_task)]
    
    running_loss = 0
    if train_metrics is None:
        running_correct = [0 for _ in range(multi_task)]

    for batch, data in enumerate(train_dataloader):
        
        input = data[0]
        input = input.to(args.device).float()  
        optimizer.zero_grad()
        
        loss = 0
        for idx in range(multi_task):
            labels = data[idx+1]
            labels = labels.to(args.device).long()      

            start_time = time.time()
            output = model(input, task_id=idx)
            end_time = time.time()
            
            task_loss = criterion(output, labels)

            loss += task_loss
            
            predictions[idx] = output
            y_true[idx].extend(labels.detach().cpu().numpy())
            y_pred[idx].extend(torch.argmax(predictions, axis=1).detach().cpu().numpy())
            
            # Update metrics
            if train_metrics is not None:
                for name, metric in train_metrics[idx].items():
                    if name == "PredictionTime":
                        metric.update(start_time, end_time)
                    else:
                        metric.update(predictions[idx].cpu(), labels.cpu())
            else:
                running_correct[idx] += (y_pred[idx] == labels.cpu()).sum().item()
                        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(y_true[0])

    # Train outputs
    epoch_loss = running_loss / len(train_dataloader.dataset)

    # compute metrics at the end of this epoch
    if train_metrics is not None:
        metrics_dict = [tm.compute() for tm in train_metrics]
        if args.ddp:
            for m in metrics_dict:
                torch.distributed.all_reduce(m)
        epoch_acc = [m["Accuracy"].item() for m in metrics_dict]
    else:
        epoch_acc = [correct / len(train_dataloader.dataset) for correct in running_correct]


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

    if args.num_tasks == "2":
        multi_task = 2
    elif args.num_tasks == "3":
        multi_task = 3
    elif args.num_tasks == "2_tomato":
        multi_task = 2
        
    y_pred = [[] for _ in range(multi_task)]
    y_true = [[] for _ in range(multi_task)]
    
    predictions = [[] for _ in range(multi_task)]
    
    running_loss = 0
    if test_metrics is None:
        running_correct = [0 for _ in range(multi_task)]
        
    with torch.no_grad():
        for batch, data in enumerate(test_dataloader):
            input = data[0]
            input = input.to(args.device).float()  
            loss = 0

            for idx in range(multi_task):
                labels = data[idx+1]
                labels = labels.to(args.device).long()      

                start_time = time.time()
                output = model(input, task_id=idx)
                end_time = time.time()

                task_loss = criterion(output, labels)
                loss += task_loss
                
                predictions[idx] = output
                y_true[idx].extend(labels.detach().cpu().numpy())
                y_pred[idx].extend(torch.argmax(predictions, axis=1).detach().cpu().numpy())
                
                # Update metrics
                if test_metrics is not None:
                    for name, metric in test_metrics[idx].items():
                        if name == "PredictionTime":
                            metric.update(start_time, end_time)
                        else:
                            metric.update(predictions[idx].cpu(), labels.cpu())
                else:
                    running_correct[idx] += (y_pred[idx] == labels.cpu()).sum().item()
                    
                    
            running_loss += loss.item() * len(y_true[0])

    # Test outputs
    epoch_loss = running_loss / len(test_dataloader.dataset)

    # compute metrics at the end of this epoch
    if test_metrics is not None:
        metrics_dict = [tm.compute() for tm in test_metrics]
        if args.ddp:
            for m in metrics_dict:
                torch.distributed.all_reduce(m)
        epoch_acc = [m["Accuracy"].item() for m in metrics_dict]
    else:
        epoch_acc = [correct / len(test_dataloader.dataset) for correct in running_correct]


    if return_preds:
        return epoch_loss, epoch_acc, y_true, y_pred
    else:
        return epoch_loss, epoch_acc
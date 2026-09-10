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

    # number of tasks
    multi_task = len(args.num_classes)
    
    y_pred = [[] for _ in range(multi_task)]
    y_true = [[] for _ in range(multi_task)]
        
    running_loss, running_samples = 0, 0
    if train_metrics is None:
        running_correct = [0 for _ in range(multi_task)]

    for batch, data in enumerate(train_dataloader):
        
        input = data[0]
        input = input.to(args.device).float()  
        
        batch_size = input.size(0)
        optimizer.zero_grad()
        
        loss = 0
        for idx in range(multi_task):
            labels = data[idx+1]
            labels = labels.to(args.device).long()      
            
            start_time = time.time()
            output = model(input, task_idx=idx)
            end_time = time.time()
            
            task_loss = criterion(output, labels)
            loss += task_loss
            
            preds = output.argmax(dim=1)
            y_true[idx].append(labels)
            y_pred[idx].append(preds)
            
            # Update metrics
            if train_metrics is not None:
                for name, metric in train_metrics[idx].items():
                    if name == "PredictionTime":
                        metric.update(start_time, end_time, batch_size=batch_size)
                    else:
                        metric.update(output, labels)
            else:
                running_correct[idx] += (preds == labels).sum().item()
                        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_size
        running_samples += batch_size
        
    y_true = [torch.cat(task_labels) for task_labels in y_true]
    y_pred = [torch.cat(task_preds).detach() for task_preds in y_pred]
    
    # Train outputs
    epoch_loss = (running_loss / running_samples)
    
    # compute metrics at the end of this epoch
    if train_metrics is not None:
        metrics_dict = [tm.compute() for tm in train_metrics]
        epoch_acc = [m["Accuracy"].item() for m in metrics_dict]
    else:
        epoch_acc = [correct / running_samples for correct in running_correct]


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

    # number of tasks
    multi_task = len(args.num_classes)
        
    y_pred = [[] for _ in range(multi_task)]
    y_true = [[] for _ in range(multi_task)]
        
    running_loss, running_samples = 0, 0
    if test_metrics is None:
        running_correct = [0 for _ in range(multi_task)]
        
    with torch.no_grad():
        for batch, data in enumerate(test_dataloader):
            input = data[0]
            input = input.to(args.device).float()  
            
            batch_size = input.size(0)
            
            loss = 0

            for idx in range(multi_task):
                labels = data[idx+1]
                labels = labels.to(args.device).long()      

                start_time = time.time()
                output = model(input, task_idx=idx)
                end_time = time.time()

                task_loss = criterion(output, labels)
                loss += task_loss
                
                preds = output.argmax(dim=1)
                y_true[idx].append(labels)
                y_pred[idx].append(preds)
                
                # Update metrics
                if test_metrics is not None:
                    for name, metric in test_metrics[idx].items():
                        if name == "PredictionTime":
                            metric.update(start_time, end_time, batch_size=batch_size)
                        else:
                            metric.update(output, labels)
                else:
                    running_correct[idx] += (preds == labels).sum().item()

            running_loss += loss.item() * batch_size
            running_samples += batch_size

    y_true = [torch.cat(task_labels) for task_labels in y_true]
    y_pred = [torch.cat(task_preds).detach() for task_preds in y_pred]
    
    # Test outputs
    epoch_loss = (running_loss / running_samples)

    # compute metrics at the end of this epoch
    if test_metrics is not None:
        metrics_dict = [tm.compute() for tm in test_metrics]
        epoch_acc = [m["Accuracy"].item() for m in metrics_dict]
    else:
        epoch_acc = [correct / running_samples for correct in running_correct]


    if return_preds:
        return epoch_loss, epoch_acc, y_true, y_pred
    else:
        return epoch_loss, epoch_acc
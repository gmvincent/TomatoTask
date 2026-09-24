import warnings
import os

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:comet_ml"

import comet_ml
#from comet_ml.integration.pytorch import log_model

import os
import torch
import signal
import numpy as np
from tqdm import tqdm

import torch.multiprocessing as mp
import torch.distributed as dist

from utils.cometml_logger import create_experiment, log_experiment, log_model_weights, plot_distribution
from utils import parse_args, setup_ddp, cleanup_ddp, set_seed, handle_sigterm
from utils.metrics import initialize_metrics, gather_tensor
from utils.early_stopping import EarlyStopping
from utils.class_weights import class_pixel_counts, class_weights

from models.model_hub import get_model
from data_utils import create_data_loader

# Train and Test Functions
from utils.train_single_utils import train_model as train_single_model
from utils.train_single_utils import test_model as test_single_model

from utils.train_multi_utils import train_model as train_multi_model
from utils.train_multi_utils import test_model as test_multi_model

model_functions = {
    "single": (train_single_model, test_single_model),
    "multi": (train_multi_model, test_multi_model),
    #"single_trad": (train_stat_model, test_stat_model) #TODO: setup traditional machine learning models for signle_task
}

def main_worker(rank, args):
    try:
        # set random seeds
        set_seed(args.random_seed, rank, True)
        
        if args.ddp:
            setup_ddp(args, rank)
        if torch.cuda.is_available():
            args.device = torch.device(f"cuda:{args.gpu[rank]}")
        
        if rank == 0:
            print(f"Created Exp: {rank}")
            experiment = create_experiment(args)
        else:
            experiment = None
        
        # create dataloaders
        if rank==0: print("Creating Dataloaders", flush=True)
        train_dataloader, val_dataloader, test_dataloader, classes_dict  = create_data_loader(args, rank) 
        
        if not isinstance(args.task, list): # single-task
            args.classes = list(range(len(classes_dict)+3)) if args.task == "regression" else list(classes_dict.values())
            args.num_classes = len(args.classes)
        else: # multi-task
            args.classes = []
            for t in args.task:
                if t == "regression":
                    task_classes = list(range(len(classes_dict[t])+3))
                else:
                    task_classes = list(classes_dict[t].values())    
                args.classes.append(task_classes) # list of each task classes
            args.num_classes = [len(class_lst) for class_lst in args.classes]
        
        if isinstance(args.task, list) and len(args.task) != len(args.num_classes):
            raise ValueError(f"len(task)={len(args.task)} != len(num_classes)={len(args.num_classes)}.")
        
        _, _, _ = main(args, experiment, [train_dataloader, val_dataloader, test_dataloader], rank)
        
    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        raise
    
    finally:  
        if args.ddp:
            cleanup_ddp()
        
        if rank == 0 and experiment is not None:
            experiment.end()
    
def main(args, experiment, dataloaders, rank):
    single_task = not isinstance(args.task, list)
    
    # load dataloaders
    train_dataloader, val_dataloader, test_dataloader = dataloaders
    
    if rank == 0 and args.dataset_name not in ["spirals", "graph_meshes"]:
        if args.ddp:
            plot_train_loader = torch.utils.data.DataLoader(
                train_dataloader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
            )
            plot_val_loader = torch.utils.data.DataLoader(
                val_dataloader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
            )
            plot_test_loader = torch.utils.data.DataLoader(
                test_dataloader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
            )
        else:
            plot_train_loader = train_dataloader
            plot_val_loader = val_dataloader
            plot_test_loader = test_dataloader

        plot_distribution(args, experiment, plot_train_loader, mode="train")
        plot_distribution(args, experiment, plot_val_loader, mode="val")
        plot_distribution(args, experiment, plot_test_loader, mode="test")
        
    seg_weights = None
    if single_task and args.task == "segmentation":
        counts = class_pixel_counts(train_dataloader, args.task, args.num_classes, args.device)
        seg_weights = class_weights(counts[0], scheme="inverse_sqrt").to(args.device)
        if rank == 0:
            experiment.log_parameters(
                {f"class_{c}_weight": w for c, w in enumerate(seg_weights.tolist())}
            )
    elif not single_task and "segmentation" in args.task:
        counts = class_pixel_counts(train_dataloader, args.task, args.num_classes, args.device)
        seg_weights = {
            i: class_weights(c, scheme="inverse_sqrt").to(args.device) for i, c in counts.items()
        }
        if rank == 0:
            experiment.log_parameters({
                f"task_{i}_class_{c}_weight": w
                for i, w_task in seg_weights.items()
                for c, w in enumerate(w_task.tolist())
            })
    
    if args.ddp: dist.barrier(device_ids=[args.gpu[rank]])
       
    batch = next(iter(train_dataloader))  
    images = batch[0]
    args.input_channels = images.shape[1]
    
    # get model by name
    model = get_model(args, args.model_name, single_task)
    
    # multi-gpu training
    if args.dataparallel:
        model = torch.nn.DataParallel(model)
    elif args.ddp:
        model.to(args.device)
        #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu[rank]],
            output_device=args.gpu[rank],
        )
    else:
        model.to(args.device)
        
    # Train Model
    if single_task:
        if args.model_name not in ["svm", "rf"]: #TODO: set-up traditional machine learning models
            train_model, test_model = model_functions["single"]
    else:
        train_model, test_model = model_functions["multi"]
    
    # Initialize metrics
    train_metrics, val_metrics, test_metrics = initialize_metrics(args)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=10, min_delta=1e-3)
    
    if rank==0: print("Begin Training", flush=True)

    # Train/Fit the Models
    if args.model_name not in ["svm", "rf"]:
        
        criterion_map = {
            "regression": torch.nn.MSELoss(),
            "classification": torch.nn.CrossEntropyLoss(),
            "segmentation": torch.nn.CrossEntropyLoss(weight=seg_weights),
        }
        if single_task:
            criterion = criterion_map[args.task]
        else:
            # TODO: fix the segmentation loss weighting for multi-task
            criterion = [criterion_map[t]() for t in args.task]
        
        if args.optimizer_name == 'Adam':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.25) 
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
        
        model = model.to(args.device)        
        
        if args.ddp: dist.barrier(device_ids=[args.gpu[rank]]) # synchronize before starting training
        
        if rank == 0:
            epoch_pbar = tqdm(range(args.epochs), total=args.epochs, desc=f"Training Model (rank {rank})", unit="epoch") 
        else:
            epoch_pbar = range(args.epochs)
        for epoch in epoch_pbar:  
            if args.ddp:
                train_dataloader.sampler.set_epoch(epoch)  
                val_dataloader.sampler.set_epoch(epoch)                 
            
            # reset metrics for this epoch
            if single_task:
                train_metrics.reset()
                val_metrics.reset()
                test_metrics.reset()
            else:
                [tm.reset() for tm in train_metrics]
                [vm.reset() for vm in val_metrics]
                [tm.reset() for tm in test_metrics]
                
            train_loss, train_acc, y_true_train, y_pred_train = train_model(
                args,
                model,
                optimizer,
                criterion,
                train_dataloader,
                epoch,
                train_metrics,
                return_preds=True,
            )

            val_loss, val_acc, y_true, y_pred = test_model(
                args,
                model,
                optimizer,
                criterion,
                val_dataloader,
                epoch,
                val_metrics,
                return_preds=True,
            )        
            
            if args.ddp:
                dist.all_reduce(train_loss_t := torch.tensor(train_loss, device=args.device), op=dist.ReduceOp.AVG)
                dist.all_reduce(val_loss_t := torch.tensor(val_loss, device=args.device), op=dist.ReduceOp.AVG)
                
                train_loss, val_loss = train_loss_t.item(), val_loss_t.item()
                
                if (epoch >= args.epochs - 1) or (epoch % args.print_freq == 0):
                    # train gather
                    y_true_train = gather_tensor(args, y_true_train)
                    y_pred_train = gather_tensor(args, y_pred_train)
                    
                    # val gather
                    y_true = gather_tensor(args, y_true)
                    y_pred = gather_tensor(args, y_pred)
                        
            stop_signal = torch.tensor(0, device=args.device)
            
            # Output intermediate statistics
            if rank == 0 and experiment is not None:
                log_experiment(args, experiment, model, train_dataloader, train_metrics, train_loss, epoch, y_true_train, y_pred_train, mode="train")
                log_experiment(args, experiment, model, val_dataloader, val_metrics, val_loss, epoch, y_true, y_pred, mode="val")
                
                # Log learning rate for this epoch
                current_lr = optimizer.param_groups[0]['lr']
                experiment.log_metric(f"learning_rate", current_lr, step=epoch)                
                     
                epoch_pbar.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss, "Val acc": val_acc})
                
                # Check Early Stopping
                if epoch > 1:
                    early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                        if isinstance(val_acc, list):
                            acc_str = ", ".join([f"{a:.3f}" for a in val_acc])
                        else:
                            acc_str = f"{val_acc:.5f}"
                        print(
                            f"\nStopped at Epoch: {epoch} \tVal Accuracy: {acc_str} \tVal Loss: {val_loss:.5f}"
                        )
                        stop_signal += 1
            if args.ddp:
                dist.broadcast(stop_signal, src=0)
                
            if stop_signal.item() > 0:
                if args.ddp:
                    if rank == 0:
                        model.module.load_state_dict(early_stopping.best_weights)
                    for tensor in model.module.state_dict().values():
                        dist.broadcast(tensor, src=0)
                else:
                    model.load_state_dict(early_stopping.best_weights)

                args.epochs = epoch + 1
                break
            
            # Step lr scheduler
            scheduler.step(val_loss)
        
        if args.ddp: dist.barrier(device_ids=[args.gpu[rank]])
        
        # Test model
        test_loss, test_acc, y_true, y_pred = test_model(
            args,
            model,
            optimizer,
            criterion,
            test_dataloader,
            epoch,
            test_metrics,
            return_preds=True,
        ) 
        if args.ddp:
            y_true = gather_tensor(args, y_true)
            y_pred = gather_tensor(args, y_pred)        
            dist.all_reduce(test_loss_t := torch.tensor(test_loss, device=args.device), op=dist.ReduceOp.AVG)
            test_loss = test_loss_t.item()
               
        if rank == 0 and experiment is not None:
            log_experiment(args, experiment, model, test_dataloader, test_metrics, test_loss, epoch, y_true, y_pred, mode="test")          
            if isinstance(test_acc, list):
                acc_str = ", ".join([f"{a:.3f}" for a in test_acc])
            else:
                acc_str = f"{test_acc:.5f}"
            print(f"\nFinished Training: \tTest Accuracy: {acc_str} \tTest Loss: {test_loss:.5f}")
            
            log_model_weights(args, experiment, model)
        
    else:
        #TODO: set up traditional machine learning models
        train_metrics.reset()
        val_metrics.reset()
        test_metrics.reset()

        train_loss, train_acc, y_train, y_pred_train = train_model(args, model, train_dataloader, args.epochs - 1, train_metrics, return_preds=True)
        val_loss, val_acc, y_val, y_pred_val = test_model(args, model, val_dataloader, args.epochs - 1, val_metrics, return_preds=True, task="val")
        test_loss, test_acc, y_test, y_pred = test_model(args, model, test_dataloader, args.epochs - 1, test_metrics, return_preds=True, task="test")
        
        # Log experiments
        log_experiment(args, experiment, train_metrics, train_loss, args.epochs - 1, y_train, y_pred_train, mode="train")
        log_experiment(args, experiment, val_metrics, val_loss, args.epochs - 1, y_val, y_pred_val, mode="val")
        log_experiment(args, experiment, test_metrics, test_loss, args.epochs - 1, y_test, y_pred, mode="test")

    print("End Training", flush=True)

    return train_loss, val_loss, test_loss


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args("./configs/default_config.yaml", desc="single_task")
    
    #os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    
    # set rank
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0
    
    # set device
    if args.gpu and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu))

        if args.dataparallel or args.ddp:
            if rank==0: print(f"Multi-GPU enabled. Using GPUs: {args.gpu}")
            args.gpu = list(range(len(args.gpu))) 
        else:
            print(f"Single-GPU mode. Using GPU {args.gpu[0]}")
            torch.cuda.set_device(f"cuda:{args.gpu[rank]}")
            args.device = torch.device(f"cuda:{args.gpu[rank]}")
    else:
        args.device = torch.device("cpu")
        print("Using CPU.")

    if args.ddp:
        args.world_size = len(args.gpu)
        args.lr = args.lr * args.world_size
        args.batch_size = args.batch_size // args.world_size
        
        signal.signal(signal.SIGTERM, handle_sigterm)
        signal.signal(signal.SIGINT, handle_sigterm)
        
        try:
            mp.spawn(main_worker, args=(args,), nprocs=args.world_size)
        except Exception as e:
            print(f"Training failed: {e}")
            if dist.is_initialized():
                dist.destroy_process_group()
            raise
    
    else:
        main_worker(0, args)
        
    torch.cuda.empty_cache()
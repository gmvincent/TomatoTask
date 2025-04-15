import comet_ml
from comet_ml.integration.pytorch import log_model

import optuna

import cProfile
import pstats

import os
import time
import torch
import random
import numpy as np
from tqdm import tqdm

import torch.multiprocessing as mp
import torch.distributed as dist

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from utils.cometml_logger import create_experiment, log_experiment, log_model_weights, plot_distribution
from utils import parse_args, setup_ddp, cleanup_ddp
from utils.metrics import initialize_metrics, log_metrics
from utils.early_stopping import EarlyStopping

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
    args.rank = rank
    
    if args.ddp:
        setup_ddp(args, rank)
        # Ensure each process only uses the assigned GPU
        torch.cuda.set_device(rank)
        args.device = torch.device(f'cuda:{rank}')
                
        dist.barrier()
    
    if rank == 0:
        print(f"Created Exp: {rank}")
        experiment = create_experiment(args)
    else:
        experiment = None
    
    # create dataloaders
    train_dataloader, val_dataloader, test_dataloader, classes_dict  = create_data_loader(args) 
    if args.num_tasks in ["1", "tomato"]:
        args.classes = list(classes_dict.values())
        args.num_classes = len(args.classes)
    else:
        args.classes = [[v for v in inner.values()] for inner in classes_dict.values()] # list of each task classes
    
    _, _, _ = main(args, experiment, [train_dataloader, val_dataloader, test_dataloader], rank)
    
    if args.ddp:
        cleanup_ddp()
    
    if rank == 0 and experiment is not None:
        experiment.end()
    
def main(args, experiment, dataloaders, rank):
    single_task = False
    if args.num_tasks in ["1", "tomato"]:
        single_task = True
        
    # load dataloaders
    train_dataloader, val_dataloader, test_dataloader = dataloaders
    
    #if rank == 0:
    #    plot_distribution(args, experiment, train_dataloader, args.classes, mode="train")
    #    plot_distribution(args, experiment, val_dataloader, args.classes, mode="val")
    #    plot_distribution(args, experiment, test_dataloader, args.classes, mode="test")
    
    
    # get number of features
    images, _ = next(iter(train_dataloader)) 
    args.input_channels = images.shape[1]
    
    # get model by name
    model = get_model(args, args.model_name, single_task)
    
    # multi-gpu training
    if args.dataparallel:
        model = torch.nn.DataParallel(model)
    elif args.ddp:
        torch.cuda.set_device(rank)
        model.to(torch.device(f"cuda:{rank}"))

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
        )
    else:
        model.to(args.device)
        
    # Train Model
    if single_task:
        if args.model_name not in ["svm", "rf"]: #TODO: set-up traditional machine learning models
            train_model, test_model = model_functions["single"]
    elif not single_task:
        train_model, test_model = model_functions["multi"]
    
    # Initialize metrics
    train_metrics, val_metrics, test_metrics = initialize_metrics(args)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=25, min_delta=1e-4)
    
    print("Begin Training", flush=True)

    # Train/Fit the Models
    if args.model_name not in ["svm", "rf"]:
        criterion = torch.nn.CrossEntropyLoss()
        
        if args.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.25) 
        
        model = model.to(args.device)        
        
        if args.ddp:
            dist.barrier() # synchronize before starting training
        
        epoch_pbar = tqdm(range(args.epochs), total=args.epochs, desc=f"Training Model (rank {args.rank})", unit="epoch") 
        for epoch in epoch_pbar:  
            if args.ddp:
                train_dataloader.sampler.set_epoch(epoch)  
                val_dataloader.sampler.set_epoch(epoch)                 
            
            # reset metrics for this epoch
            train_metrics.reset()
            val_metrics.reset()
            test_metrics.reset()
            
            train_loss, train_acc, y_true_train, y_pred_train = train_model(
                args,
                model,
                optimizer,
                scheduler,
                criterion,
                train_dataloader,
                epoch,
                train_metrics,
                return_preds=True,
            )
            
            if rank == 0 and experiment is not None:
                log_experiment(args, experiment, train_metrics, train_loss, epoch, y_true_train, y_pred_train, mode="train")

                # Log learning rate for this epoch
                current_lr = optimizer.param_groups[0]['lr']
                experiment.log_metric(f"learning_rate", current_lr, step=epoch)
                
                if (epoch >= args.epochs-1) or (epoch % args.print_freq == 0):
                    print(
                        "\nEpoch: %d \t Accuracy: %.5f \tLoss: %.5f" % (epoch, train_acc, train_loss), flush=True
                    )
            
                    #TODO: plot class activation maps
                    
                        
            val_loss, val_acc, y_true, y_pred = test_model(
                args,
                model,
                optimizer,
                scheduler,
                criterion,
                val_dataloader,
                epoch,
                val_metrics,
                return_preds=True,
            )        

            if rank == 0 and experiment is not None:
                log_experiment(args, experiment, val_metrics, val_loss, epoch, y_true, y_pred, mode="val")

                # Step lr scheduler
                scheduler.step(val_loss)
                #dist.barrier()
                
                # Output intermediate statistics
                if ((epoch >= args.epochs - 1) or (epoch % args.print_freq == 0)):
                    print("\nVal Accuracy: %.5f \tVal Loss: %.5f" % (val_acc, val_loss), flush=True)
            
                epoch_pbar.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss, "Val acc": val_acc})
                
                # Check Early Stopping
                if epoch > 1:
                    early_stopping(val_loss, model, epoch, experiment)
                    if early_stopping.early_stop:
                        print(
                            "\nStopped at Epoch: %d \tVal Accuracy: %.5f \tVal Loss: %.5f" % (epoch, val_acc, val_loss)
                        )
                        model.load_state_dict(early_stopping.best_weights)
                        args.epochs = epoch + 1
                        break
        
        # Test model
        test_loss, test_acc, y_true, y_pred = test_model(
            args,
            model,
            optimizer,
            scheduler,
            criterion,
            test_dataloader,
            epoch,
            test_metrics,
            return_preds=True,
            task="test",
        ) 
               
        if rank == 0 and experiment is not None:
            log_experiment(args, experiment, test_metrics, test_loss, epoch, y_true, y_pred, mode="test")

            print("\nFinished Training: \tTest Accuracy: %.5f \tTest Loss: %.5f" % (test_acc, test_loss))
            
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
    
    args = parse_args("./configs/default_config.yaml", desc="single_task")
    
    # set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    
    # set device
    if args.dataparallel or args.ddp:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu))
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Multi-GPU enabled. Using GPUs: {args.gpu}")
    else:
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu[0])
        args.device = torch.device(f"cuda:{args.gpu[0]}" if torch.cuda.is_available() and args.gpu else "cpu")
        print(f"Single-device mode. Using device: {args.device}")
    
    if args.ddp:
        args.world_size = len(args.gpu)
        args.lr = args.lr * args.world_size
        args.batch_size = args.batch_size * args.world_size
        
        mp.spawn(main_worker, args=((args,)), nprocs=args.world_size,)
    else:
        # Single GPU or DataParallel logic
        main_worker(0, args)    
        
    torch.cuda.empty_cache()

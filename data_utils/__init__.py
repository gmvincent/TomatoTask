import os
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torchvision.transforms as transforms

from data_utils.plantvillage import PlantVillage
from data_utils.plantvillage import single_task_task as pv_single_class
from data_utils.plantvillage import multi_task as pv_multi_class
from data_utils.plantvillage import tomato_task as pv_tomato_class

from data_utils.plantdoc import PlantDoc
from data_utils.plantdoc import single_task_task as pd_single_class
from data_utils.plantdoc import multi_task as pd_multi_class
from data_utils.plantdoc import tomato_task as pd_tomato_class

from data_utils.aichallenger import AIChallenger
from data_utils.aichallenger import single_task_task as ac_single_class
from data_utils.aichallenger import multi_task as ac_multi_class
from data_utils.aichallenger import tomato_task as ac_tomato_class

def create_data_loader(args):
    # set random seed
    torch.manual_seed(args.random_seed)
    
    # normalize - ImageNet Values
    normal = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
    
    # Set transforms
    train_augmentations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (64, 64), interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            normal,
        ],
    )
    
    val_augmentations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (64, 64), interpolation=transforms.InterpolationMode.NEAREST
            ),
            normal,
        ],
    )
    
    train_ds, val_ds, test_ds, classes = get_datasets(args, args.dataset_name, train_augmentations, val_augmentations)
    
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=False,
        persistent_workers=False,
        shuffle=True,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_ds,
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=True,
        persistent_workers=False,
        shuffle=False,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds,
        num_workers=0,
        batch_size=args.batch_size,
        drop_last=True,
        persistent_workers=False,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, classes
    
def get_datasets(args, dataset_name, train_augs, val_augs):
    
    if dataset_name.lower() == "plantvillage":
        
        data_path = "PlantVillage_Dataset"
        train_ds = PlantVillage(
                    root=data_path,
                    split="Train",
                    num_tasks=args.num_tasks,
                    transform=train_augs,
                )

        val_ds = PlantVillage(
                    root=data_path,
                    split="Val",
                    num_tasks=args.num_tasks,
                    transform=val_augs,
                )

        test_ds = PlantVillage(
                    root=data_path,
                    split="Test",
                    num_tasks=args.num_tasks,
                    transform=val_augs,
                )

        if args.num_tasks == "1":
            classes = pv_single_class
        elif args.num_tasks == "2":
            classes = pv_multi_class
        elif args.num_tasks == "tomato":
            classes = pv_tomato_class
        
    elif dataset_name.lower() == "plantdoc":
        
        data_path = "PlantDoc_Dataset"
        full_ds = PlantDoc(
                    root=data_path,
                    split="train",
                    num_tasks=args.num_tasks,
                    transform=val_augs,
                )
        
        total_size = len(full_ds)
        indices = list(range(total_size))
        train_indices, val_indices = train_test_split(indices, test_size=0.3, random_state=args.random_seed)

        
        train_ds = torch.utils.data.Subset(
            PlantDoc(
                    root=data_path,
                    split="train",
                    num_tasks=args.num_tasks,
                    transform=train_augs,
                ), 
            train_indices)
        
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        
        test_ds = PlantDoc(
                    root=data_path,
                    split="test",
                    num_tasks=args.num_tasks,
                    transform=val_augs,
                )
        
        if args.num_tasks == "1":
            classes = pd_single_class
        elif args.num_tasks == "2":
            classes = pd_multi_class
        elif args.num_tasks == "tomato":
            classes = pd_tomato_class
            
    elif dataset_name.lower() == "aichallenger":
        
        data_path = "aichallenger"
        full_ds = AIChallenger(
                    root=data_path,
                    split="training_set",
                    num_tasks=args.num_tasks,
                    transform=val_augs,
                )
        
        total_size = len(full_ds)
        indices = list(range(total_size))
        train_indices, val_indices = train_test_split(indices, test_size=0.3, random_state=args.random_seed)

        
        train_ds = torch.utils.data.Subset(
            AIChallenger(
                    root=data_path,
                    split="training_set",
                    num_tasks=args.num_tasks,
                    transform=train_augs,
                ), 
            train_indices)
        
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        
        test_ds = AIChallenger(
                    root=data_path,
                    split="testing_set",
                    num_tasks=args.num_tasks,
                    transform=val_augs,
                )
        
        if args.num_tasks == "1":
            classes = ac_single_class
        elif args.num_tasks == "3":
            classes = ac_multi_class
        elif args.num_tasks == "2_tomato":
            classes = ac_tomato_class
        
    elif dataset_name.lower() == "tomatotask":
        #TODO: finish developing tomatotask dataloader
        print("Dataset not ready!")
        
    
    return train_ds, val_ds, test_ds, classes
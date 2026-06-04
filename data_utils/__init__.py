import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torchvision.transforms as transforms

from data_utils.plantvillage import PlantVillage
from data_utils.plantvillage import single_task as pv_single_class
from data_utils.plantvillage import multi_task as pv_multi_class
from data_utils.plantvillage import tomato_task as pv_tomato_class

from data_utils.plantdoc import PlantDoc
from data_utils.plantdoc import single_task as pd_single_class
from data_utils.plantdoc import multi_task as pd_multi_class
from data_utils.plantdoc import tomato_task as pd_tomato_class

from data_utils.aichallenger import AIChallenger
from data_utils.aichallenger import single_task as ac_single_class
from data_utils.aichallenger import multi_task as ac_multi_class
from data_utils.aichallenger import tomato_task as ac_tomato_class

from data_utils.morphology_data import MorphologyDataset
from data_utils.morphology_data import classes as morphology_classes

from data_utils.tomatotask_2d import TomatoTask2D
from data_utils.tomatotask_2d import classes as sideview_classes

from data_utils.tomatotask_3d import TomatoTask3D
from data_utils.tomatotask_3d import classes as mesh_classes
from data_utils.tomatotask_3d import mesh_to_graph

TOMATOTASK3D_types = {
    "meshes": "mesh",
    "spirals": "spiral",
    "pointclouds": "pcd",
    "graph_meshes": "graph",
}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def create_data_loader(args, rank):
    # set random seed
    torch.manual_seed(args.random_seed)
    
    # normalize - ImageNet Values
    RGB_MEAN, RGB_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    img_size = (224, 224)
    if args.dataset_name == "tomatotask2d_rgbd":
        normalize = transforms.Normalize(RGB_MEAN + [0.0], RGB_STD + [1.0])
    else:
        normalize = transforms.Normalize(RGB_MEAN, RGB_STD)

    # Set transforms
    train_transforms = [
        transforms.ToTensor(),
        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        #transforms.RandomGrayscale(p=0.2),
        #transforms.GaussianBlur(kernel_size=3),
        #transforms.RandomHorizontalFlip(p=0.2),
        #transforms.RandomVerticalFlip(p=0.2),
        #transforms.RandomRotation(degrees=15),
        transforms.Resize(img_size),
        normalize
    ]

    val_transforms = [transforms.ToTensor(), transforms.Resize(img_size), normalize]
    
    if args.dataset_name == "voxel_images":
        train_augmentations = None #augment_voxel
        val_augmentations = None
    else:
        train_augmentations = transforms.Compose(train_transforms)
        val_augmentations = transforms.Compose(val_transforms)
    
    train_ds, val_ds, test_ds, classes = get_datasets(args, rank, args.dataset_name, train_augmentations, val_augmentations)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=args.world_size, rank=rank, shuffle=True) if args.ddp else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=args.world_size, rank=rank, shuffle=False) if args.ddp else None
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds, num_replicas=args.world_size, rank=rank, shuffle=False) if args.ddp else None
    
    g = torch.Generator()
    g.manual_seed(args.random_seed + rank)
    base_loader_args = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "worker_init_fn": seed_worker,
        "generator": g,
    }
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        sampler=train_sampler, 
        shuffle=(train_sampler is None),
        drop_last=args.ddp,
        **base_loader_args,
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_ds,
        sampler=val_sampler, 
        shuffle=False,
        **base_loader_args,
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds,
        sampler=test_sampler, 
        shuffle=False,
        **base_loader_args,
    )

    return train_loader, val_loader, test_loader, classes
    
def get_datasets(args, rank, dataset_name, train_augs, val_augs):
    
    if dataset_name.lower() == "plantvillage":
        
        data_path = "PlantVillage_Dataset"
        train_ds = PlantVillage(
                    root=os.path.join(args.root, data_path),
                    split="Train",
                    num_tasks=args.num_tasks,
                    transform=train_augs,
                )

        val_ds = PlantVillage(
                    root=os.path.join(args.root, data_path),
                    split="Val",
                    num_tasks=args.num_tasks,
                    transform=val_augs,
                )

        test_ds = PlantVillage(
                    root=os.path.join(args.root, data_path),
                    split="Test",
                    num_tasks=args.num_tasks,
                    transform=val_augs,
                )

        if args.num_tasks == "1":
            classes = pv_single_class
        elif args.num_tasks in ["2", "1v2"]:
            classes = pv_multi_class
        elif args.num_tasks == "tomato":
            classes = pv_tomato_class
        
    elif dataset_name.lower() == "plantdoc":
        
        data_path = "PlantDoc_Dataset"
        full_ds = PlantDoc(
                    root=os.path.join(args.root, data_path),
                    split="train",
                    num_tasks=args.num_tasks,
                    transform=val_augs,
                )
        
        total_size = len(full_ds)
        indices = list(range(total_size))
        train_indices, val_indices = train_test_split(indices, test_size=0.3, random_state=args.random_seed)
        
        train_ds = torch.utils.data.Subset(
            PlantDoc(
                    root=os.path.join(args.root, data_path),
                    split="train",
                    num_tasks=args.num_tasks,
                    transform=train_augs,
                ), 
            train_indices)
        
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        
        test_ds = PlantDoc(
                    root=os.path.join(args.root, data_path),
                    split="test",
                    num_tasks=args.num_tasks,
                    transform=val_augs,
                )
        
        if args.num_tasks == "1":
            classes = pd_single_class
        elif args.num_tasks in ["2", "1v2"]:
            classes = pd_multi_class
        elif args.num_tasks == "tomato":
            classes = pd_tomato_class
            
    elif dataset_name.lower() == "aichallenger":
        
        data_path = "aichallenger"
        full_ds = AIChallenger(
                    root=os.path.join(args.root, data_path),
                    split="training_set",
                    num_tasks=args.num_tasks,
                    transform=val_augs,
                )
        
        total_size = len(full_ds)
        indices = list(range(total_size))
        train_indices, val_indices = train_test_split(indices, test_size=0.3, random_state=args.random_seed)

        train_ds = torch.utils.data.Subset(
            AIChallenger(
                    root=os.path.join(args.root, data_path),
                    split="training_set",
                    num_tasks=args.num_tasks,
                    transform=train_augs,
                ), 
            train_indices)
        
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        
        test_ds = AIChallenger(
                    root=os.path.join(args.root, data_path),
                    split="testing_set",
                    num_tasks=args.num_tasks,
                    transform=val_augs,
                )
        
        if args.num_tasks == "1":
            classes = ac_single_class
        elif args.num_tasks in ["2", "1v2"]:
            classes = ac_multi_class
        elif args.num_tasks == "2_tomato":
            classes = ac_tomato_class
        
    elif dataset_name.lower() == "tomatotask":
        #TODO: finish developing tomatotask dataloader
        if rank==0: print("Dataset not ready!")
        
    
    elif dataset_name.lower() in TOMATOTASK3D_types:
        
        data_path = "cmwilli5_drive/gmvincen_data/TomatoTask_Datasets/TomatoTask-3D"
        
        representation = TOMATOTASK3D_types[dataset_name.lower()]
        
        train_ds = TomatoTask3D(
                    root=os.path.join(args.root, data_path),
                    split="train",
                    target_faces=args.target_faces,
                    representation=representation,
                )
        
        val_ds = TomatoTask3D(
                    root=os.path.join(args.root, data_path),
                    split="val",
                    target_faces=args.target_faces,
                    representation=representation,
                )
        
        test_ds = TomatoTask3D(
                    root=os.path.join(args.root, data_path),
                    split="test",
                    target_faces=args.target_faces,
                    representation=representation,
                )
        
        total_size = len(train_ds) + len(val_ds) + len(test_ds)
        if rank==0: print(f"Total Dataset Size: {total_size}")

        classes = mesh_classes
    
    elif dataset_name.lower() == "morphology_features":
        
        data_path = "cmwilli5_drive/gmvincen_data/TomatoTask_Datasets/TomatoTask"
        full_ds = MorphologyDataset(
                    root=os.path.join(args.root, data_path),
                    split="train",
                    transform=None,
                )
        
        val_ds = MorphologyDataset(
                    root=os.path.join(args.root, data_path),
                    split="val",
                    transform=None,
                )
        
        test_ds = MorphologyDataset(
                    root=os.path.join(args.root, data_path),
                    split="test",
                    transform=None,
                )
        
        total_size = len(train_ds) + len(val_ds) + len(test_ds)
        if rank==0: print(f"Total Dataset Size: {total_size}")

        classes = morphology_classes
        
    elif dataset_name.lower() in ["tomatotask2d_rgb", "tomatotask2d_rgbd"]:
        data_path = "cmwilli5_drive/gmvincen_data/TomatoTask_Datasets/TomatoTask-2D"
        
        train_ds = TomatoTask2D(
                    root=os.path.join(args.root, data_path),
                    split="train",
                    transform=train_augs,
                    depth=True if dataset_name.lower() == "tomatotask2d_rgbd" else False,
                    num_tasks=args.num_tasks,
                    num_views=args.num_views,
                )
        
        val_ds = TomatoTask2D(
                    root=os.path.join(args.root, data_path),
                    split="val",
                    transform=val_augs,
                    depth=True if dataset_name.lower() == "tomatotask2d_rgbd" else False,
                    num_tasks=args.num_tasks,
                    num_views=args.num_views,
                )
        
        test_ds = TomatoTask2D(
                    root=os.path.join(args.root, data_path),
                    split="test",
                    transform=val_augs,
                    depth=True if dataset_name.lower() == "tomatotask2d_rgbd" else False,
                    num_tasks=args.num_tasks,
                    num_views=args.num_views,
                )
        
        total_size = len(train_ds) + len(val_ds) + len(test_ds)
        if rank==0: print(f"Total Dataset Size: {total_size}")
        
        classes = sideview_classes

    return train_ds, val_ds, test_ds, classes
# TomatoTask

A flexible PyTorch training framework for plant-disease image analysis. Supports classification, regression, and semantic segmentation in single-task or heterogeneous multi-task models. 

Supports a range of off-the-shelf torchvision backbones.


## Dependencies
The  repo requires
* Python3
* CUDA

## Installation

```bash
# Clone the repository
git clone https://github.com/gmvincent/TomatoTask.git

# Navigate into the directory
cd TomatoTask

# Install dependencies
pip install -r requirements.txt
pip install .
```

## Usage

To train a multi-task ResNet-18 with TomatoTask 2D:
```bash
python main.py --model_name "resnet18" --dataset_name "tomatotask2d_rgb" --num_tasks "3" --task ["classification", "segmentation", "segmentation"] --pretrained False --gpu 5
```

To train a single task (classification) ResNet-18 model with TomatoTask 2D (using RGB-Depth representations): 
```bash
python main.py --model_name "resnet18" --dataset_name "tomatotask2d_rgbd" --num_tasks "1" --task "classification" --gpu 0
```

### Supported Datasets
* **TomatoTask** (2D and 3D variants)
* PlantVillage
* PlantDoc
* AIChallenger

### Supported Models
Loaded via torchvision with optional pretrained ImageNet weights, and an adaptable input layer (RGB, RGB-D, or other channel counts):

* ResNet: `resnet18`, `resnet34`, `resnet50`
* MobileNet: `mobilenet_v2`, `mobilenet_v3_large`, `mobilenet_v3_small`
* EfficientNet: `efficientnet_b0`–`efficientnet_b7`
* DenseNet: `densenet161`
* VGG: `vgg11`, `vgg13`, `vgg16`
* ViT: `vit_b_16`, `vit_b_32`
* Swin: `swin_t`, `swin_b`, `swin_v2_t`, `swin_v2_b`

### Distributed Training
* Single-GPU, DataParallel, and DistributedDataParallel (DDP) are all supported via config/CLI flags.
* DDP handles process-group setup and teardown, SIGTERM cleanup, and free-port selection for the master process.

## Citation

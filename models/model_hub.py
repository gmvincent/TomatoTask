import os
import json

import torch
import torchvision.models as models
from torchvision.models.segmentation.deeplabv3 import DeepLabV3
from models.model_heads import build_head, build_segmentation_model
from models.multitask_torch import MultiTask_Model

model_file = os.path.join(os.path.dirname(__file__), "torch_hub_models.json")
with open(model_file, "r") as f:
    torch_hub_models = json.load(f)

available_models = sorted(torch_hub_models.keys())

def get_model(args, model_name, single_task=True):
    """Build a model for any of `available_models`, in any of clf/reg/seg
    task, for 1 or many tasks. Driven entirely by:
      args.task           -- str, or list[str] (one per task) for multi-task
      args.num_classes    -- int, or list[int] (one per task) for multi-task
      args.num_tasks      -- e.g. 1, or an int > 1; anything else (like a
                              dataset name) is treated as single-task
      args.pretrained     -- True / False / path to a checkpoint
      args.input_channels -- 3 to leave the input layer untouched
    """
    if model_name not in torch_hub_models:
            raise ValueError(f"Model '{model_name}' not found in available models.")
  
    model = _build_backbone(args, model_name)
    model = modify_input_layer(args, model_name, model)
    
    if single_task:
        if args.task == "segmentation":
            model = build_segmentation_model(model, model_name, args.num_classes, in_channels=args.input_channels)
        else:
            model = modify_head(model, model_name, args.task, args.num_classes)
            
    else:
        model = MultiTask_Model(model_name, model, args.num_classes, args.task, in_channels=args.input_channels)
 
    get_target_layer(args, model_name, model)
 
    return model

def _build_backbone(args, model_name):
    tasks = args.task if isinstance(args.task, list) else [args.task]
    kwargs = {}
    if ("segmentation" in tasks) and (model_name.lower() in ["resnet50", "resnet101"]):
            kwargs["replace_stride_with_dilation"] = [False, True, True]

    if isinstance(args.pretrained, bool):
        weights = models.get_model_weights(model_name).DEFAULT if args.pretrained else None
        model = models.get_model(model_name, weights=weights, **kwargs)
    
    elif isinstance(args.pretrained, str) and os.path.isfile(args.pretrained):
        model = models.get_model(model_name, weights=None, **kwargs)
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    else:
        raise ValueError(f"Invalid pretrained argument: {args.pretrained}")
    
    return model

def _locate_head(model):
    if hasattr(model, "fc"):
        return model, "fc", model.fc.in_features
    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, torch.nn.Sequential):
            return classifier, len(classifier) - 1, classifier[-1].in_features
        return model, "classifier", classifier.in_features
    if hasattr(model, "head"):
        return model, "head", model.head.in_features
    if hasattr(model, "heads"):
        return model.heads, "head", model.heads.head.in_features
    
    raise ValueError("Unknown model architecture. Can't find classifier head.")

def modify_head(model, model_name, task, num_classes):
    """Classification & regression only. Segmentation goes through build_segmentation_model."""
    owner, attr, in_features = _locate_head(model)
    new_head = build_head(task, in_features, num_classes, model_name=model_name)

    if isinstance(attr, int):
        owner[attr] = new_head
    else:
        setattr(owner, attr, new_head)
    
    return model

def modify_input_layer(args, model_name, model):
    
    if args.input_channels == 3:
        return model
    
    model_name = model_name.lower()

    input_layers = {
        "resnet":    (lambda m: m.conv1,
                      lambda m, layer: setattr(m, "conv1", layer)),
        "vgg":       (lambda m: m.features[0],
                      lambda m, layer: m.features.__setitem__(0, layer)),
        "dense":     (lambda m: m.features.conv0,
                      lambda m, layer: setattr(m.features, "conv0", layer)),
        "mobile":    (lambda m: m.features[0][0],
                      lambda m, layer: m.features[0].__setitem__(0, layer)),
        "efficient": (lambda m: m.features[0][0],
                      lambda m, layer: m.features[0].__setitem__(0, layer)),
        "vit":       (lambda m: m.conv_proj,
                      lambda m, layer: setattr(m, "conv_proj", layer)),
        "swin":      (lambda m: m.features[0][0],
                      lambda m, layer: m.features[0].__setitem__(0, layer)),
    }

    for key, (getter, setter) in input_layers.items():
        if model_name.startswith(key):
            old_conv = getter(model)
            
            new_conv = torch.nn.Conv2d(
                args.input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            
            new_conv = new_conv.to(
                old_conv.weight.device,
                old_conv.weight.dtype,
            )
                        
            with torch.no_grad():
                if args.input_channels == 1:
                    new_conv.weight[:] = old_conv.weight.mean(
                        dim=1, keepdim=True
                    )
                else:
                    n_copy = min(3, args.input_channels) # if args.input_channels ==2
                    new_conv.weight[:, :n_copy] = old_conv.weight[:, :n_copy]
                    if args.input_channels > 3:
                        new_conv.weight[:, 3:] = (
                            old_conv.weight[:, :1]
                            .repeat(1, args.input_channels - 3, 1, 1)
                        )

            setter(model, new_conv)

            return model

    raise ValueError(f"Unsupported architecture: {model_name}")


def get_target_layer(args, model_name, model):
    model_name = model_name.lower()

    # Segmentation and multi-task models both hold their trunk in `.backbone`
    root = model.backbone if isinstance(model, (DeepLabV3, MultiTask_Model)) else model
 
    model_targets = {
        "fasterrcnn": lambda m: m.backbone,
        "resnet":     lambda m: m.layer4[-1],
        "vgg":        lambda m: m.features[-1],
        "dense":      lambda m: m.features[-1],
        "mobile":     lambda m: m.features[-1],
        "efficient":  lambda m: m.features[-1][0],
        "mnasnet":    lambda m: m.layers[-1],
        "vit":        lambda m: m.encoder.ln,
        "swin":       lambda m: m.features[-1][0].norm1,
    }
 
    for key, target_fn in model_targets.items():
        if model_name.startswith(key):
            args.target_layers = [target_fn(root)]
            return

    raise ValueError(f"Model '{model_name}' does not have a predefined target_layer for GradCAM visualization.")
import os
import json

import torch
import torchvision.models as models
#from torchvision.models import get_model, get_model_weights

model_file = os.path.join(os.path.dirname(__file__), "torch_hub_models.json")
torch_hub_models = json.load(open(model_file, "r"))

available_models = sorted(torch_hub_models.keys())


def get_model(args, model_name, single_task=True):
    
    if model_name not in torch_hub_models:
        raise ValueError(f"Model '{model_name}' not found in available models.")
    
    # load model and weights
    if isinstance(args.pretrained, bool) and args.pretrained:
        weights_enum = models.get_model_weights(model_name)
        weight_class = getattr(weights_enum, torch_hub_models[model_name]["weights"])
        model = models.get_model(model_name, weights=weight_class, num_classes=args.num_classes)
    
    elif isinstance(args.pretrained, bool) and not args.pretrained:
        model = models.get_model(model_name, weights=None, num_classes=args.num_classes)
    
    elif isinstance(args.pretrained, str) and os.path.isfile(args.pretrained):
        model = models.get_model(model_name, weights=None, num_classes=args.num_classes)
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    else:
        raise ValueError(f"Invalid pretrained argument: {args.pretrained}")
    
    # set up multi-task heads
    if not single_task:
        #TODO: implement multi-task modeling
        return -1
    
    get_target_layer(args, model_name, model)
    
    return model

def get_target_layer(args, model_name, model):
    
    if "fasterrcnn" in model_name:
        args.target_layers = model.backbone
    elif "resnet" in model_name:
        args.target_layers = model.layer4[-1]
    elif "vgg" in model_name:
        args.target_layers = model.features[-1]
    elif "dense" in model_name:
        args.target_layers = model.features[-1]
    elif "mobile" in model_name:
        args.target_layers = model.features[-1]
    elif "mnasnet" in model_name:
        args.target_layers = model.layers[-1]
    elif model_name.starts_with("vit"):
        args.target_layers = model.blocks[-1].norm1
    elif model_name.starts_with("swin"):
        args.target_layers = model.layers[-1].blocks[-1].norm1
    elif "efficient" in model_name:
        args.target_layers = model.features[-1][0]
    else:
        raise ValueError(f"Model '{model_name}' does not have a predefined target_layer for GradCAM visualization.")
import os
import json

import torch
import torchvision.models as models
from models.multitask_torch import MultiTask_Model

model_file = os.path.join(os.path.dirname(__file__), "torch_hub_models.json")
torch_hub_models = json.load(open(model_file, "r"))

available_models = sorted(torch_hub_models.keys())


def get_model(args, model_name, single_task=True):
    
    if single_task:
        if model_name not in torch_hub_models:
            raise ValueError(f"Model '{model_name}' not found in available models.")
        
        # load model and weights
        if isinstance(args.pretrained, bool) and args.pretrained:
            weights_enum = models.get_model_weights(model_name)
            weights = weights_enum.DEFAULT
            
            model = models.get_model(model_name, weights=weights)
            model = modify_classifier(model, args.num_classes)
        
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
    else:
        if model_name not in torch_hub_models:
            raise ValueError(f"Model '{model_name}' not found in available models.")
        
        # load model and weights
        if isinstance(args.pretrained, bool) and args.pretrained:
            weights_enum = models.get_model_weights(model_name)
            weights = weights_enum.DEFAULT
            
            model = models.get_model(model_name, weights=weights)

        
        elif isinstance(args.pretrained, bool) and not args.pretrained:
            model = models.get_model(model_name, weights=None)
        
        elif isinstance(args.pretrained, str) and os.path.isfile(args.pretrained):
            model = models.get_model(model_name, weights=None)
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        else:
            raise ValueError(f"Invalid pretrained argument: {args.pretrained}")
        model = MultiTask_Model(model_name, model, args.num_classes)
    get_target_layer(args, model_name, model, single_task)
    
    return model

def modify_classifier(model, num_classes):
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'):
        classifier = model.classifier
        if isinstance(classifier, torch.nn.Sequential):
            in_features = classifier[-1].in_features
            classifier[-1] = torch.nn.Linear(in_features, num_classes)
        else:
            in_features = classifier.in_features
            model.classifier = torch.nn.Linear(in_features, num_classes)
    elif hasattr(model, 'head'):
        in_features = model.head.in_features
        model.head = torch.nn.Linear(in_features, num_classes)
    elif hasattr(model, 'heads'):
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Unknown model architecture. Can't find classifier head.")
    return model

def get_target_layer(args, model_name, model, single_task):
    model_name = model_name.lower()

    model_targets = {
        "fasterrcnn": lambda m, st: m.backbone if st else m.encoder,
        "resnet":     lambda m, st: m.layer4[-1] if st else m.encoder[-1],
        "vgg":        lambda m, st: m.features[-1] if st else m.encoder[-1],
        "dense":      lambda m, st: m.features[-1] if st else m.encoder[-1],
        "mobile":     lambda m, st: m.features[-1] if st else m.encoder[-1],
        "mnasnet":    lambda m, st: m.layers[-1] if st else m.encoder[-1],
        "vit":        lambda m, st: m.encoder.ln if st else m.norm,
        "swin":       lambda m, st: m.features[-1][0].norm1 if st else m.features[-1][0].norm1,
        "efficient":  lambda m, st: m.features[-1][0] if st else m.encoder[-1][0],
    }

    matched = False
    for key, target_fn in model_targets.items():
        if key in model_name:
            args.target_layers = [target_fn(model, single_task)]
            matched = True
            break

    if not matched:
        raise ValueError(f"Model '{model_name}' does not have a predefined target_layer for GradCAM visualization.")
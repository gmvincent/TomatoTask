import os
import json

import torch
from torchvision.models import get_model, get_model_weights

model_file = os.path.join(os.path.dirname(__file__), "torch_hub_models.json")
torch_hub_models = json.load(open(model_file, "r"))

available_models = sorted(torch_hub_models.keys())


def get_model(args, model_name):
    
    if model_name not in torch_hub_models:
        raise ValueError(f"Model '{model_name}' not found in available models.")
    
    # load model and weights
    if isinstance(args.pretrained, bool) and args.pretrained:
        weights_enum = get_model_weights(model_name)
        weight_class = getattr(weights_enum, torch_hub_models[model_name]["weights"])
        model = get_model(model_name, weights=weight_class)
    
    elif isinstance(args.pretrained, bool) and not args.pretrained:
        model = get_model(model_name, weights=None)
    
    elif isinstance(args.pretrained, str) and os.path.isfile(args.pretrained):
        model = get_model(model_name, weights=None)
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    else:
        raise ValueError(f"Invalid pretrained argument: {args.pretrained}")
    
    # set up multi-task heads
    if args.num_tasks not in ["1", "tomato"]:
        #TODO: implement multi-task modeling
        return -1
    
    return model
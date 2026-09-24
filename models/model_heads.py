import math
 
import torch
import torch.nn.functional as F
import torchvision.models as models

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3

def build_head(task, in_features, num_classes, model_name=None):
    """Build the appropriate head module for a given mode"""
    if task == "classification":
        return torch.nn.Linear(in_features, num_classes)
 
    elif task == "regression":
        return torch.nn.Linear(in_features, 1)
 
    elif task == "segmentation":
        return DeepLabHead(in_features, num_classes)
             
    raise ValueError(f"Unsupported task '{task}'. Use one of: classification, regression, segmentation.")

def build_feature_extractor(model, model_name, aux=False):
    """Cut a classification model down to a backbone returning
    {"out": (B, C, H, W)} (+ "aux" if requested, + "cls" for ViT).
    Returns (backbone, image_size to probe with). Shared by the single-task
    segmentation builder and MultiTask_Model. Does not modify `model`."""
 
    if model_name.lower().startswith("vit"):
        if aux:
            raise ValueError("aux loss isn't supported for ViT backbones.")
        return ViTFeatures(model), model.image_size  # ViT requires exactly this input size
    
    model_layers = {
        "resnet":    {"layer4": "out", "layer3": "aux"},
        "vgg":       {"features": "out"},
        "dense":     {"features": "out"},
        "mobile":    {"features": "out"},
        "efficient": {"features": "out"},
        "mnasnet":   {"layers": "out"},
        "swin":      {"permute": "out"},  # features -> norm -> permute -> avgpool -> ...
    }
    
    for key, return_layers in model_layers.items():
        if model_name.lower().startswith(key):
            break
    else:
        raise ValueError(f"No feature return layer defined for '{model_name}'.")
    
    if aux and "aux" not in return_layers.values():
        raise ValueError(f"aux loss isn't defined for '{model_name}'.")

    if not aux:
        return_layers = {
            k: v for k, v in return_layers.items()
            if v == "out"
        }

    return IntermediateLayerGetter(model, return_layers=return_layers), 224

def build_segmentation_model(model, model_name, num_classes, in_channels=3, aux=False):

    backbone, image_size = build_feature_extractor(model, model_name, aux)
    channels = probe_channels(backbone, in_channels, image_size)
    
    classifier = DeepLabHead(channels["out"], num_classes)
    aux_classifier = FCNHead(channels["aux"], num_classes) if aux else None
    
    return DeepLabV3(backbone, classifier, aux_classifier)    

class ViTFeatures(torch.nn.Module):
    """Makes a torchvision ViT behave like an IntermediateLayerGetter:
    returns {"out": (B, hidden_dim, H/patch, W/patch)}.
 
    Every piece of the ViT it uses is registered as a submodule, so it moves
    with .to(device), shows up in .parameters() and is saved in state_dict."""
 
    def __init__(self, vit):
        super().__init__()
        self.conv_proj = vit.conv_proj
        self.class_token = vit.class_token
        self.encoder = vit.encoder  # adds pos_embedding, runs the blocks and final ln
        self.image_size = vit.image_size
 
    def forward(self, x):
        b = x.shape[0]
        x = self.conv_proj(x)                        # (B, C, gh, gw)
        gh, gw = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)             # (B, gh*gw, C)
        x = torch.cat([self.class_token.expand(b, -1, -1), x], dim=1)
        x = self.encoder(x)
        x = x[:, 1:].transpose(1, 2).reshape(b, -1, gh, gw)
        return {"out": x}

@torch.no_grad()
def probe_channels(backbone, in_channels, image_size):
    """Read feature channel counts off a dummy forward pass instead of hard-coding them."""
    was_training = backbone.training
    backbone.eval()
    p = next(backbone.parameters())
    dummy = torch.zeros(1, in_channels, image_size, image_size, device=p.device, dtype=p.dtype)
    feats = backbone(dummy)
    backbone.train(was_training)
    return {k: v.shape[1] for k, v in feats.items()}

def classifier_prefix(classifier):
    """Split a Sequential classifier. Used by MultiTask_Model."""
    if isinstance(classifier, torch.nn.Sequential):
        prefix = torch.nn.Sequential(*list(classifier.children())[:-1])
        in_features = classifier[-1].in_features
    elif isinstance(classifier, torch.nn.Linear):
        prefix = torch.nn.Identity()
        in_features = classifier.in_features
    else:
        raise ValueError(f"Unexpected classifier structure: {type(classifier)}")
    return prefix, in_features
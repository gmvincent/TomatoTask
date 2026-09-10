import math
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def build_head(task, in_features, num_classes, model_name=None):
    """Build the appropriate head module for a given mode"""
    if task == "classification":
        return torch.nn.Linear(in_features, num_classes)
 
    elif task == "regression":
        return torch.nn.Linear(in_features, num_classes)
 
    elif task == "segmentation":
        if model_name is not None and model_name.lower().startswith(("vit", "swin")):
            raise ValueError(
                f"'{model_name}' does not expose a spatial feature map at its head "
                "(ViT/Swin pool to tokens before the head), so a DeepLabHead-style "
                "segmentation head isn't valid here -- use ViTSegmentationModel or "
                "SwinSegmentationModel instead, which restructure the forward pass."
            )
        return models.segmentation.deeplabv3.DeepLabHead(in_features, num_classes)
 
    raise ValueError(f"Unsupported task '{task}'. Use one of: classification, regression, segmentation.")

def last_conv_channels(module):
    """Channel count of the last Conv2d found in `module`. Used by MultiTask_Model."""
    conv_layers = [m for m in module.modules() if isinstance(m, torch.nn.Conv2d)]
    if not conv_layers:
        raise ValueError("No Conv2d layers found to infer spatial channel count.")
    return conv_layers[-1].out_channels

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


class ViTSegmentationHead(nn.Module):
    """Reshapes a ViT's patch-token sequence back into a spatial grid and
    upsamples by `patch_size` to produce a per-pixel class map.
 
    Assumes a square patch grid and a power-of-2 patch_size -- true for
    vit_b_16 (16) and vit_b_32 (32), NOT for e.g. a 14-patch ViT variant."""
 
    def __init__(self, embed_dim, patch_size, num_classes, decoder_channels=256):
        super().__init__()
        if patch_size & (patch_size - 1) != 0:
            raise ValueError(f"ViTSegmentationHead needs a power-of-2 patch_size, got {patch_size}")
 
        n_stages = int(math.log2(patch_size))
        layers = [
            nn.Conv2d(embed_dim, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        ]
        in_ch = decoder_channels
        for _ in range(n_stages):
            out_ch = max(in_ch // 2, 16)
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        layers.append(nn.Conv2d(in_ch, num_classes, kernel_size=1))
        self.decoder = nn.Sequential(*layers)
 
    def forward(self, patch_tokens):
        # patch_tokens: (B, N, C), cls token already dropped, N a perfect square
        b, n, c = patch_tokens.shape
        h = w = int(math.sqrt(n))
        if h * w != n:
            raise ValueError(f"Expected a square patch grid, got {n} patches.")
        grid = patch_tokens.transpose(1, 2).reshape(b, c, h, w)
        return self.decoder(grid)
 
 
class ViTSegmentationModel(nn.Module):
    """Wraps a torchvision ViT backbone for segmentation. Needed because
    segmentation needs the full patch-token sequence, not just whatever
    survives the backbone's own pooling to a single cls-token vector -- so,
    unlike classification/regression, this can't be done by swapping out
    `model.heads` alone; the forward pass itself has to change."""
 
    def __init__(self, vit_model, num_classes):
        super().__init__()
        self.class_token = vit_model.class_token
        self.encoder = vit_model.encoder
        self._process_input = vit_model._process_input
        self.seg_head = ViTSegmentationHead(vit_model.hidden_dim, vit_model.patch_size, num_classes)
 
    def forward(self, x):
        tokens = self._process_input(x)
        n = tokens.shape[0]
        cls = self.class_token.expand(n, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = self.encoder(tokens)  # applies its own pos_embedding, layers, and ln
        patch_tokens = tokens[:, 1:]  # drop cls token, keep the spatial grid
        out = self.seg_head(patch_tokens)
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
 
 
class SwinSegmentationModel(nn.Module):
    """Swin already keeps a spatial (B,H,W,C) feature map before its own
    pooling -- permute to (B,C,H,W) and reuse the same DeepLabHead used for
    the CNN backbones. No token-grid reshaping needed, unlike ViT."""
 
    def __init__(self, swin_model, num_classes):
        super().__init__()
        self.features = swin_model.features
        self.norm = swin_model.norm
        self.permute = swin_model.permute
        # NOT last_conv_channels(features): Swin's downsampling (PatchMerging)
        # uses nn.Linear, not Conv2d, so scanning for the last Conv2d would
        # find the initial patch-embed stem's channel count, not the final
        # stage's. `head.in_features` is the correct final channel count.
        channels = swin_model.head.in_features
        self.seg_head = build_head("segmentation", channels, num_classes)
 
    def forward(self, x):
        feat = self.permute(self.norm(self.features(x)))
        out = self.seg_head(feat)
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
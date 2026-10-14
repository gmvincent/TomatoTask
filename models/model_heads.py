import math
from collections import OrderedDict
 
import torch
import torch.nn.functional as F

from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3

from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as smp_init

def build_head(task, in_features, num_classes, model_name=None):
    """Build the appropriate head module for a given mode"""
    if task == "classification":
        return torch.nn.Linear(in_features, num_classes)
 
    elif task == "regression":
        return torch.nn.Linear(in_features, 1)
 
    elif task == "segmentation":
        raise ValueError("Use build_segmentation_head() for segmentation heads.")
             
    raise ValueError(f"Unsupported task '{task}'. Use one of: classification, regression, segmentation.")

class UNetSegHead(torch.nn.Module):
    """smp's U-Net decoder + segmentation head.
    Takes the backbone's feature dict, returns logits at input resolution."""
 
    def __init__(self, channels, num_classes, decoder_channels=None,
                 add_center_block=False, **decoder_kwargs):
        super().__init__()
        enc_channels = [c for k, c in channels.items() if k.startswith("skip")] + [channels["out"]]
        if len(enc_channels) < 2:
            raise ValueError("smp's UnetDecoder needs a multi-scale encoder "
                             "(ViT only has one scale).")
 
        n_blocks = len(enc_channels) - 1
        if decoder_channels is None:
            decoder_channels = ([256, 128, 64, 32, 16] + [16] * n_blocks)[:n_blocks]
        decoder_channels = list(decoder_channels)
        if len(decoder_channels) != n_blocks:
            raise ValueError(f"This backbone needs {n_blocks} decoder_channels, "
                             f"got {len(decoder_channels)}.")
 
        self.decoder = UnetDecoder(
            encoder_channels=enc_channels,
            decoder_channels=decoder_channels,
            n_blocks=n_blocks,
            add_center_block=add_center_block,
            **decoder_kwargs,  # e.g. attention_type="scse", interpolation_mode="bilinear"
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1], out_channels=num_classes, kernel_size=3,
        )
 
        smp_init.initialize_decoder(self.decoder)
        smp_init.initialize_head(self.segmentation_head)
 
    def forward(self, feats):
        maps = [v for k, v in feats.items() if k.startswith("skip")] + [feats["out"]]
        return self.segmentation_head(self.decoder(maps))
 
 
class DeepLabSegHead(torch.nn.Module):
    """DeepLabHead that takes the feature dict, so both heads share one interface."""
 
    def __init__(self, channels, num_classes):
        super().__init__()
        self.head = DeepLabHead(channels["out"], num_classes)
 
    def forward(self, feats):
        return self.head(feats["out"])
 
 
def build_segmentation_head(arch, channels, num_classes, model_name, **kwargs):
    """Segmentation head for a feature dict from build_feature_extractor.
    Output is logits; U-Net's are at input size, DeepLab's need upsampling."""
    arch = arch.lower()
    if arch == "unet":
        return UNetSegHead(channels, num_classes,
                           add_center_block=model_name.lower().startswith("vgg"),  # mirrors smp.Unet
                           **kwargs)
    if arch == "deeplabv3":
        return DeepLabSegHead(channels, num_classes)
    raise ValueError(f"Unsupported arch '{arch}'. Use 'unet' or 'deeplabv3'.")

def build_feature_extractor(model, model_name, aux=False):
    """Cut a classification model down to a backbone returning
    {"skip0", ..., "out"} (+ "aux" if requested). ViT returns {"out", "cls"}.
    Returns (backbone, image_size to probe with). Shared by the single-task
    segmentation builder and MultiTask_Model. Does not modify `model`."""
    name = model_name.lower()
 
    if name.startswith("vit"):
        if aux:
            raise ValueError("aux loss isn't supported for ViT backbones.")
        return ViTFeatures(model), model.image_size  # ViT requires exactly this input size
 
    model_stages = {
        "resnet":    (lambda m: m,          "layer4", False),
        "vgg":       (lambda m: m.features, None,     False),
        "dense":     (lambda m: m.features, None,     False),
        "mobile":    (lambda m: m.features, None,     False),
        "efficient": (lambda m: m.features, None,     False),
        "mnasnet":   (lambda m: m.layers,   None,     False),
        "swin":      (lambda m: torch.nn.Sequential(OrderedDict([*m.features.named_children(),
                                                                ("norm", m.norm)])), None, True),
    }
    # Stages returned by name instead of "last layer at each resolution"
    explicit_stages = {
        "resnet": ("relu", "layer1", "layer2", "layer3", "layer4"),
    }
 
    for key, (get_container, stop_at, channels_last) in model_stages.items():
        if name.startswith(key):
            break
    else:
        raise ValueError(f"No feature return layer defined for '{model_name}'.")
 
    layers = OrderedDict()
    for child_name, child in get_container(model).named_children():
        layers[child_name] = child
        if child_name == stop_at:
            break
 
    image_size = 224
    return_idx, has_full_res = _stage_indices(layers, _input_channels(model), image_size, channels_last)
 
    explicit = next((v for k, v in explicit_stages.items() if name.startswith(k)), None)
    if explicit is not None:
        layer_names = list(layers)
        return_idx = [layer_names.index(n) for n in explicit]
 
    if aux and len(return_idx) < 2:
        raise ValueError(f"aux loss isn't defined for '{model_name}'.")
 
    backbone = MultiScaleBackbone(layers, return_idx, channels_last, aux,
                                  include_input=not has_full_res)
    return backbone, image_size

def build_segmentation_model(model, model_name, num_classes, in_channels=3, aux=False,
                             arch="unet", **head_kwargs):
    backbone, image_size = build_feature_extractor(model, model_name, aux)
    channels = probe_channels(backbone, in_channels, image_size)
    aux_classifier = FCNHead(channels["aux"], num_classes) if aux else None
 
    arch = arch.lower()
    if arch == "deeplabv3":
        # plain DeepLabHead (not the dict adapter) so DeepLabV3 checkpoints still load
        classifier = DeepLabHead(channels["out"], num_classes)
        return DeepLabV3(backbone, classifier, aux_classifier)
 
    if arch == "unet":
        classifier = build_segmentation_head(arch, channels, num_classes, model_name, **head_kwargs)
        return SMPUnetSegmentation(backbone, classifier, aux_classifier)
 
    raise ValueError(f"Unsupported arch '{arch}'. Use 'unet' or 'deeplabv3'.")

class MultiScaleBackbone(torch.nn.Module):
    """Returns the last feature map at each resolution:
    {"skip0": input resolution, "skip1": ..., "out": deepest (+ "aux": second deepest)}.
    If the network has no full-res layer, skip0 is the input itself (like smp's
    identity stage). DeepLabV3 only reads "out"/"aux" and ignores the skips."""
 
    def __init__(self, layers, return_idx, channels_last=False, aux=False, include_input=False):
        super().__init__()
        self.layers = torch.nn.Sequential(layers)
        self.return_idx = set(return_idx)
        self.channels_last = channels_last
        self.aux = aux
        self.include_input = include_input
 
    def forward(self, x):
        feats = [x] if self.include_input else []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.return_idx:
                feats.append(x.permute(0, 3, 1, 2).contiguous() if self.channels_last else x)
 
        aux = feats[-2] if self.aux else None
        feats = _fill_stride_gaps(feats)

        out = OrderedDict((f"skip{i}", f) for i, f in enumerate(feats[:-1]))
        out["out"] = feats[-1]
        if aux is not None:
            out["aux"] = aux
        return out
    
class SMPUnetSegmentation(torch.nn.Module):
    """Backbone + U-Net head. Same {"out", "aux"} output as DeepLabV3."""
 
    def __init__(self, backbone, classifier, aux_classifier=None):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
 
    def forward(self, x):
        input_size = x.shape[-2:]
        feats = self.backbone(x)
 
        result = OrderedDict()
        result["out"] = self.classifier(feats)  # already at input size
 
        if self.aux_classifier is not None:
            aux = self.aux_classifier(feats["aux"])
            result["aux"] = F.interpolate(aux, size=input_size, mode="bilinear", align_corners=False)
        return result

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
    try:
        feats = backbone(dummy)
    finally:
        backbone.train(was_training)
    return {k: v.shape[1] for k, v in feats.items()}

@torch.no_grad()
def _stage_indices(layers, in_channels, image_size, channels_last):
    """Index of the last child at each resolution, and whether the first is full-res."""
    seq = torch.nn.Sequential(layers)
    modes = {m: m.training for m in seq.modules()}
    p = next(seq.parameters())
    x = torch.zeros(1, in_channels, image_size, image_size, device=p.device, dtype=p.dtype)
 
    last_at = OrderedDict()
    seq.eval()  # don't disturb BatchNorm running stats while probing
    try:
        for i, layer in enumerate(seq):
            x = layer(x)
            hw = tuple(x.shape[1:3]) if channels_last else tuple(x.shape[-2:])
            last_at[hw] = i
    finally:
        for m, training in modes.items():
            m.training = training
 
    has_full_res = next(iter(last_at)) == (image_size, image_size)
    return list(last_at.values()), has_full_res

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

def _input_channels(model):
    """Read in_channels off the stem conv, so callers don't have to pass it."""
    conv = next((m for m in model.modules() if isinstance(m, torch.nn.Conv2d)), None)
    if conv is None:
        raise ValueError("Couldn't find a Conv2d to infer input channels from.")
    return conv.in_channels

def _fill_stride_gaps(feats):
    """Insert 0-channel placeholders where a level skips a 2x step (e.g. Swin's
    stride-4 patch embedding), like smp's transformer encoders, so the U-Net
    decoder still upsamples 2x per block."""
    filled = [feats[0]]
    for f in feats[1:]:
        prev = filled[-1]
        n_missing = round(math.log2(prev.shape[-2] / f.shape[-2])) - 1
        for _ in range(n_missing):
            h, w = (s - s // 2 for s in prev.shape[-2:])  # ceil halving
            prev = prev.new_empty(prev.shape[0], 0, h, w)
            filled.append(prev)
        filled.append(f)
    return filled
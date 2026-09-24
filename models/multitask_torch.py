import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from models.model_heads import build_head, build_feature_extractor, classifier_prefix, probe_channels

class _VectorTaskHead(nn.Module):
    """A cloned hidden-FC prefix + a fresh final Linear sized for this task.
    Kept as its own module so pretrained hidden layers are preserved per-task 
    instead of being shared or discarded."""
 
    def __init__(self, prefix, final_linear):
        super().__init__()
        self.prefix = prefix
        self.final = final_linear
 
    def forward(self, x):
        return self.final(self.prefix(x))

def _find_classifier(model):
    """The original classification head, whatever torchvision named it."""
    for attr in ("fc", "classifier", "head", "heads"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise ValueError(f"Can't find a classifier head on {type(model).__name__}.")

def _vector_pool(model, name):
    """How the original model turns its feature map into a vector."""
    if name.startswith("dense"):
        return nn.Sequential(nn.ReLU(), nn.AdaptiveAvgPool2d(1))
    if hasattr(model, "avgpool"):
        # VGG's is AdaptiveAvgPool2d(7), which its classifier prefix expects.
        return model.avgpool
    return nn.AdaptiveAvgPool2d(1)

class MultiTask_Model(nn.Module):
    """One shared backbone, one head per task. Any mix of classification,
    regression and segmentation, for any backbone build_feature_extractor supports."""
    
    def __init__(self, base_name, base_model, num_classes, tasks, in_channels=3):
        super(MultiTask_Model, self).__init__()
        
        if len(num_classes) != len(tasks):
            raise ValueError(
                f"Got {len(num_classes)} num_classes entries for {len(tasks)} tasks; "
                "pass one num_classes per task."
            )
        
        self.base_name = base_name.lower()
        self.tasks = list(tasks)
        self.num_tasks = len(num_classes)
        
        self.is_vit = self.base_name.startswith("vit")

        # take vector features from original model
        vector_prefix, vector_in_features = classifier_prefix(_find_classifier(base_model))
        self.vector_pool = None if self.is_vit else _vector_pool(base_model, self.base_name)
        
        # build feature extractor
        self.backbone, image_size = build_feature_extractor(base_model, self.base_name)
        channels = probe_channels(self.backbone, in_channels, image_size)
        
        self.task_heads = nn.ModuleList()
        for task, n_classes in zip(self.tasks, num_classes):
            if task == "segmentation":
                head = build_head(task, channels["out"], n_classes)
            else:
                final_linear = build_head(task, vector_in_features, n_classes)
                head = _VectorTaskHead(copy.deepcopy(vector_prefix), final_linear)
            self.task_heads.append(head)
    
    def _to_vector(self, feats):
        if self.is_vit:
            return feats["cls"]
        
        return torch.flatten(self.vector_pool(feats["out"]), 1)
    
    def _run_head(self, idx, feats, out_size):
        head = self.task_heads[idx]
        
        if self.tasks[idx] == "segmentation":
            logits = head(feats["out"])
            return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        
        return head(self._to_vector(feats))
    
    def forward(self, x, task_idx=None):
        feats = self.backbone(x)
        
        if task_idx is not None:
            return self._run_head(task_idx, feats, x.shape[-2:])
        
        return [self._run_head(i, feats, x.shape[-2:]) for i in range(self.num_tasks)]

#TODO: start with the shared trunk then do the cascade model

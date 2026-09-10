import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from models.model_heads import build_head, last_conv_channels, classifier_prefix

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

class MultiTask_Model(nn.Module):
    def __init__(self, base_name, base_model, num_classes, tasks):
        super(MultiTask_Model, self).__init__()
        
        if len(num_classes) != len(tasks):
            raise ValueError(
                f"Got {len(num_classes)} num_classes entries for {len(tasks)} tasks; "
                "pass one num_classes per task."
            )
        
        self.base_name = base_name
        self.num_tasks = len(num_classes)
        self.original_classifier = None

        # Setup encoder and store original classifier
        if base_name.startswith('resnet'):
            self.encoder = nn.Sequential(*list(base_model.children())[:-2])  # Remove avgpool + fc
            self.vector_pool = nn.AdaptiveAvgPool2d(1)
            vector_prefix, vector_in_features = nn.Identity(), base_model.fc.in_features
        elif base_name.startswith('mobile'):
            self.encoder = base_model.features
            self.vector_pool = nn.AdaptiveAvgPool2d(1)
            vector_prefix, vector_in_features = classifier_prefix(base_model.classifier)
        elif base_name.startswith('efficient'):
            self.encoder = base_model.features
            self.vector_pool = base_model.avgpool
            vector_prefix, vector_in_features = classifier_prefix(base_model.classifier)
        elif base_name.startswith('vgg'):
            self.encoder = base_model.features
            self.vector_pool = base_model.avgpool 
            vector_prefix, vector_in_features = classifier_prefix(base_model.classifier)
        elif base_name.startswith('dense'):
            self.encoder = base_model.features
            self.vector_pool = nn.AdaptiveAvgPool2d(1)
            vector_prefix, vector_in_features = nn.Identity(), base_model.classifier.in_features
        elif base_name.startswith('swin'):
            self.features = base_model.features
            self.norm = base_model.norm
            self.permute = base_model.permute
            self.vector_pool = nn.AdaptiveAvgPool2d(1)
            vector_prefix, vector_in_features = nn.Identity(), base_model.head.in_features
        elif self.base_name.startswith('vit'):
            self.encoder = base_model.encoder
            self.class_token = base_model.class_token
            self.norm = self.encoder.ln  # kept for get_target_layer's `m.norm` lookup
            self._process_input = base_model._process_input
            vector_prefix, vector_in_features = nn.Identity(), base_model.heads.head.in_features
        else:
            raise ValueError(f"Unsupported backbone type: {base_name}")

        
        if base_name.startswith(('resnet', 'mobile', 'efficient', 'vgg', 'dense')):
            self.spatial_channels = last_conv_channels(self.encoder)
            
        # Create one head per task
        # TODO: set up multi-task for different tasks (i.e., classification and segmentation)
        self.task_heads = nn.ModuleList()
        for task, n_classes in zip(self.tasks, num_classes):
            if task == "segmentation":
                head = build_head(task, self.spatial_channels, n_classes, model_name=base_name)
            else:
                final_linear = build_head(task, vector_in_features, n_classes, model_name=base_name)
                head = _VectorTaskHead(copy.deepcopy(vector_prefix), final_linear)
            self.task_heads.append(head)
        
    def forward(self, x, task_idx):
        task = self.tasks[task_idx]
        head = self.task_heads[task_idx]
        
        if self.base_name.startswith(('resnet', 'mobile', 'efficient', 'vgg', 'dense')):
            if self.base_name.startswith('dense'):
                feat_map = F.relu(self.encoder(x), inplace=True)
            else:
                feat_map = self.encoder(x)
 
            if task == "segmentation":
                out = head(feat_map)
                # DeepLabHead's output resolution follows the encoder's
                # downsampling; resize back to the input resolution.
                return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
 
            pooled = torch.flatten(self.vector_pool(feat_map), 1)
            return head(pooled)
        elif self.base_name.startswith('swin'):
            feat = self.features(x)
            feat = self.permute(self.norm(feat))
            pooled = torch.flatten(self.vector_pool(feat), 1)
            return head(pooled)
        elif self.base_name.startswith('vit'):
            feat = self._process_input(x)
            n = feat.shape[0]
            batch_class_token = self.class_token.expand(n, -1, -1)
            feat = torch.cat((batch_class_token, feat), dim=1)
            feat = self.encoder(feat)  # applies its own pos_embedding, layers, and ln
            pooled = feat[:, 0]
            return head(pooled)
 
        raise ValueError(f"Unsupported backbone type: {self.base_name}")


#TODO: start with the shared trunk then do the cascade model

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

class MultiTask_Model(nn.Module):
    def __init__(self, base_name, base_model, num_classes):
        super(MultiTask_Model, self).__init__()

        self.base_name = base_name
        self.num_tasks = len(num_classes)
        self.original_classifier = None

        # Setup encoder and store original classifier
        if base_name.startswith('resnet'):
            self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # Remove final FC
            self.feature_dim = base_model.fc.in_features
            self.original_classifier = base_model.fc
        elif base_name.startswith('efficientnet'):
            self.encoder = base_model.features
            self.pooling = base_model.avgpool
            self.feature_dim = base_model.classifier[1].in_features
            self.original_classifier = copy.deepcopy(base_model.classifier)
        elif base_name.startswith('vgg'):
            self.encoder = base_model.features
            self.pooling = base_model.avgpool
            self.feature_dim = base_model.classifier[3].in_features
            self.original_classifier = copy.deepcopy(base_model.classifier)
        elif base_name.startswith('dense'):
            self.encoder = base_model.features
            self.feature_dim = base_model.classifier.in_features
            self.original_classifier = copy.deepcopy(base_model.classifier)
        elif base_name.startswith('swin'):
            self.features = base_model.features
            self.feature_dim = base_model.head.in_features
            self.norm = base_model.norm
            self.permute = base_model.permute
            self.original_classifier = copy.deepcopy(base_model.head)
        elif self.base_name.startswith('vit'):
            self.model = base_model
            self.conv = base_model.conv_proj
            self.blocks = base_model.encoder.layers
            self.norm = base_model.encoder.ln
            self.feature_dim = base_model.heads.head.in_features
            self.original_classifier = copy.deepcopy(base_model.heads)

            self.patch_size = getattr(base_model, "patch_size", 16)
            seq_length = 1 + (224 // self.patch_size) ** 2
            hidden_dim = self.norm.normalized_shape[0]
            self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        
            self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            raise ValueError(f"Unsupported backbone type: {base_name}")

        # Create one head per task, cloned from original classifier
        self.task_heads = nn.ModuleList()
        for n_classes in num_classes:
            head = copy.deepcopy(self.original_classifier)
            if isinstance(head, nn.Sequential):
                if isinstance(head[-1], nn.Linear):
                    head[-1] = nn.Linear(self.feature_dim, n_classes)
            elif isinstance(head, nn.Linear):
                head = nn.Linear(self.feature_dim, n_classes)
            else:
                raise ValueError(f"Unexpected classifier structure: {type(head)}")
            self.task_heads.append(head)
        
    def forward(self, x, task_idx):
        if self.base_name.startswith('resnet'):
            x = self.encoder(x)
            x = torch.flatten(x, 1)
        elif self.base_name.startswith('efficientnet') or self.base_name.startswith('vgg'):
            x = self.encoder(x)
            x = self.pooling(x)
            x = torch.flatten(x, 1)
        elif self.base_name.startswith('dense'):
            x = self.encoder(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        elif self.base_name.startswith('swin'):
            x = self.features(x)
            x = self.permute(self.norm(x))
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        elif self.base_name.startswith('vit'):
            x = self.conv(x)
            x = x.flatten(2).transpose(1, 2)  # (B, N, D)
            
            cls_token = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
            x = torch.cat((cls_token, x), dim=1) 
            x = x + self.pos_embedding[:, :x.size(1), :]

            for block in self.blocks:
                x = block(x)

            x = self.norm(x)
            x = x[:, 0] 
        else:
            raise ValueError(f"Unsupported backbone type: {self.base_name}")

        return self.task_heads[task_idx](x)

#TODO: start with the shared trunk then do the cascade model

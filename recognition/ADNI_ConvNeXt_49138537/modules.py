import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNeXtAD(nn.Module):
    """ConvNeXt-Tiny backbone for AD vs CN classification"""
    def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_stages: bool = True):
        super().__init__()
        if pretrained:
            self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            self.backbone = convnext_tiny(weights=None)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

        if freeze_stages:
            for name, p in self.backbone.features.named_parameters():
                if name.split('.')[0] in {'0', '1'}:  # stem + stage1
                    p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


def make_criterion(name='ce', class_weights=None):
    if name.lower() == 'ce':
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        return nn.CrossEntropyLoss(weight=w)
    elif name.lower() == 'focal':
        class FocalLoss(nn.Module):
            def __init__(self, weight=None, gamma=2.0):
                super().__init__()
                self.ce = nn.CrossEntropyLoss(weight=weight)
                self.gamma = gamma
            def forward(self, logits, targets):
                ce = self.ce(logits, targets)
                pt = torch.exp(-ce)
                return ((1 - pt) ** self.gamma * ce).mean()
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        return FocalLoss(weight=w)
    else:
        raise ValueError(f'Unknown loss: {name}')

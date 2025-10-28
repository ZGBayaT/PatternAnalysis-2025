import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNeXtAD(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_stages: bool = True):
    def forward(self, x):
        return self.backbone(x)

def make_criterion(name='ce', class_weights=None):

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from pathlib import Path
import torch

ROOT = Path("/Users/zadehbayat/Documents/Comp3710/demo3_git/ADNI")
TRAIN_DIR = ROOT / "AD_NC" / "train"
TEST_DIR  = ROOT / "AD_NC" / "test"

class AddGaussianNoise:
    def __init__(self, std=0.01):
        self.std = std
    def __call__(self, x):
        # x: Tensor in [0,1]
        return (x + self.std * torch.randn_like(x)).clamp(0.0, 1.0)
    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std})"

IMAGENET_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def train_transform():
    #make small change to training set increase ability of normalization
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomAffine(
            degrees=7,
            translate=(0.04, 0.04),
            scale=(0.97, 1.03),
            interpolation=InterpolationMode.BICUBIC,
            fill=0
        ),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        #add gaussian noise decrsase overfit
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.10),
        transforms.RandomApply([AddGaussianNoise(std=0.01)], p=0.20),
        IMAGENET_NORM,
    ])

def test_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        IMAGENET_NORM,
    ])

def _pin_memory_for_device():
    return torch.cuda.is_available()  # CUDA: True; MPS/CPU: False

def get_train(batch_size=32, workers=4):
    ds = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform())
    g = torch.Generator()
    g.manual_seed(42)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=_pin_memory_for_device(),
        persistent_workers=(workers > 0),
        generator=g,
    )

def get_test(batch_size=64, workers=4):
    ds = datasets.ImageFolder(root=TEST_DIR, transform=test_transform())
    g = torch.Generator()
    g.manual_seed(42)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=_pin_memory_for_device(),
        persistent_workers=(workers > 0),
        generator=g,
    )

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image

# 数据目录
ROOT = Path("/Users/zadehbayat/Documents/Comp3710/demo3_git/ADNI")
TRAIN_DIR = ROOT / "AD_NC" / "train"
TEST_DIR  = ROOT / "AD_NC" / "test"

# 标准化
IMAGENET_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# 定义 transform（图像预处理流程）
def data_processing():
    data_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        IMAGENET_NORM,
    ])
    return data_tf


# 创建 DataLoader
def get_train():
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=data_processing())
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    return train_loader

def get_test():
    test_dataset  = datasets.ImageFolder(root=TEST_DIR,  transform=data_processing())
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    return test_loader


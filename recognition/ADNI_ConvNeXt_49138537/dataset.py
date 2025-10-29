from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path("/Users/zadehbayat/Documents/Comp3710/demo3_git/ADNI")
TRAIN_DIR = ROOT / "AD_NC" / "train"
TEST_DIR  = ROOT / "AD_NC" / "test"

IMAGENET_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # ← 确保 3 通道
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    IMAGENET_NORM,
])

test_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    IMAGENET_NORM,
])

def main():
    # ====== 数据集 ======
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=test_tf)

    # 类别映射（按字母序）：期望 {'AD': 0, 'NC': 1}
    print("class_to_idx:", train_ds.class_to_idx)

    # ====== DataLoader（多进程 OK）======
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ====== 简单检查 ======
    x, y = next(iter(train_loader))
    print("train batch images:", x.shape)  # [B, 3, 224, 224]
    print("train batch labels (first 8):", y[:8])

    # 统计类内样本量（可选）
    print("train counts:", Counter(train_ds.targets))
    print("test  counts:", Counter(test_ds.targets))


if __name__ == "__main__":
    main()

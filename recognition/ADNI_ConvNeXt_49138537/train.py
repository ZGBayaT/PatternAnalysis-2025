
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py — Generic image classification training script.

- Imports dataset helpers from dataset.py if available; otherwise falls back to torchvision ImageFolder.
- Imports model builders from modules.py if available; otherwise falls back to torchvision.models.resnet18.
- Supports CUDA AMP; on MPS/CPU AMP is disabled automatically.
- Saves: last.pt, best.pt, history.csv, metrics.png under --outdir.
- Prints a concise summary each epoch.

Example:
    python train.py \
        --train_dir /path/to/train \
        --test_dir  /path/to/test \
        --variant tiny \
        --batch 32 --epochs 20 --lr 3e-4 --workers 4
"""
import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

# Try to import user-provided modules
HAS_DATASET = False
HAS_MODULES = False
try:
    import dataset as user_dataset
    HAS_DATASET = True
except Exception as e:
    user_dataset = None

try:
    import modules as user_modules
    HAS_MODULES = True
except Exception as e:
    user_modules = None

# Fallbacks
from torchvision import datasets, transforms
from torchvision.models import resnet18

# ---------------------------
# Utilities
# ---------------------------
def seed_everything(seed: int = 42):
    import random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS autocast is not universally stable; we disable amp for MPS below.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def count_params(model: nn.Module) -> float:
    p = sum(p.numel() for p in model.parameters())
    return p / 1e6

# ---------------------------
# Dataloaders
# ---------------------------
def _default_transforms(img_size: int = 224, train_aug: bool = True):
    t = []
    # If user_dataset defines IMAGENET_NORM use that
    if HAS_DATASET and hasattr(user_dataset, "IMAGENET_NORM"):
        norm = user_dataset.IMAGENET_NORM
    else:
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    if train_aug:
        t += [
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ]
    else:
        t += [
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            norm,
        ]
    # Optional grayscale-to-3ch if requested by user_dataset
    if HAS_DATASET and hasattr(user_dataset, "ensure_three_channels"):
        # A hook some users like to provide
        return user_dataset.ensure_three_channels(transforms.Compose(t))
    else:
        # Many medical sets are single-channel; make them 3-ch if upstream provided a helper.
        # Otherwise, do nothing (ImageFolder handles 3ch JPGs fine).
        return transforms.Compose(t)

def build_loaders_from_user(args) -> Optional[Tuple[DataLoader, DataLoader, list]]:
    """
    Try various common entry points the user might have implemented in dataset.py.
    Return (train_loader, test_loader, class_names) or None if not available.
    """
    if not HAS_DATASET:
        return None
    # 1) Common "get_train"/"get_test"
    try:
        if hasattr(user_dataset, "train_transform") and hasattr(user_dataset, "test_transform"):
            train_tf = user_dataset.train_transform()
            test_tf  = user_dataset.test_transform()
        else:
            train_tf = _default_transforms(args.img_size, True)
            test_tf  = _default_transforms(args.img_size, False)

        # If dataset.py defines TRAIN_DIR/TEST_DIR, use them unless CLI overrides
        train_dir = Path(getattr(user_dataset, "TRAIN_DIR", args.train_dir or "")) if args.train_dir is None else Path(args.train_dir)
        test_dir  = Path(getattr(user_dataset, "TEST_DIR",  args.test_dir  or "")) if args.test_dir  is None else Path(args.test_dir)

        if train_dir and train_dir.exists():
            train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_tf)
            num_classes = len(train_ds.classes)
            if test_dir and test_dir.exists():
                test_ds = datasets.ImageFolder(root=str(test_dir), transform=test_tf)
            else:
                test_ds = None
            train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                      num_workers=args.workers, pin_memory=True,
                                      persistent_workers=(args.workers > 0))
            test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                                      num_workers=args.workers, pin_memory=True,
                                      persistent_workers=(args.workers > 0)) if test_ds is not None else None
            return train_loader, test_loader, train_ds.classes
    except Exception:
        pass

    # 2) A single builder like build_dataloaders(...) that returns loaders & classes
    for fn_name in ["build_dataloaders", "get_loaders", "get_dataloaders"]:
        if hasattr(user_dataset, fn_name):
            try:
                out = getattr(user_dataset, fn_name)(batch_size=args.batch, workers=args.workers, img_size=args.img_size)
                # Expect tuple (train_loader, test_loader, classes) or dict
                if isinstance(out, tuple) and len(out) >= 2:
                    if len(out) == 2:
                        train_loader, test_loader = out
                        classes = getattr(train_loader.dataset, "classes", list(range(getattr(args, "num_classes", 2))))
                    else:
                        train_loader, test_loader, classes = out[:3]
                    return train_loader, test_loader, classes
                if isinstance(out, dict) and "train" in out:
                    train_loader = out["train"]
                    test_loader = out.get("test", None)
                    classes = out.get("classes", getattr(train_loader.dataset, "classes", None))
                    return train_loader, test_loader, classes
            except Exception:
                pass

    return None

def build_loaders_fallback(args) -> Tuple[DataLoader, Optional[DataLoader], list]:
    """Fallback to torchvision ImageFolder using CLI dirs."""
    if args.train_dir is None or not Path(args.train_dir).exists():
        raise SystemExit("Error: --train_dir must be provided (or dataset.py must define TRAIN_DIR).")
    train_tf = _default_transforms(args.img_size, True)
    test_tf  = _default_transforms(args.img_size, False)
    train_ds = datasets.ImageFolder(args.train_dir, transform=train_tf)
    classes = train_ds.classes
    test_loader = None
    if args.test_dir is not None and Path(args.test_dir).exists():
        test_ds = datasets.ImageFolder(args.test_dir, transform=test_tf)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=args.workers, pin_memory=True,
                                 persistent_workers=(args.workers > 0))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=(args.workers > 0))
    return train_loader, test_loader, classes

# ---------------------------
# Model
# ---------------------------
def build_model(num_classes: int, variant: str = "tiny") -> nn.Module:
    # Try user-provided builders first
    if HAS_MODULES:
        # Preferred: build_convnext(variant=..., num_classes=...)
        if hasattr(user_modules, "build_convnext"):
            return user_modules.build_convnext(variant=variant, num_classes=num_classes)
        # Or ConvNeXt class with default ctor signature
        if hasattr(user_modules, "ConvNeXt"):
            # Try common configs dict
            if hasattr(user_modules, "_VARIANTS") and variant in getattr(user_modules, "_VARIANTS"):
                depths, dims = user_modules._VARIANTS[variant]
                return user_modules.ConvNeXt(depths=depths, dims=dims, num_classes=num_classes)
            else:
                # Fallback instantiation; may fail if constructor signature differs
                try:
                    return user_modules.ConvNeXt([3,3,9,3], [96,192,384,768], num_classes=num_classes)
                except Exception:
                    pass
    # Fallback to torchvision resnet18
    from torchvision.models import resnet18
    m = resnet18(weights=None)
    # Change head
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m

def build_criterion() -> nn.Module:
    if HAS_MODULES and hasattr(user_modules, "create_criterion"):
        try:
            return user_modules.create_criterion()
        except Exception:
            pass
    return nn.CrossEntropyLoss()

# ---------------------------
# Training / Eval
# ---------------------------
@dataclass
class HistoryRow:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: Optional[float]
    val_acc: Optional[float]
    lr: float

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def run_one_epoch(model, loader, criterion, optimizer, device, amp=False):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_samples = 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp) if amp else None

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.detach().item() * bs
        total_acc  += accuracy_from_logits(logits.detach(), targets) * bs
        n_samples  += bs

    return total_loss / n_samples, total_acc / n_samples

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_samples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_from_logits(logits, targets) * bs
        n_samples  += bs
    return total_loss / n_samples, total_acc / n_samples

def save_checkpoint(state: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--train_dir", type=str, default=None, help="Path to training folder (ImageFolder).")
    parser.add_argument("--test_dir",  type=str, default=None, help="Path to test/val folder (ImageFolder).")
    parser.add_argument("--img_size",  type=int, default=224)
    parser.add_argument("--batch",     type=int, default=32)
    parser.add_argument("--workers",   type=int, default=4)
    # Model / Optim
    parser.add_argument("--variant",   type=str, default="tiny", help="Model variant if using ConvNeXt in modules.py")
    parser.add_argument("--epochs",    type=int, default=20)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--wd",        type=float, default=1e-4)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--outdir",    type=str, default="runs/exp")
    parser.add_argument("--resume",    type=str, default=None, help="Path to a checkpoint to resume from.")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    use_amp = (device.type == "cuda")  # AMP only on CUDA

    # Data
    loaders = build_loaders_from_user(args)
    if loaders is None:
        train_loader, test_loader, classes = build_loaders_fallback(args)
    else:
        train_loader, test_loader, classes = loaders
    num_classes = len(classes) if classes is not None else 2

    # Model
    model = build_model(num_classes=num_classes, variant=args.variant).to(device)
    # Criterion / Optim
    criterion = build_criterion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Resume
    start_epoch = 0
    best_acc = -1.0
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    history_csv = outdir / "history.csv"

    if args.resume is not None and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", -1.0)
        print(f"=> Resumed from {args.resume} (epoch {start_epoch})")

    print(f"=> Device: {device.type}")
    print(f"=> Classes: {classes} (num_classes={num_classes})")
    print(f"=> Model params: {count_params(model):.2f} M")

    # History CSV header
    if start_epoch == 0 or not history_csv.exists():
        with open(history_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # Train
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, amp=use_amp)

        # Val (if we have test_loader; else reuse train stats for logging)
        if test_loader is not None:
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        else:
            val_loss, val_acc = None, None

        # Log
        lr_cur = optimizer.param_groups[0]["lr"]
        with open(history_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr_cur])

        dt = time.time() - t0
        msg = f"Epoch {epoch+1}/{args.epochs} | " \
              f"train_loss={train_loss:.4f} acc={train_acc:.3f}"
        if val_acc is not None:
            msg += f" | val_loss={val_loss:.4f} acc={val_acc:.3f}"
        msg += f" | lr={lr_cur:.2e} | {dt:.1f}s"
        print(msg)

        # Save last & best
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "classes": classes,
            "variant": args.variant,
        }
        save_checkpoint(state, outdir / "last.pt")
        if val_acc is not None and val_acc > best_acc:
            best_acc = val_acc
            state["best_acc"] = best_acc
            save_checkpoint(state, outdir / "best.pt")

    # Final test (if available)
    if test_loader is not None:
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"=> Final: val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    # Plot metrics
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv(history_csv)
        # Loss plot
        plt.figure()
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")
        if "val_loss" in df and df["val_loss"].notna().any():
            plt.plot(df["epoch"], df["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "loss.png", dpi=150)
        plt.close()

        # Accuracy plot
        plt.figure()
        plt.plot(df["epoch"], df["train_acc"], label="train_acc")
        if "val_acc" in df and df["val_acc"].notna().any():
            plt.plot(df["epoch"], df["val_acc"], label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "acc.png", dpi=150)
        plt.close()

        # Combined (for quick glance)
        plt.figure()
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")
        if "val_loss" in df and df["val_loss"].notna().any():
            plt.plot(df["epoch"], df["val_loss"], label="val_loss")
        plt.plot(df["epoch"], df["train_acc"], label="train_acc")
        if "val_acc" in df and df["val_acc"].notna().any():
            plt.plot(df["epoch"], df["val_acc"], label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "metrics.png", dpi=150)
        plt.close()

        print(f"=> Plots saved to {outdir}")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()

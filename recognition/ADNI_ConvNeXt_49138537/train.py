#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py — Image classification training script using user-provided dataset.py and modules.py.

- Uses dataset helpers from dataset.py (expects get_train(...) and get_test(...)).
- Uses model builders from modules.py (expects build_convnext(..., variant="tiny") or ConvNeXt tiny).
- Supports CUDA AMP; on MPS/CPU AMP is disabled automatically.
- Saves: last.pt, best.pt, history.csv, metrics.png under --outdir.
"""

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

# Import user modules
import dataset as user_dataset
import modules as user_modules

# ----------------------------
# Utils
# ----------------------------
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
#    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#       return torch.device("mps")
    return torch.device("cpu")

def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

@dataclass
class TrainState:
    epoch: int = 0
    best_acc: float = 0.0
    best_path: Optional[Path] = None

# ----------------------------
# Core train/eval loops
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    n = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == targets).sum().item()
        n += images.size(0)

    return running_loss / max(1, n), running_correct / max(1, n)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    n = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == targets).sum().item()
        n += images.size(0)

    return running_loss / max(1, n), running_correct / max(1, n)

# ----------------------------
# Build model via modules.py
# ----------------------------
def build_model(num_classes: int, variant: str = "tiny") -> nn.Module:
    """
    Tries common entrypoints in modules.py in order:
    1) user_modules.build_convnext(variant=..., num_classes=...)
    2) user_modules.ConvNeXt(...)
    3) user_modules.convnext_tiny(...)
    """
    # Option 1: build_convnext
    if hasattr(user_modules, "build_convnext"):
        try:
            return user_modules.build_convnext(variant=variant, num_classes=num_classes)
        except TypeError:
            # some versions don't accept keyword names
            return user_modules.build_convnext(variant, num_classes)
        except Exception:
            pass

    # Option 2: direct class + variants table
    if hasattr(user_modules, "ConvNeXt"):
        # guess variant dims/depths if provided
        if hasattr(user_modules, "_VARIANTS") and variant in getattr(user_modules, "_VARIANTS"):
            depths, dims = user_modules._VARIANTS[variant]
            return user_modules.ConvNeXt(depths=depths, dims=dims, num_classes=num_classes)

    # Option 3: a tiny helper function
    for name in ("convnext_tiny", "tiny", "build_tiny"):
        if hasattr(user_modules, name):
            return getattr(user_modules, name)(num_classes=num_classes)

    raise RuntimeError("Could not build model from modules.py. Expected build_convnext(...) or convnext_tiny(...).")

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ConvNeXt (tiny) on AD/NC dataset using dataset.py & modules.py")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--variant", type=str, default="tiny", choices=["tiny", "small", "base", "large", "tiny_in"])
    parser.add_argument("--outdir", type=str, default="runs/exp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"=> Device: {device.type}")

    # ----------------------------
    # Dataloaders from dataset.py
    # ----------------------------
    # Expect dataset.py to already split AD/NC into train/test
    if hasattr(user_dataset, "get_classes"):
        classes = user_dataset.get_classes()
        num_classes = len(classes)
    else:
        # Fallback: assume binary AD/NC
        classes = ["AD", "NC"]
        num_classes = 2
    print(f"=> Classes: {classes} (num_classes={num_classes})")

    if hasattr(user_dataset, "get_train"):
        train_loader = user_dataset.get_train(batch_size=args.batch, workers=args.workers)
    else:
        raise RuntimeError("dataset.py must provide get_train(batch_size, workers).")

    if hasattr(user_dataset, "get_test"):
        val_loader = user_dataset.get_test(batch_size=args.batch, workers=args.workers)
    else:
        raise RuntimeError("dataset.py must provide get_test(batch_size, workers).")

    # ----------------------------
    # Model / Optim / Loss
    # ----------------------------
    model = build_model(num_classes=num_classes, variant=args.variant)
    model.to(device)
    print(f"=> Model params: {count_params(model):.2f} M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cosine schedule (no warmup for simplicity)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP scaler only for CUDA
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    state = TrainState()
    history_path = outdir / "history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"])

    # ----------------------------
    # Train Loop
    # ----------------------------
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} | lr={lr_now:.2e} | {elapsed:.1f}s")

        # append history
        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{lr_now:.6e}", f"{elapsed:.2f}"])

        # Save last
        torch.save({"epoch": epoch, "model": model.state_dict()}, outdir / "last.pt")

        # Save best
        if val_acc > state.best_acc:
            state.best_acc = val_acc
            state.best_path = outdir / "best.pt"
            torch.save({"epoch": epoch, "model": model.state_dict()}, state.best_path)

    # ----------------------------
    # Final test on test set (reuse val_loader as test_loader provided by dataset.get_test)
    # ----------------------------
    test_loss, test_acc = evaluate(model, val_loader, criterion, device)
    print(f"=> Test: loss={test_loss:.4f} acc={test_acc:.3f}")

    # ----------------------------
    # Plot metrics
    # ----------------------------
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        hist = pd.read_csv(history_path)
        # Loss plot
        plt.figure()
        plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
        plt.plot(hist["epoch"], hist["val_loss"], label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "metrics_loss.png", dpi=150)
        plt.close()

        # Acc plot
        plt.figure()
        plt.plot(hist["epoch"], hist["train_acc"], label="train_acc")
        plt.plot(hist["epoch"], hist["val_acc"], label="val_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "metrics_acc.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")



if __name__ == "__main__":
    main()

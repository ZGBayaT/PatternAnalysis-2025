import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Local modules
import dataset as ds
import modules as mdl


def parse_args():
    p = argparse.ArgumentParser(description="Train ConvNeXt on custom dataset")
    # Data / loader
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    # Optim
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9, help="(only used if --optim sgd)")
    p.add_argument("--optim", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    # Model
    p.add_argument("--model", choices=["tiny", "small"], default="tiny")
    # Training
    p.add_argument("--amp", action="store_true", help="enable mixed precision")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    # Scheduler
    p.add_argument("--sched", choices=["cosine", "step", "none"], default="cosine")
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--step-size", type=int, default=10, help="for StepLR")
    p.add_argument("--gamma", type=float, default=0.1, help="for StepLR")
    # Checkpoints / logging
    p.add_argument("--out", type=str, default="/Users/zadehbayat/Documents/Comp3710/demo3_git/outputs/exp1")
    p.add_argument("--resume", type=str, default="", help="path to checkpoint(.pt) to resume from")
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        if criterion is not None:
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)
    acc = correct / max(1, total)
    avg_loss = (loss_sum / max(1, total)) if criterion is not None else float("nan")
    return {"acc": acc, "loss": avg_loss}


def build_model(num_classes, which="tiny"):
    if which == "tiny" and hasattr(mdl, "convnext_tiny"):
        return mdl.convnext_tiny(num_classes)
    if which == "small" and hasattr(mdl, "convnext_small"):
        return mdl.convnext_small(num_classes)
    # Fallback to ConvNeXt if helpers are absent
    if hasattr(mdl, "ConvNeXt"):
        return mdl.ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes)
    raise RuntimeError("Could not construct model from modules.py")


def build_optimizer(name, params, lr, weight_decay, momentum):
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    raise ValueError(name)


def build_scheduler(name, optimizer, epochs, warmup_epochs, steps_per_epoch, step_size, gamma):
    if name == "none":
        return None, lambda e, i: 1.0

    if name == "step":
        sched = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        # no warmup
        return sched, lambda e, i: optimizer.param_groups[0]["lr"]

    # cosine with linear warmup
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(0, warmup_epochs) * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps and warmup_steps > 0:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return sched, lambda e, i: optimizer.param_groups[0]["lr"]


def save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    # Dataloaders
    train_loader = ds.get_train(batch_size=args.batch_size, workers=args.workers)
    test_loader = ds.get_test(batch_size=args.batch_size, workers=args.workers)

    # Infer classes
    if hasattr(train_loader.dataset, "classes"):
        num_classes = len(train_loader.dataset.classes)
        class_names = list(train_loader.dataset.classes)
    else:
        # fallback – try to probe from batch
        x, y = next(iter(train_loader))
        num_classes = int(y.max().item()) + 1
        class_names = [str(i) for i in range(num_classes)]

    model = build_model(num_classes, which=args.model).to(device)

    # Loss
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer & scheduler
    optimizer = build_optimizer(args.optim, model.parameters(), args.lr, args.weight_decay, args.momentum)
    steps_per_epoch = max(1, len(train_loader))
    scheduler, _ = build_scheduler(
        args.sched, optimizer, args.epochs, args.warmup_epochs, steps_per_epoch, args.step_size, args.gamma
    )

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    best_acc = 0.0

    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt))
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and args.amp and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("best_acc", 0.0)

    if args.eval_only:
        metrics = evaluate(model, test_loader, device, criterion)
        print(f"[Eval] acc={metrics['acc']:.4f} loss={metrics['loss']:.4f}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        time_start = time.time()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None and args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                # step per-epoch later
                pass
            elif scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            epoch_correct += preds.eq(labels).sum().item()
            epoch_total += images.size(0)

        # epoch end
        if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            scheduler.step()

        train_loss = epoch_loss / max(1, epoch_total)
        train_acc = epoch_correct / max(1, epoch_total)

        # evaluate
        metrics = evaluate(model, test_loader, device, criterion)
        test_acc, test_loss = metrics["acc"], metrics["loss"]
        dt = time.time() - time_start

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={test_loss:.4f} val_acc={test_acc:.4f} | "
            f"time={dt:.1f}s"
        )

        # save "last"
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if args.amp else None,
                "best_acc": best_acc,
                "class_names": class_names,
                "args": vars(args),
            },
            out_dir / "last.pt",
        )

        # save "best"
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "best_acc": best_acc,
                    "class_names": class_names,
                    "args": vars(args),
                },
                out_dir / "best.pt",
            )

    # final eval on best (if exists)
    best_path = out_dir / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        final = evaluate(model.to(device), test_loader, device, criterion)
        print(f"[Best] acc={final['acc']:.4f} loss={final['loss']:.4f}")


if __name__ == "__main__":
    main()

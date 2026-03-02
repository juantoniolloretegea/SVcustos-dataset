#!/usr/bin/env python3
"""
SVcustos — Entrenamiento de ResNet para clasificación de polígonos polares
==========================================================================
Entrena ResNet34 (transfer learning desde ImageNet) para clasificar
vectores ternarios representados como polígonos polares.

Uso:
  python train_resnet.py --level n16
  python train_resnet.py --level n16 --model resnet50 --epochs 30

Requiere: torch, torchvision, pyyaml
Requiere GPU para rendimiento óptimo (funciona en CPU pero lento).

DOI:   https://doi.org/10.21428/39829d0b.1129de25
Autor: Juan Antonio Lloret Egea
Licencia: CC BY-NC-ND 4.0
"""

import argparse
import os
import sys
import time
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from torchvision import datasets, models, transforms
except ImportError:
    print("Error: PyTorch no instalado.")
    print("Instala con: pip install torch torchvision")
    print("Para GPU CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)


# ══════════════════════════════════════════════════════════
# DATA TRANSFORMS
# ══════════════════════════════════════════════════════════
def get_transforms():
    """
    Data transforms for training and validation.
    Training includes augmentation (rotation, flip) since polar
    polygons are rotationally meaningful — a rotated intrusion
    pattern is still an intrusion pattern.
    """
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(224),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


# ══════════════════════════════════════════════════════════
# MODEL SETUP
# ══════════════════════════════════════════════════════════
def create_model(model_name, num_classes, pretrained=True):
    """Create a ResNet model with transfer learning."""
    if model_name == "resnet34":
        model = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
    elif model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    elif model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    # Replace final FC layer for our 3 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


# ══════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════
def train_model(model, dataloaders, dataset_sizes, criterion,
                optimizer, scheduler, device, num_epochs=25):
    """Standard PyTorch training loop with validation."""
    start = time.time()
    best_acc = 0.0
    best_weights = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"  Epoch {epoch+1}/{num_epochs}")
        print(f"  {'─' * 40}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            print(f"    {phase:5s} → Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = model.state_dict().copy()

        print()

    elapsed = time.time() - start
    print(f"  Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"  Best val accuracy: {best_acc:.4f}")

    model.load_state_dict(best_weights)
    return model, history


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="SVcustos — Train ResNet on polar polygon images"
    )
    parser.add_argument("--level", required=True, choices=["n16", "n25", "n36"])
    parser.add_argument("--model", default=None, help="Model override (resnet18/34/50)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config_dir) / f"{args.level}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n = cfg["level"]["n"]
    model_name = args.model or cfg["training"]["model"]
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = cfg["training"]["learning_rate"]
    step_size = cfg["training"]["lr_step_size"]
    gamma = cfg["training"]["lr_gamma"]
    num_workers = cfg["training"].get("num_workers", 4)

    data_dir = Path("data") / f"n{n}"
    if not data_dir.exists():
        print(f"Error: Dataset not found at {data_dir}")
        print(f"Run first: python generate_dataset.py --level {args.level}")
        sys.exit(1)

    # Device
    if args.no_gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"═══ SVcustos ResNet Training ═══")
    print(f"  Level: n={n}")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print()

    # Data
    data_transforms = get_transforms()
    image_datasets = {
        x: datasets.ImageFolder(str(data_dir / x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size,
            shuffle=(x == "train"), num_workers=num_workers
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    print(f"  Classes: {class_names}")
    print(f"  Train: {dataset_sizes['train']} images")
    print(f"  Val: {dataset_sizes['val']} images")
    print()

    # Model
    num_classes = len(class_names)
    model = create_model(model_name, num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Train
    model, history = train_model(
        model, dataloaders, dataset_sizes, criterion,
        optimizer, scheduler, device, num_epochs=epochs
    )

    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"svcustos_n{n}_{model_name}_{timestamp}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "config": cfg,
        "history": history,
        "best_val_acc": max(history["val_acc"]),
    }, str(model_path))
    print(f"  Model saved: {model_path}")

    # Save history
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    history_path = results_dir / f"history_n{n}_{timestamp}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved: {history_path}")


if __name__ == "__main__":
    main()

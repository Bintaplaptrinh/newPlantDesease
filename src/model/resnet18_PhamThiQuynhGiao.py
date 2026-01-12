# script train resnet18 (chủ yếu dùng để tái train/ghi ra .pth)

import os
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torchvision.models as models

from src.model.mobilenetv3_NguyenHuuTuanPhat import EarlyStopping, validate

from preprocessing import prepare_data


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4


def build_model(num_classes: int, device: str):
    # resnet18 (pretrained imagenet)
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def load_model(path: str, num_classes: int, device: str):
    # load từ disk nếu có, nếu không thì build mới
    if os.path.exists(path):
        m = torch.load(path, weights_only=False, map_location=device)
        m = m.to(device)
        m.eval()
        return m
    return build_model(num_classes=num_classes, device=device)


def load_pth(path: str, device: str):
    # load model đã train từ file .pth
    m = torch.load(path, weights_only=False, map_location=device)
    m = m.to(device)
    m.eval()
    return m


def train(model, train_loader, valid_loader, device: str, config: TrainConfig, criterion, scaler, save_path: str = "resnet.pth"):
    # chỉ train phần fc
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)

    early_stop = EarlyStopping(patience=5)

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")

    _use_adabn = True
    _adabn_batches = 30
    _use_tta_flip = True

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            preds = out.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(y.size(0))

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = validate(
            model,
            valid_loader,
            criterion,
            device,
            use_adabn=_use_adabn,
            adabn_batches=_adabn_batches,
            use_tta_flip=_use_tta_flip,
        )

        scheduler.step()
        lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        print(
            f"Epoch [{epoch+1}/{config.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"LR: {lr}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, save_path)

        if early_stop.step(val_loss):
            print("early stop")
            break

    return history


def train_pipeline(
    save_path: str = "resnet.pth",
    epochs: int = 30,
    batch_size: int = 64,
    num_workers: int = 8,
    image_size: int = 224,
    device: str | None = None,
):
    # download + gộp dataset, sau đó train resnet18
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data = prepare_data(image_size=image_size, batch_size=batch_size, num_workers=num_workers)
    model = build_model(num_classes=data["num_classes"], device=device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler()
    cfg = TrainConfig(epochs=epochs)
    history = train(
        model,
        data["train_loader"],
        data["valid_loader"],
        device=device,
        config=cfg,
        criterion=criterion,
        scaler=scaler,
        save_path=save_path,
    )

    return {
        "history": history,
        "num_classes": data["num_classes"],
        "class_names": data["class_names"],
        "save_path": save_path,
    }

# script train mobilenetv3 (chủ yếu dùng để tái train/ghi ra .pth)

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from preprocessing import prepare_data


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    use_adabn: bool = True
    adabn_batches: int = 30
    use_tta_flip: bool = True


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float):
        # trả về True nếu nên dừng sớm
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def build_model(num_classes: int, device: str):
    # mobilenetv3 large (pretrained imagenet)
    model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
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


def _batchnorm_modules(model: nn.Module):
    # lấy danh sách module batchnorm
    return [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]


def _save_bn_state(model: nn.Module):
    # lưu running stats của batchnorm để restore sau adabn
    bns = _batchnorm_modules(model)
    state = []
    for bn in bns:
        state.append(
            (
                bn.running_mean.detach().clone(),
                bn.running_var.detach().clone(),
                bn.num_batches_tracked.detach().clone() if hasattr(bn, "num_batches_tracked") else None,
            )
        )
    return bns, state


def _restore_bn_state(bns, state):
    # restore running stats của batchnorm
    for bn, (rm, rv, nbt) in zip(bns, state):
        bn.running_mean.copy_(rm)
        bn.running_var.copy_(rv)
        if nbt is not None and hasattr(bn, "num_batches_tracked"):
            bn.num_batches_tracked.copy_(nbt)


@torch.no_grad()
def adapt_batchnorm(model: nn.Module, loader, device: str, num_batches: int = 20):
    # chạy adabn: cập nhật running mean/var theo dữ liệu target
    bns = _batchnorm_modules(model)
    if len(bns) == 0:
        return

    was_training = model.training
    model.train()

    for i, (x, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        _ = model(x)
        if (i + 1) >= num_batches:
            break

    model.train(was_training)


@torch.no_grad()
def validate(model, val_loader, criterion, device: str, use_adabn: bool, adabn_batches: int, use_tta_flip: bool):
    # vòng lặp validation
    if use_adabn:
        bns, bn_state = _save_bn_state(model)
        adapt_batchnorm(model, val_loader, device=device, num_batches=adabn_batches)

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        if use_tta_flip:
            out_flip = model(torch.flip(x, dims=[3]))
            out = (out + out_flip) / 2.0

        loss = criterion(out, y)
        total_loss += float(loss.item())

        preds = out.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    avg_loss = total_loss / len(val_loader)
    acc = correct / total

    if use_adabn:
        _restore_bn_state(bns, bn_state)

    return avg_loss, acc


def train(model, train_loader, valid_loader, device: str, config: TrainConfig, save_path: str = "mobilenet_v3.pth"):
    # chỉ train phần classifier
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)

    scaler = torch.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    early_stop = EarlyStopping(patience=5)

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")

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
            use_adabn=config.use_adabn,
            adabn_batches=config.adabn_batches,
            use_tta_flip=config.use_tta_flip,
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
    save_path: str = "mobilenet_v3.pth",
    epochs: int = 30,
    batch_size: int = 64,
    num_workers: int = 8,
    image_size: int = 224,
    device: str | None = None,
):
    # download + gộp dataset, sau đó train mobilenetv3
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data = prepare_data(image_size=image_size, batch_size=batch_size, num_workers=num_workers)
    model = build_model(num_classes=data["num_classes"], device=device)
    cfg = TrainConfig(epochs=epochs)
    history = train(model, data["train_loader"], data["valid_loader"], device=device, config=cfg, save_path=save_path)

    return {
        "history": history,
        "num_classes": data["num_classes"],
        "class_names": data["class_names"],
        "save_path": save_path,
    }

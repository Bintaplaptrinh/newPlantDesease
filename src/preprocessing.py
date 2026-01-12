# module chuẩn bị dữ liệu: download dataset kaggle, build transforms/datasets/loaders,
# và materialize dataset vào thư mục data/ để tiện chạy web/server

import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


@dataclass
class DatasetPaths:
    train_path_sub: str
    valid_path_sub: str
    train_path: str
    valid_path: str


def build_transforms(image_size: int = 224):
    # augmentations dùng trong notebook
    train_transform_list = [
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.25, hue=0.02)],
            p=0.8,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
            p=0.3,
        ),
        transforms.RandomPerspective(distortion_scale=0.25, p=0.2),
    ]

    try:
        train_transform_list.insert(2, transforms.RandAugment(num_ops=2, magnitude=9))
    except Exception:
        # randaugment không có ở một số phiên bản torchvision
        pass

    train_transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
    ]

    train_transform = transforms.Compose(train_transform_list)

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def download_kaggle_datasets() -> DatasetPaths:
    # download 2 dataset giống notebook (một bản chính augmented + một bản phụ)
    import kagglehub

    # dataset phụ: cấu trúc thư mục NewPlantDiseasesDataset/{train,valid}
    path_sub = kagglehub.dataset_download("tunphtnguynhu/newplantdiseasesdataset")

    # dataset chính: bản augmented, cấu trúc thư mục dài hơn
    path_main = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")

    train_path_sub = str(os.path.join(path_sub, "NewPlantDiseasesDataset", "train"))
    valid_path_sub = str(os.path.join(path_sub, "NewPlantDiseasesDataset", "valid"))

    train_path = str(
        os.path.join(
            path_main,
            "New Plant Diseases Dataset(Augmented)",
            "New Plant Diseases Dataset(Augmented)",
            "train",
        )
    )
    valid_path = str(
        os.path.join(
            path_main,
            "New Plant Diseases Dataset(Augmented)",
            "New Plant Diseases Dataset(Augmented)",
            "valid",
        )
    )

    return DatasetPaths(
        train_path_sub=train_path_sub,
        valid_path_sub=valid_path_sub,
        train_path=train_path,
        valid_path=valid_path,
    )


class RemapByClassName(Dataset):
    def __init__(self, base_ds: ImageFolder, class_to_new_idx: Dict[str, int], allowed_indices: Sequence[int]):
        # wrapper dataset: remap label theo tên class, và chỉ lấy subset theo allowed_indices
        self.base_ds = base_ds
        self.class_to_new_idx = class_to_new_idx
        self.allowed_indices = list(allowed_indices)

    def __len__(self):
        return len(self.allowed_indices)

    def __getitem__(self, i: int):
        base_i = self.allowed_indices[i]
        x, y_old = self.base_ds[base_i]
        class_name = self.base_ds.classes[y_old]
        y_new = self.class_to_new_idx[class_name]
        return x, y_new


def build_datasets(
    paths: DatasetPaths,
    train_transform,
    val_transform,
    exclude_old_injection_classes: Optional[Set[str]] = None,
):
    # build train/valid cho dataset chính, sau đó inject một phần old valid để tăng dữ liệu
    exclude_old_injection_classes = exclude_old_injection_classes or set()

    train_base_ds = ImageFolder(paths.train_path, transform=train_transform)
    valid_base_ds = ImageFolder(paths.valid_path, transform=val_transform)

    # dùng valid của dataset phụ làm nguồn inject (transform khác nhau cho train/val)
    old_valid_train_raw = ImageFolder(paths.valid_path_sub, transform=train_transform)
    old_valid_val_raw = ImageFolder(paths.valid_path_sub, transform=val_transform)

    new_class_to_idx = train_base_ds.class_to_idx

    # map theo tên class để align label giữa 2 dataset
    class_name_map = {
        name: new_class_to_idx[name]
        for name in old_valid_train_raw.classes
        if (name in new_class_to_idx) and (name not in exclude_old_injection_classes)
    }

    old_mappable_indices: List[int] = []
    excluded_samples = 0
    for i, (_, y_old) in enumerate(old_valid_train_raw.samples):
        class_name = old_valid_train_raw.classes[y_old]
        if class_name in exclude_old_injection_classes:
            excluded_samples += 1
            continue
        if class_name in class_name_map:
            old_mappable_indices.append(i)

    old_valid_train_mapped = RemapByClassName(old_valid_train_raw, class_name_map, old_mappable_indices)
    old_valid_val_mapped = RemapByClassName(old_valid_val_raw, class_name_map, old_mappable_indices)

    n_old = len(old_valid_train_mapped)
    # split 50/50 phần old valid: một nửa inject vào train, phần còn lại inject vào valid
    n_old_to_train = int(0.5 * n_old)  # OLD_VALID_SPLIT_TO_TRAIN=0.5

    # cố định seed để kết quả split ổn định
    g = torch.Generator().manual_seed(42)  # SEED=42
    perm = torch.randperm(n_old, generator=g).tolist()
    idx_old_train = perm[:n_old_to_train]
    idx_old_val = perm[n_old_to_train:]

    old_to_train_ds = Subset(old_valid_train_mapped, idx_old_train)
    old_to_valid_ds = Subset(old_valid_val_mapped, idx_old_val)

    train_ds = ConcatDataset([train_base_ds, old_to_train_ds])
    valid_ds = ConcatDataset([valid_base_ds, old_to_valid_ds])

    return {
        "train_ds": train_ds,
        "valid_ds": valid_ds,
        "train_base_ds": train_base_ds,
        "valid_base_ds": valid_base_ds,
        "old_to_train_ds": old_to_train_ds,
        "old_to_valid_ds": old_to_valid_ds,
        "excluded_samples": excluded_samples,
        "class_names": train_base_ds.classes,
        "num_classes": len(train_base_ds.classes),
    }


def build_loaders(train_ds, valid_ds, batch_size: int = 64, num_workers: int = 8):
    # tạo dataloader cho train/valid
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader


def default_exclude_old_injection_classes() -> Set[str]:
    # loại bỏ một số class từ dataset phụ vì dễ làm lệch nhãn/giảm chất lượng
    return {
        "Apple___Black_rot",
        "Cherry_(including_sour)___Powdery_mildew",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Peach___Bacterial_spot",
        "Potato___Early_blight",
        "Potato___healthy",
        "Potato___Late_blight",
        "Strawberry___Leaf_scorch",
        "Tomato___Target_Spot",
    }


def prepare_data(image_size: int = 224, batch_size: int = 64, num_workers: int = 8):
    # download + build + merge dataset, sau đó tạo dataloader
    paths = download_kaggle_datasets()
    train_t, val_t = build_transforms(image_size=image_size)

    pack = build_datasets(
        paths,
        train_t,
        val_t,
        exclude_old_injection_classes=default_exclude_old_injection_classes(),
    )
    train_loader, valid_loader = build_loaders(
        pack["train_ds"],
        pack["valid_ds"],
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return {
        **pack,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "train_transform": train_t,
        "val_transform": val_t,
        "paths": paths,
    }


def _try_make_junction(dst: str, src: str) -> bool:
    # tạo junction trên windows để data/ trỏ tới dataset mà không cần copy
    if os.path.exists(dst):
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    try:
        # mklink /J dùng được cho thư mục trên windows
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", dst, src],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.returncode == 0 and os.path.exists(dst)
    except Exception:
        return False


def materialize_data_folder(paths: DatasetPaths, data_root: str = "data"):
    # đảm bảo dataset download xuất hiện dưới data/ để tiện nộp bài/chạy demo
    base_main = os.path.commonpath([paths.train_path, paths.valid_path])
    base_sub = os.path.commonpath([paths.train_path_sub, paths.valid_path_sub])

    out_dir = os.path.abspath(data_root)
    kaggle_dir = os.path.join(out_dir, "kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    link_main = os.path.join(kaggle_dir, "new-plant-diseases-dataset")
    link_sub = os.path.join(kaggle_dir, "newplantdiseasesdataset")

    ok_main = _try_make_junction(link_main, base_main)
    ok_sub = _try_make_junction(link_sub, base_sub)

    # fallback: nếu tạo junction fail thì ghi path ra file để người dùng tự xử lí
    if not (ok_main and ok_sub):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "DATASET_PATHS.txt"), "w", encoding="utf-8") as f:
            f.write(f"main_base={base_main}\n")
            f.write(f"sub_base={base_sub}\n")

    return {
        "data_root": out_dir,
        "main_base": base_main,
        "sub_base": base_sub,
        "main_link": link_main,
        "sub_link": link_sub,
        "main_link_ok": ok_main,
        "sub_link_ok": ok_sub,
    }

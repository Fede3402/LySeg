import os 
import json
import torch 
from monai.data import PersistentDataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    CropForegroundd, NormalizeIntensityd, RandCropByPosNegLabeld, 
    RandAffined, RandGaussianNoised, RandFlipd, ToTensord, ConcatItemsd
)

def get_loaders(json_path: str, batch_size: int = 4):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Impostiamo la cache su una cartella temporanea (si resetta al riavvio della macchina)
    cache_dir = "/tmp/monai_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # 1. TRASFORMAZIONI COMUNI (Deterministiche, vengono salvate in cache)
    common_transforms = [
        LoadImaged(keys=["pet", "ct", "label"]),
        EnsureChannelFirstd(keys=["pet", "ct","label"]),
        ConcatItemsd(keys=["pet", "ct"], name="image"),
        # Ricampionamento isotropico (2x2x2 mm)
        Spacingd(keys=["image", "label"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear", "nearest")),
        # Rimozione background inutile
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Normalizzazione Z-Score per singolo canale (CT e PET trattate separatamente)
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]

    # 2. TRASFORMAZIONI DI TRAINING (Casuali, applicate "al volo" dalla cache)
    train_transforms = Compose(common_transforms + [
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=(128, 128, 128), pos=1, neg=1, num_samples=2
        ),
        # Augmentation stile nnU-Net
        RandAffined(
            keys=["image", "label"], mode=("bilinear", "nearest"),
            prob=0.2, rotate_range=(0.2, 0.2, 0.2), scale_range=(0.1, 0.1, 0.1)
        ),
        RandGaussianNoised(keys=["image"], prob=0.1, std=0.05),
        RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
        ToTensord(keys=["image", "label", "suv_prior"])
    ])

    # 3. TRASFORMAZIONI DI VALIDATION (Nessuna augmentation)
    val_transforms = Compose(common_transforms + [
        ToTensord(keys=["image", "label", "suv_prior"])
    ])

    # Utilizzo del PersistentDataset per saltare il collo di bottiglia della CPU
    train_ds = PersistentDataset(data=data["training"], transform=train_transforms, cache_dir=cache_dir)
    val_ds = PersistentDataset(data=data["validation"], transform=val_transforms, cache_dir=cache_dir)

    # Con il PersistentDataset e la Multi-GPU, possiamo alzare i num_workers
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    return train_loader, val_loader

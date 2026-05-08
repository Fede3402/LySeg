import os 
import pandas as pd 
import torch 
import json
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    ToTensord,
    ConcatItemsd
)


def get_loaders(json_path: str, batch_size: int = 2):
    """
    Carica il JSON e restituisce i loader per training e validation.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    train_transforms = Compose([
        LoadImaged(keys=["pet","ct","label"]),
        EnsureChannelFirstd(keys=["pet", "ct","label"]),
        ConcatItemsd(keys=["pet","ct"], name = "image"),

        ScaleIntensityd(keys=["image"]),
        
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1, neg=1, num_samples=2
        ),
        ToTensord(keys=["image", "label", "suv_prior"])
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label", "suv_prior"])
    ])

    # Creazione dei Dataset MONAI
    train_ds = Dataset(data=data["training"], transform=train_transforms)
    val_ds = Dataset(data=data["validation"], transform=val_transforms)

    # Creazione dei DataLoader di PyTorch
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    return train_loader, val_loader

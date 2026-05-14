import json
import numpy as np
import torch
from monai.transforms import MapTransform

def compute_train_prior_stats(json_path: str, prior_key: str = "SUV_Mean"):
    """
    Calcola media e deviazione standard di una specifica feature radiomica
    esclusivamente sullo split di training.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    priors = []
    for item in data["training"]:
        if prior_key in item:
            priors.append(float(item[prior_key]))
            
    if not priors:
        raise ValueError(f"Chiave '{prior_key}' non trovata nel training set.")
        
    priors = np.array(priors)
    mean = np.mean(priors)
    std = np.std(priors)
    
    if std == 0:
        std = 1e-8
        
    return mean, std

class ExtractAndNormalizePriorD(MapTransform):
    """
    Estrae una feature specifica dal dizionario, la normalizza (Z-score)
    e la salva in una chiave target come tensore di shape (1,).
    """
    def __init__(self, source_key: str, target_key: str, prior_mean: float, prior_std: float, allow_missing_keys=False):
        super().__init__([source_key], allow_missing_keys)
        self.source_key = source_key
        self.target_key = target_key
        self.mean = prior_mean
        self.std = prior_std

    def __call__(self, data):
        d = dict(data)
        if self.source_key in d:
            val = float(d[self.source_key])
            norm_val = (val - self.mean) / self.std
            d[self.target_key] = torch.tensor([norm_val], dtype=torch.float32)
        return d
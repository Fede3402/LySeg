import json

import pandas as pd
import numpy as np
import torch
from monai.transforms import MapTransform

class AppendZScorePriorMapd(MapTransform):
    """
    Genera un terzo canale spaziale calcolando lo Z-score locale 
    dell'immagine PET basato su media e deviazione standard del fegato.
    Input atteso: (C, D, H, W) -> Output: (C+1, D, H, W).
    """
    def __init__(self, keys, prior_key="prior", mean_idx=0, std_idx=2, pet_idx=1, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.prior_key = prior_key
        self.mean_idx = mean_idx
        self.std_idx = std_idx
        self.pet_idx = pet_idx

    def __call__(self, data):
        d = dict(data)
        
        if self.prior_key not in d:
            raise KeyError(f"Chiave prior '{self.prior_key}' mancante nel JSON.")
            
        prior_list = d[self.prior_key]
        
        if not isinstance(prior_list, list):
            raise TypeError(f"Il valore in '{self.prior_key}' deve essere una lista. Trovato: {type(prior_list)}")
            
        try:
            suv_mean = float(prior_list[self.mean_idx])
            suv_std = float(prior_list[self.std_idx])
        except IndexError:
            raise IndexError(f"Indici {self.mean_idx} o {self.std_idx} fuori range per la lista prior di lunghezza {len(prior_list)}.")
        
        # Prevenzione divisione per zero in casi limite o di segmentazione epatica assente
        if suv_std < 1e-6:
            suv_std = 1e-6
            
        for key in self.keys:
            img = d[key]
            
            # Estrazione esclusiva del canale PET (mantenendo la dimensione del canale)
            pet_channel = img[self.pet_idx:self.pet_idx+1, ...]
            
            # Calcolo della mappa Z-score
            z_score_map = (pet_channel - suv_mean) / suv_std
            
            # Concatenazione: (CT, PET, Z-Map)
            d[key] = torch.cat([img, z_score_map], dim=0)
            
        return d
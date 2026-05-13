import torch 
import torch.nn as nn 

class LiverConditioning(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        inner_dim = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(1, inner_dim),
            nn.SiLU(inplace=True),
            nn.Linear(inner_dim, channels),
            nn.Sigmoid()
          )
        
    def forward(self, x, suv_prior):
        # x : feauture maps (B, C, D, H, W)
        # suv_prior : value from csv (B, 1)
        b, c = x.shape[:2]
        weights = self.fc(suv_prior).view(b, c, 1, 1, 1) # Reshaping vector for multiplication

        return x * weights
    
class FiLMconditioning(nn.Module):
    def __init__(self, num_prior_stats, target_channels, reduction=4, dropout_rate=0.2):
        super().__init__()

        inner_dim = max(target_channels // reduction, 8)

        self.mlp = nn.Sequential(
            nn.LayerNorm(num_prior_stats),
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_prior_stats, inner_dim),
            nn.LayerNorm(inner_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(inner_dim, target_channels*2)
        )
        
        # Zero-Init per l'ultimo layer: parte come trasformazione identità
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, prior_stats):
        # x : feauture maps (B, C, D, H, W) 
        # prior_stats : value from csv (B, num_prior_stats)
        b, c = x.shape[:2]
        # Parameters calculation. Output shape: (B, C*2)
        film_params = self.mlp(prior_stats)
        # Reshape for multiplication along D, H, W
        film_params = film_params.view(b, c * 2, 1, 1, 1)
        # Parameters split. Each one will have shape (B, C, 1, 1, 1)
        gamma, beta = torch.split(film_params, c, dim=1)
        # Transformation : gamma*x + beta
        # Aggiungiamo 1 a gamma per centrare la scalatura attorno all'identità all'inizio del training
        out = x * ( gamma + 1 ) + beta

        return out

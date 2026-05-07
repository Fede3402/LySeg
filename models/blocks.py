import torch 
import torch.nn as nn 

class LiverConditioning(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
          )
        
    def forward(self, x, suv_prior):
        # x : feauture maps (B, C, D, H, W)
        # suv_prior : value from csv (B, 1)
        b,c,_,_,_ = x.size() 
        weights = self.fc(suv_prior).view(b, c, 1, 1, 1) #Reshaping vector for moltiplication

        return x * weights




import os 
import numpy as np
import torch 
import torch.nn as nn
import monai 
import json 

from tqdm import tqdm


from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


from ..models.nets import ConditionedBasicUNet
from ..dataset.data_loader import get_loaders

# CONFIGURATION 

json_path = "kaggle/working/dataset_kaggle.json"
checkpoint_dir = "../checkpoints/"

os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PARAMETERS 
val_interval = 5
max_epochs = 100 
batch_size = 4
lr = 1e-4 

def train():
    # Data loading
    train_loader, val_loader = get_loaders(json_path, batch_size)

    # Model definition 
    model = ConditionedBasicUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        features = (16, 32, 64, 128, 256, 16)
    ).to(device)

    # -- MULTI-GPU --
    if torch.cuda.device_count() > 1:
        print(f"[*] Ottimizzazione: Rilevate {torch.cuda.device_count()} GPU. Attivazione DataParallel!")
        model = torch.nn.DataParallel(model)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_dice = -1
    print("Starting training loop...")

    # Training loop
    for epoch in range(max_epochs):
        
        print(f"{epoch}/{max_epochs}")
        
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_loader, desc=f"training", leave=False)

        for step, batch in enumerate(progress_bar, 1):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            suv_prior = batch["suv_prior"].view(-1,1).float().to(device)

            optimizer.zero_grad()

            outputs = model(inputs, suv_prior)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress_bar.set_postfix({"Loss_Batch": f"{loss.item():.4f}"})

        epoch_loss /= step
        print(f"[*] Fine Epoca {epoch} - Training Loss Media: {epoch_loss:.4f}")

        
        if epoch % val_interval == 0:
            print("Starting validation")

            model.eval()

            val_progress_bar = tqdm(val_loader, desc="Validation", leave=False)

            with torch.no_grad():
                for val_batch in val_progress_bar:
                    val_images = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)
                    val_suv = val_batch["suv_prior"].view(-1, 1).float().to(device)

                    val_outputs = sliding_window_inference(
                        val_images, (128, 128, 128), 4, model, suv_prior=val_suv
                    )
                    
                    val_outputs = (torch.sigmoid(val_outputs) > 0.5).float()
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                
                print(f">>> VALIDAZIONE Epoca {epoch} completata! Dice Score: {metric:.4f}")

                # Salvataggio del modello migliore
                if metric > best_dice:
                    best_dice = metric
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
                    print(f"    [!] Nuovo record di Dice Score! Modello salvato in {checkpoint_dir}")

if __name__ == "__main__":
    train()


            



        





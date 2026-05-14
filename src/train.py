import os 
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch 
import torch.nn as nn
import monai 
import json 

from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter



from ..models.nets import PriorUNet
from ..dataset.data_loader import get_loaders
from ..utilities.utils import get_film_stats, get_grad_norm, save_checkpoint, get_window_predictor




# CONFIGURATION 

json_path = "kaggle/working/dataset_kaggle.json"
checkpoint_dir = "../checkpoints"

os.makedirs(checkpoint_dir, exist_ok=True)


# PARAMETERS 
val_interval = 5
max_epochs = 100 
batch_size = 4
lr = 1e-4 

def train():
    # Data loading
    train_loader, val_loader = get_loaders(json_path, batch_size)

    # Model definition 
    model = PriorUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        features = (16, 32, 64, 128, 256),
        num_prior_stats=11,
        dropout=0.2
    )

    if torch.cuda.device_count() > 1:
        print(f"[*] Ottimizzazione: Rilevate {torch.cuda.device_count()} GPU. Attivazione DataParallel!")
        model = torch.nn.DataParallel(model)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. SPOSTI IL MODELLO SULLA GPU QUI, una volta sola
    model = model.to(device)


    # Loss function and optimizer
    loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    loss_history = []
    dice_history = []

    best_dice = -1
    print("Starting training loop...")

    writer = SummaryWriter("runs/prior_unet_experiment")


    # Training loop
    for epoch in range(max_epochs):
        
        print(f"{epoch}/{max_epochs}")
        
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_loader, desc=f"training", leave=False, dynamic_ncols=True)

        for step, batch in enumerate(progress_bar, 1):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            priors = batch["prior"].float().to(device)

            optimizer.zero_grad()

            outputs = model(inputs, priors)
            loss = loss_function(outputs, labels)
            loss.backward()

            raw_model = model.module if hasattr(model, "module") else model

            # Monitora spinta gradienti su Film
            g_norm = get_grad_norm(model)

            optimizer.step()

            # Monitora quanto sono "cresciuti" i pesi del FiLM
            w_norm = get_film_stats(raw_model)

            global_step = epoch * len(train_loader) + step
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("Train/GradNorm", g_norm, global_step)
            writer.add_scalar("Train/FiLMWeightNorm", w_norm, global_step)


            epoch_loss += loss.item()

            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "G": f"{g_norm:.5f}",
                "W": f"{w_norm:.5f}"
               })

        epoch_loss /= step
        
        loss_history.append(epoch_loss)

        writer.add_scalar("Train/Loss_epoch", epoch_loss, epoch)

        print(f"[*] Fine Epoca {epoch} - Training Loss Media: {epoch_loss:.4f}")

        del inputs, labels, priors, outputs, loss
        torch.cuda.empty_cache()

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("Train/LearningRate", current_lr, epoch)

        if epoch % val_interval == 0:
            print("Starting validation")

            model.eval()

            val_progress_bar = tqdm(val_loader, desc="Validation", leave=False)

            with torch.no_grad():
                for val_batch in val_progress_bar:
                    val_images = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)
                    val_priors = val_batch["prior"].float().to(device)

                    # Usiamo questa funzione 
                    predictor = get_window_predictor(model, val_priors)


                    val_outputs = sliding_window_inference(
                        val_images, 
                        (128,128,128), 
                        sw_batch_size=1, 
                        predictor = predictor 
                        )
                    
                    val_outputs = (torch.sigmoid(val_outputs) > 0.5).float()
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_history.append(metric)


                dice_metric.reset()
                
                print(f">>> VALIDAZIONE Epoca {epoch} completata! Dice Score: {metric:.4f}")

                # Salvataggio del modello migliore
                if metric > best_dice:
                    best_dice = metric
                    save_checkpoint(epoch, model, optimizer, scheduler, best_dice, loss_history, dice_history, checkpoint_dir)
                    print(f"    [!] Nuovo record di Dice Score! Modello salvato in {checkpoint_dir}")
                
                del val_images, val_labels, val_priors, val_outputs
                # Svuotiamo la cache della GPU fisicamente
                torch.cuda.empty_cache()
    
    writer.close()            

if __name__ == "__main__":
    train()


            



        

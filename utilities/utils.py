import torch 
import os 

import torch.nn as nn
import numpy as np

from collections import OrderedDict



def get_film_stats(model):
    """
    Calcola la norma media dei pesi dei moduli FiLM per monitorare l'apprendimento.
    """
    total_norm = 0.0
    count = 0
    # Cerchiamo tutti i moduli FiLMconditioning nella rete
    for name, module in model.named_modules():
        if "film" in name.lower(): # Cerca i layer che hai chiamato film4, film3, ecc.
            for param in module.parameters():
                if param.requires_grad:
                    total_norm += param.norm(2).item()
                    count += 1
    return total_norm / count if count > 0 else 0.0

def get_grad_norm(model):
    """
    Calcola la norma dei gradienti dei moduli FiLM.
    Utile da chiamare DOPO loss.backward() e PRIMA di optimizer.step().
    """
    total_grad_norm = 0.0
    count = 0
    for name, p in model.named_parameters():
        if "film" in name.lower() and p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item()
            count += 1
    return total_grad_norm / count if count > 0 else 0.0

def save_checkpoint(
        epoch, 
        model, 
        optimizer, 
        scheduler, 
        dice_best, 
        loss_history,
        dice_history,
        checkpoint_dir
     ):
    
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state,
        'scheduler_state_dict': scheduler.state_dict(),
        'dice_best': dice_best,
        'loss_history': loss_history,
        'dice_history': dice_history
    
    }

    # checkpoint_dir configuration 
    os.makedir(os.path.dirname(checkpoint_dir), exist_ok=True)

    torch.save(state, checkpoint_dir)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Carica i pesi adattandosi a qualsiasi configurazione (1 GPU o 2+ GPU).
    """
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None

    checkpoint = torch.load(filepath, map_location='cpu') # Carica su CPU per sicurezza
    state_dict = checkpoint['model_state_dict']

    # 1. Se il file ha 'module.' ma il modello attuale NO
    new_state_dict = OrderedDict()
    curr_has_module = hasattr(model, 'module')
    
    for k, v in state_dict.items():
        name = k
        if k.startswith('module.') and not curr_has_module:
            name = k[7:] # rimuove 'module.'
        elif not k.startswith('module.') and curr_has_module:
            name = 'module.' + k # aggiunge 'module.'
        new_state_dict[name] = v

    # Carica i pesi modificati
    model.load_state_dict(new_state_dict)
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    print(f"Checkpoint caricato correttamente (Epoca {checkpoint['epoch']})")
    return checkpoint

def get_window_predictor(model, priors):
    """
    Factory function che restituisce il predictor per la sliding window.
    Gestisce automaticamente l'espansione dei prior in base al batch size della patch.
    """
    def window_pred(patch_batch):
        # Estrae quanti campioni sta analizzando la sliding window in questo momento
        current_batch_size = patch_batch.shape[0]
        
        # Clona i prior per farli combaciare con il batch size delle patch
        expanded_priors = priors.repeat(current_batch_size, 1)
        
        # Esegue il forward pass sul modello
        return model(patch_batch, expanded_priors)
        
    return window_pred

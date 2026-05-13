import torch 
import torch.nn as nn
import numpy as np



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
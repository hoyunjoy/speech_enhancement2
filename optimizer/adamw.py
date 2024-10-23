import torch

def Optimizer(parameters, lr, weight_decay, adam_b1, adam_b2, **kwargs):
    
    print('Initialized AdamW optimizer')
    
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=[adam_b1, adam_b2])
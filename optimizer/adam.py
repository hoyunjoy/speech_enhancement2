import torch

def Optimizer(parameters, lr, weight_decay, **kwargs):
    
    print('Initialized Adam optimizer')
    
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
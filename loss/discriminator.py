import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb, numpy
from models.generator import phase_losses

class LossFunction(nn.Module):
    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()
        
        self.mse_loss = nn.MSELoss()
        
        print('Initialized MP-SENet Discriminator Loss')
        
    def forward(self, data, label):
        # Flatten to match lable shape
        data = data.flatten()
        
        # L2 magnitude loss
        nloss = self.mse_loss(data, label)
        
        return nloss
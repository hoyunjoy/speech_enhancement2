import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb, numpy

class LossFunction(nn.Module):
    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()
        
        self.criterion = torch.nn.MSELoss()
        
        print('Initialized MSE Loss')
        
    def forward(self, x, label=None):
        
        nloss = self.criterion(x, label)
        
        return nloss
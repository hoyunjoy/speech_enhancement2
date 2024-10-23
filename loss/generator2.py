import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

def phase_losses(phase_r, phase_g, n_fft, **kwargs):

    dim_freq = n_fft // 2 + 1
    dim_time = phase_r.size(-1)

    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - torch.eye(dim_freq)).to(phase_g.device)
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - torch.eye(dim_time)).to(phase_g.device)
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r-phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r-gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r-iaf_g))
    
    # 코사인 유사도 기반 손실 계산 (CSPC Loss)
    cosine_similarity = torch.cos(anti_wrapping_function(phase_r-phase_g))
    cspc_loss = torch.mean(1 - cosine_similarity)

    return ip_loss, gd_loss, iaf_loss, cspc_loss

def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

class LossFunction(nn.Module):
    def __init__(self, n_fft, **kwargs):
        super(LossFunction, self).__init__()
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.phase_losses = phase_losses
        self.n_fft = n_fft
        
        
        
        print('Initialized MP-SENet Generator Loss')
        
    def forward(self, data, label=None):
        
        enhan_wav, enhan_mag, enhan_pha, enhan_com, metric_g   = data
        clean_wav, clean_mag, clean_pha, clean_com, one_labels = label
        
        # L2 magnitude loss
        loss_mag = self.mse_loss(clean_mag, enhan_mag)
        
        # Anti-wrapping phase loss
        loss_ip, loss_gd, loss_iaf, cspc_loss = self.phase_losses(clean_pha, enhan_mag, self.n_fft)
        loss_pha = loss_ip + loss_gd + loss_iaf + cspc_loss
        
        # L2 complex loss
        loss_com = self.mse_loss(clean_com, enhan_com) * 2
        
        # # L2 consistency loss
        # loss_stft = self.mse_loss(enhan_com, enhan_com_hat) * 2
        
        # Time loss
        loss_time = self.l1_loss(clean_wav, enhan_wav)
        
        # Metric loss
        loss_metric = self.mse_loss(metric_g.flatten(), one_labels)
        
        ## Ablation study is needed.
        loss_gen_all = (loss_mag * 0.9
                        + loss_pha * 0.3
                        + loss_com * 0.1
                        + loss_metric * 0.05
                        + loss_time * 0.2)
        
        nloss = loss_gen_all
        
        return nloss, (loss_mag, loss_pha, loss_com, loss_metric, loss_time)
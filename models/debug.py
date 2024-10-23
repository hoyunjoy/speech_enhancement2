import pdb
import torch
import torch.nn as nn

class SpeechEnhancementNet(nn.Module):
    def __init__(self, batch_size, n_fft, **kwargs):
        super(SpeechEnhancementNet, self).__init__()
        
        self.f_bin      = n_fft//2 + 1
        self.batch_size = batch_size
        
        # encoder
        self.encoder = nn.ModuleDict({                           # [batch_size, num_channels, f_bins, time]
            "encoder_block1": self.create_encoder_block(1, 8),   # [batch_size,   1, f_bins,   time  ] -> [batch_size, 128, f_bins-2, time-2]
            "encoder_block2": self.create_encoder_block(8, 4),   # [batch_size, 128, f_bins-2, time-2] -> [batch_size,  64, f_bins-4, time-4]
            "encoder_block3": self.create_encoder_block(4, 2)    # [batch_size,  64, f_bins-4, time-4] -> [batch_size,  32, f_bins-6, time-6]
        })
        
        # lstm
        self.lstm_layer = nn.LSTM(input_size=2*(self.f_bin-6), hidden_size=2*(self.f_bin-6), num_layers=2, batch_first=True, bidirectional=False)

        # decoder
        self.decoder = nn.ModuleDict({
            "decoder_block1": self.create_decoder_block(2, 4),                                  # [batch_size,  32, f_bins-6, time-6] -> [batch_size,  64, f_bins-4, time-4]
            "decoder_block2": self.create_decoder_block(4, 8),                                  # [batch_size,  64, f_bins-4, time-4] -> [batch_size, 128, f_bins-2, time-2]
            "decoder_block3": nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3), # [batch_size, 128, f_bins-2, time-2] -> [batch_size,   1, f_bins  , time  ]
        })
    
    def create_encoder_block(self, in_channels, out_channels):
        block = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ])
        return nn.Sequential(*block)
    
    def create_decoder_block(self, in_channels, out_channels):
        block = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU()
        ])
        return nn.Sequential(*block)
    
    def forward(self, x):
        
        # encoder
        for encoder_block in self.encoder:
            x = self.encoder[encoder_block](x)
        
        # lstm
        batch_size, n_channels, f_bins, time = x.shape
        x = x.reshape(batch_size, n_channels*(f_bins), time).permute(0, 2, 1)   # [batch_size, 32, f_bins-6, time-6]                -> [batch_size, time-6(=seq_len-6), 32*(f_bins-6)]
        x, _ = self.lstm_layer(x)                                               # [batch_size, time-6(=seq_len-6), 32*(f_bins-6)]   -> [batch_size, time-6(=seq_len-6), 32*(f_bins-6)]
        x = x.permute(0, 2, 1).reshape(batch_size, n_channels, f_bins, time)    # [batch_size, time-6(=seq_len-6), 32*(f_bins-6)]   -> [batch_size, 32, f_bins-6, time-6(=seq_len-6)]
        
        # decoder
        for decoder_block in self.decoder:
            x = self.decoder[decoder_block](x)
        
        return x

def MainModel(**kwargs):
    model = SpeechEnhancementNet(**kwargs)
    return model
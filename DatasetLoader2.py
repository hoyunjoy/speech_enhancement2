import torch
import torch.nn.functional as F
import torchaudio
import pdb
import os
import random
import numpy
import glob
import sys
import librosa
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

def mag_pha_stft(wav, n_fft, hop_length, win_length, compress_factor=1.0, center=True, **kwargs):
    hann_window = torch.hann_window(win_length).to(wav.device)
    stft_spec = torch.stft(wav, n_fft, hop_length=hop_length, win_length=win_length, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    # stft_spec = torch.stft(wav, n_fft, hop_length=hop_length, win_length=win_length, window=hann_window,
    #                        center=center, pad_mode='reflect', normalized=True, return_complex=True)
    mag = torch.abs(stft_spec)
    pha = torch.angle(stft_spec)
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)

    return mag, pha, com

def mag_pha_istft(mag, pha, n_fft, hop_length, win_length, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_length).to(com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_length, win_length=win_length,
                      window=hann_window, center=center)
    # wav = torch.istft(com, n_fft, hop_length=hop_length, win_length=win_length,
    #                   window=hann_window, center=center, normalized=True)

    return wav

# def slicing_audio(waveform, max_length):
    
#     waveform = waveform.squeeze(0) 
#     if len(waveform) > max_length:
#         waveform = waveform[:max_length]
#     else:
#         waveform = F.pad(waveform, (0, max_length-len(waveform)))

#     waveform = waveform.unsqueeze(0)
#     return waveform

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

class dataset_loader(Dataset):
    
    def __init__(self, clean_dir, noisy_dir, n_fft, win_length, hop_length, window_fn, power, sample_rate, compress_factor, **kwargs):
        
        random.seed(1234)
        
        # stft
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_fn = window_fn
        self.power = power
        self.sample_rate = sample_rate
        self.compress_factor = compress_factor
        
        # '/' is needed at the very front of path
        self.clean_folder_path = os.path.join(clean_dir)
        self.noisy_folder_path = os.path.join(noisy_dir)
        self.walker = sorted(str(p.stem) for p in Path(self.clean_folder_path).glob("*.wav"))
        
    def __len__(self):
        return len(self.walker)
    
    def __getitem__(self, idx):
        fileid = self.walker[idx]
        fileid_with_extension = fileid+'.wav'
        clean_path = os.path.join(self.clean_folder_path, fileid_with_extension) 
        noisy_path = os.path.join(self.noisy_folder_path, fileid_with_extension)
        
        clean_wav, _ = librosa.load(clean_path, sr=self.sample_rate)
        noisy_wav, _ = librosa.load(noisy_path, sr=self.sample_rate)
        clean_wav, noisy_wav = torch.FloatTensor(clean_wav), torch.FloatTensor(noisy_wav)
        
        ## Normalize
        norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0))
        clean_wav = (clean_wav * norm_factor).unsqueeze(0)
        noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
        
        assert clean_wav.size(1) == noisy_wav.size(1)
        
        ## audio slicing
        if clean_wav.size(1) >= self.sample_rate*2:
            max_wav_start = clean_wav.size(1) - self.sample_rate*2
            wav_start = random.randint(0, max_wav_start)
            clean_wav = clean_wav[:, wav_start: wav_start+self.sample_rate*2]
            noisy_wav = noisy_wav[:, wav_start: wav_start+self.sample_rate*2]
        else:
            clean_wav = F.pad(clean_wav, (0, self.sample_rate*2 - clean_wav.size(1)), 'constant')
            noisy_wav = F.pad(noisy_wav, (0, self.sample_rate*2 - noisy_wav.size(1)), 'constant')

        
        # # clean waveform preprocessing
        # clean_wav, _ = librosa.load(clean_path, sr=self.sample_rate)
        # clean_wav    = torch.from_numpy(clean_wav).unsqueeze(0)
        # clean_wav    = slicing_audio(waveform=clean_wav, max_length=self.sample_rate*2)

        # # noisy waveform preprocessing
        # noisy_wav, _ = librosa.load(noisy_path, sr=self.sample_rate)
        # noisy_wav    = torch.from_numpy(noisy_wav).unsqueeze(0)
        # noisy_wav    = slicing_audio(waveform=noisy_wav, max_length=self.sample_rate*2)
        
        # # # normalize
        # # norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0))
        # # clean_wav = (clean_wav * norm_factor)
        # # noisy_wav = (noisy_wav * norm_factor)
        
        clean_mag, clean_pha, clean_com = mag_pha_stft(clean_wav,
                                                        self.n_fft,
                                                        self.hop_length,
                                                        self.win_length,
                                                        self.compress_factor)
        
        noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_wav,
                                                        self.n_fft,
                                                        self.hop_length,
                                                        self.win_length,
                                                        self.compress_factor)
        
        return clean_wav.squeeze(0), clean_mag.squeeze(0), clean_pha.squeeze(0), clean_com.squeeze(0), noisy_mag.squeeze(0), noisy_pha.squeeze(0)



# class test_dataset_loader(Dataset):
    
#     def __init__(self, test_path, n_fft, win_length, hop_length, window_fn, power, sample_rate, compress_factor, **kwargs):

#         # stft
#         self.n_fft = n_fft
#         self.win_length = win_length
#         self.hop_length = hop_length
#         self.window_fn = window_fn
#         self.power = power
#         self.sample_rate = sample_rate
#         self.compress_factor = compress_factor

#         # '/' is needed at the very front of path
#         root = os.fspath(test_path)                                                                 # '/mnt/lynx3/datasets/LibriMix/Libri2Mix/wav8k/max/dev/'
#         self.clean_folder_path = os.path.join(root, Path("clean_testset_wav_16k"))                  # '/mnt/lynx3/datasets/LibriMix/Libri2Mix/wav8k/max/dev/s1'
#         self.noisy_folder_path = os.path.join(root, Path("noisy_testset_wav_16k"))                  # '/mnt/lynx3/datasets/LibriMix/Libri2Mix/wav8k/max/dev/mix_single'
#         self.walker = sorted(str(p.stem) for p in Path(self.clean_folder_path).glob("*.wav"))       # '/mnt/lynx3/datasets/LibriMix/Libri2Mix/wav8k/max/dev/s1/*.wav'
    
#     def __len__(self):
#         return len(self.walker)
    
#     def __getitem__(self, idx):
#         fileid = self.walker[idx]
#         fileid_with_extension = fileid+'.wav'
#         clean_path = os.path.join(self.clean_folder_path, fileid_with_extension) 
#         noisy_path = os.path.join(self.noisy_folder_path, fileid_with_extension)
        
#         ## clean waveform preprocessing
#         clean_wav, _ = librosa.load(clean_path, sr=self.sample_rate)
#         clean_wav    = torch.from_numpy(clean_wav).unsqueeze(0)
#         clean_wav    = slicing_audio(waveform=clean_wav, max_length=self.sample_rate*2)

#         ## noisy waveform preprocessing
#         noisy_wav, _ = librosa.load(noisy_path, sr=self.sample_rate)
#         noisy_wav    = torch.from_numpy(noisy_wav).unsqueeze(0)
#         noisy_wav    = slicing_audio(waveform=noisy_wav, max_length=self.sample_rate*2)

#         # ## normalize
#         # norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0))
#         # clean_wav = (clean_wav * norm_factor)
#         # noisy_wav = (noisy_wav * norm_factor)

#         clean_mag, clean_pha, clean_com = mag_pha_stft(clean_wav,
#                                                         self.n_fft,
#                                                         self.hop_length,
#                                                         self.win_length,
#                                                         self.compress_factor)
        
#         noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_wav,
#                                                         self.n_fft,
#                                                         self.hop_length,
#                                                         self.win_length,
#                                                         self.compress_factor)
        
#         return (clean_wav.squeeze(0), clean_mag.squeeze(0), clean_pha.squeeze(0), clean_com.squeeze(0), noisy_mag.squeeze(0), noisy_pha.squeeze(0))
    
#         # clean_mag, clean_pha, noisy_mag, noisy_pha: [1, 201, 321]
#         # clean_com, noisy_com: [1, 201, 321, 2]

if __name__ == '__main__':
    ## Define test data loader
    train_dataset = dataset_loader(clean_dir="/mnt/lynx4/datasets/VOICE_DEMAND/clean_trainset_28spk_wav_16k",
                                    noisy_dir="/mnt/lynx4/datasets/VOICE_DEMAND/noisy_trainset_28spk_wav_16k",
                                    n_fft=400,
                                    win_length=400,
                                    hop_length=100,
                                    window_fn=torch.hann_window,
                                    power=None,
                                    sample_rate=16000,
                                    compress_factor=0.3)
    
    ## Print shape of tensors
    clean_wav, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = train_dataset[1]
    print("clean_wav.shape:", clean_wav.shape) # [32000]
    print("clean_mag.shape:", clean_mag.shape) # [201, 321]
    print("clean_pha.shape:", clean_pha.shape) # [201, 321]
    print("clean_com.shape:", clean_com.shape) # [201, 321, 2]
    print("noisy_mag.shape:", noisy_mag.shape) # [201, 321]
    print("noisy_pha.shape:", noisy_pha.shape) # [201, 321]
    
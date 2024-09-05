import torch
import torch.nn.functional as F
import torchaudio
import pdb
import os
import random
import numpy
import glob
import sys
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


## waveform to spectrogram
## if power == None, return complex spectrogram.
def spec(n_fft, win_length, hop_length, window_fn, power, waveform):
    return torchaudio.transforms.Spectrogram(n_fft=n_fft,
                                             win_length=win_length,
                                             hop_length=hop_length,
                                             window_fn=window_fn,
                                             power=power)(waveform)

def resample(orig_freq, new_freq, waveform):
    
    return torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)(waveform)


def slicing_audio(waveform, max_length):
    
    waveform = waveform.squeeze(0) 
    if len(waveform) > max_length:
        waveform = waveform[:max_length]
    else:
        waveform = F.pad(waveform, (0, max_length-len(waveform)))

    waveform = waveform.unsqueeze(0)
    
    return waveform



def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)



class train_dataset_loader(Dataset):
    
    def __init__(self, train_path, n_fft, win_length, hop_length, window_fn, power, sample_rate, **kwargs):
        
        # stft
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_fn = window_fn
        self.power = power
        self.sample_rate = sample_rate
        
        # '/' is needed at the very front of path
        root = os.fspath(train_path)                                                             # '/mnt/lynx3/datasets/LibriMix/Libri2Mix/wav8k/max/train-100/'
        self.clean_folder_path = os.path.join(root, Path("clean_trainset_28spk_wav_16k"))         # '/mnt/lynx3/datasets/LibriMix/Libri2Mix/wav8k/max/train-100/s1'
        self.noisy_folder_path = os.path.join(root, Path("noisy_trainset_28spk_wav_16k"))         # '/mnt/lynx3/datasets/LibriMix/Libri2Mix/wav8k/max/train-100/mix_single'
        self.walker = sorted(str(p.stem) for p in Path(self.clean_folder_path).glob("*.wav"))    # '/mnt/lynx3/datasets/LibriMix/Libri2Mix/wav8k/max/train-100/s1/*.wav'
        
    def __len__(self):
        return len(self.walker)
    
    def __getitem__(self, idx):
        fileid = self.walker[idx]
        fileid_with_extension = fileid+'.wav'
        clean_path = os.path.join(self.clean_folder_path, fileid_with_extension) 
        noisy_path = os.path.join(self.noisy_folder_path, fileid_with_extension)
        
        ## clean waveform preprocessing
        clean_waveform, clean_sample_rate = torchaudio.load(clean_path, normalize=True)
        clean_waveform = resample(orig_freq=clean_sample_rate, new_freq=self.sample_rate, waveform=clean_waveform)
        clean_waveform = slicing_audio(waveform=clean_waveform, max_length=clean_sample_rate*3)

        ## noisy waveform preprocessing
        noisy_waveform, noisy_sample_rate = torchaudio.load(noisy_path, normalize=True)
        noisy_waveform = resample(orig_freq=noisy_sample_rate, new_freq=self.sample_rate, waveform=noisy_waveform)
        noisy_waveform = slicing_audio(waveform=noisy_waveform, max_length=noisy_sample_rate*3)
        
        ## applying transform to waveform
        clean_complex_spectrogram = spec(self.n_fft,
                                         self.win_length,
                                         self.hop_length,
                                         self.window_fn,
                                         self.power,
                                         clean_waveform)
        
        noisy_complex_spectrogram = spec(self.n_fft,
                                         self.win_length,
                                         self.hop_length,
                                         self.window_fn,
                                         self.power,
                                         noisy_waveform)
        
        ## getting magnitude information
        clean_magnitude_spectrogram = clean_complex_spectrogram.abs()
        noisy_magnitude_spectrogram = noisy_complex_spectrogram.abs()
        
        ## getting phase information. return radian-valued tensor.
        clean_phase_spectrogram = clean_complex_spectrogram.angle()
        noisy_phase_spectrogram = noisy_complex_spectrogram.angle()
        
        return clean_magnitude_spectrogram, noisy_magnitude_spectrogram, clean_phase_spectrogram, noisy_phase_spectrogram, clean_waveform, noisy_waveform

class test_dataset_loader(Dataset):
    
    def __init__(self, test_path, n_fft, win_length, hop_length, window_fn, power, sample_rate, **kwargs):

        # stft
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_fn = window_fn
        self.power = power
        self.sample_rate = sample_rate

        # '/' is needed at the very front of path
        root = os.fspath(test_path)                                                                 # '/mnt/lynx3/datasets/LibriMix/Libri2Mix/wav8k/max/dev/'
        self.clean_folder_path = os.path.join(root, Path("clean_testset_wav_16k"))                                     # '/mnt/lynx3/datasets/LibriMix/Libri2Mix/wav8k/max/dev/s1'
        self.noisy_folder_path = os.path.join(root, Path("noisy_testset_wav_16k"))                             # '/mnt/lynx3/datasets/LibriMix/Libri2Mix/wav8k/max/dev/mix_single'
        self.walker = sorted(str(p.stem) for p in Path(self.clean_folder_path).glob("*.wav"))       # '/mnt/lynx3/datasets/LibriMix/Libri2Mix/wav8k/max/dev/s1/*.wav'
    
    def __len__(self):
        return len(self.walker)
    
    def __getitem__(self, idx):
        fileid = self.walker[idx]
        fileid_with_extension = fileid+'.wav'
        clean_path = os.path.join(self.clean_folder_path, fileid_with_extension) 
        noisy_path = os.path.join(self.noisy_folder_path, fileid_with_extension)
        
        ## clean waveform preprocessing
        clean_waveform, clean_sample_rate = torchaudio.load(clean_path, normalize=True)
        clean_waveform = resample(orig_freq=clean_sample_rate, new_freq=self.sample_rate, waveform=clean_waveform)
        clean_waveform = slicing_audio(waveform=clean_waveform, max_length=clean_sample_rate*3)

        ## noisy waveform preprocessing
        noisy_waveform, noisy_sample_rate = torchaudio.load(noisy_path, normalize=True)
        noisy_waveform = resample(orig_freq=noisy_sample_rate, new_freq=self.sample_rate, waveform=noisy_waveform)
        noisy_waveform = slicing_audio(waveform=noisy_waveform, max_length=noisy_sample_rate*3)

        ## applying transform to waveform
        clean_complex_spectrogram = spec(self.n_fft,
                                         self.win_length,
                                         self.hop_length,
                                         self.window_fn,
                                         self.power,
                                         clean_waveform)
        
        noisy_complex_spectrogram = spec(self.n_fft,
                                         self.win_length,
                                         self.hop_length,
                                         self.window_fn,
                                         self.power,
                                         noisy_waveform)
        
        # getting magnitude information
        clean_magnitude_spectrogram = clean_complex_spectrogram.abs()
        noisy_magnitude_spectrogram = noisy_complex_spectrogram.abs()
        
        ## getting phase information. return radian-valued tensor.
        clean_phase_spectrogram = clean_complex_spectrogram.angle()
        noisy_phase_spectrogram = noisy_complex_spectrogram.angle()
        
        return clean_magnitude_spectrogram, noisy_magnitude_spectrogram, clean_phase_spectrogram, noisy_phase_spectrogram, clean_waveform, noisy_waveform

class practice_dataset_loader(Dataset):
    
    def __init__(self, **kwargs):
        self.noisy_folder_path = os.fspath('/home/hoyun/SpeechEnhancement/autoencoder_project/practice_audio')
        self.walker = sorted(str(p.stem) for p in Path(self.noisy_folder_path).glob("*.wav"))
    
    def __len__(self):
        return len(self.walker)
    
    def __getitem__(self, idx):
        fileid = self.walker[idx]
        fileid_with_extension = fileid + '.wav'
        noisy_path = os.path.join(self.noisy_folder_path, fileid_with_extension)
        
        noisy_waveform, noisy_sample_rate = torchaudio.load(noisy_path, normalize=True)
        
        resampling = torchaudio.transforms.Resample(
            orig_freq=noisy_sample_rate,
            new_freq=8000,
            )
        noisy_waveform = resampling(noisy_waveform)
        noisy_waveform = slicing_audio(waveform=noisy_waveform, max_length=24000)
        
        # define spectrogram transform
        complex_spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=None) # complex-valued spectrogram returned
        
        noisy_complex_spectrogram = complex_spectrogram(noisy_waveform)
        noisy_magnitude_spectrogram = noisy_complex_spectrogram.abs()
        noisy_phase_spectrogram = noisy_complex_spectrogram.angle()
        
        return noisy_magnitude_spectrogram, noisy_phase_spectrogram
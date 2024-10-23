import torch
import torchaudio

filename = "test5/clean.wav"
waveform, sample_rate = torchaudio.load(filename)
print(sample_rate)
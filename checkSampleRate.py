import torch
import torchaudio

filename = "test/test_noisy.wav"
waveform, sample_rate = torchaudio.load(filename)
print(sample_rate)
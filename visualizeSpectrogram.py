import torchaudio
import matplotlib.pyplot as plt
import librosa
import torch

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("frames")
    # Adjust aspect ratio to make the plot less stretched vertically
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest", cmap="inferno")

audio = "clean"
folder = "test7/"

# Load the audio file
audio_path = folder + audio + ".wav"
image_path = folder + audio + "_spec.png"

# Load the audio file
waveform, sr = torchaudio.load(audio_path)

n_fft = 400
hop_length = 100
win_length = 400
hann_window = torch.hann_window(win_length)
center = True

# Convert the waveform into a spectrogram
spec = torch.stft(waveform,
                        n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        window=hann_window,
                        center=center,
                        pad_mode='reflect',
                        # normalized=True,
                        normalized=False,
                        return_complex=True)

# Draw a single spectrogram
fig, ax = plt.subplots(figsize=(6, 6))  # Square figure
plot_spectrogram(torch.abs(spec.squeeze(0)), title=audio, ax=ax)

# Adjust layout and save the figure
fig.tight_layout()
plt.savefig(image_path)
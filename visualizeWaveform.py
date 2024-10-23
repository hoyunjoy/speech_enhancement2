import torchaudio
import matplotlib.pyplot as plt

# Load audio files
clean_audio_path = "test4/test4_clean.wav"
denoised_audio_path = "test4/test4_denoised.wav"
noisy_audio_path = "test4/test4_noisy.wav"
image_path = "test4/audio_waveforms.png"

# Load the audio files
clean_audio, sr = torchaudio.load(clean_audio_path)
denoised_audio, _ = torchaudio.load(denoised_audio_path)
noisy_audio, _ = torchaudio.load(noisy_audio_path)

# Reducing the size of the data by extracting only the first few seconds (to manage memory issues)
sample_duration = 2  # seconds
num_samples = int(sr * sample_duration)

clean_audio_segment = clean_audio[:, :num_samples]
denoised_audio_segment = denoised_audio[:, :num_samples]
noisy_audio_segment = noisy_audio[:, :num_samples]

# Plot the waveforms for clean, noisy, and denoised audio
plt.figure(figsize=(15, 8))

# Clean audio
plt.subplot(3, 1, 1)
plt.plot(clean_audio_segment.t().numpy())
plt.title('Clean Audio')
plt.grid()

# Noisy audio
plt.subplot(3, 1, 2)
plt.plot(noisy_audio_segment.t().numpy())
plt.title('Noisy Audio')
plt.grid()

# Denoised audio
plt.subplot(3, 1, 3)
plt.plot(denoised_audio_segment.t().numpy())
plt.title('Denoised Audio')
plt.grid()

plt.tight_layout()
plt.savefig(image_path)
import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms
import numpy, sys, random
import time, itertools, importlib
import torchmetrics
import matplotlib.pyplot as plt
import librosa
from pesq import pesq_batch

from torchmetrics.audio import ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality, ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio
from matplotlib.patches import Rectangle
from torchaudio.utils import download_asset
from DatasetLoader import test_dataset_loader, practice_dataset_loader
from torch.utils.tensorboard import SummaryWriter



def phasor_to_complex(mag, phase):
            real = mag * torch.cos(phase)
            imag = mag * torch.sin(phase)
            complex_value = torch.complex(real, imag)
            return complex_value



def plot_waveform(waveform, sample_rate, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)   
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)



## spectrogram.shape: [B, F, T]
def istft(n_fft, win_length, hop_length, window_fn, normalized, spectrogram, length):
    
    return torchaudio.transforms.InverseSpectrogram(n_fft=n_fft,
                                                    win_length=win_length,
                                                    hop_length=hop_length,
                                                    window_fn=window_fn,
                                                    normalized=normalized)(spectrogram, length)
    
    

class WrappedModel(nn.Module):
    
    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU
    
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model
    
    def forward(self, x, label=None):
        return self.module(x, label)
    

class SpeakerNet(nn.Module):
    def __init__(self, model, optimizer, trainfunc, **kwargs):
        super(SpeakerNet, self).__init__()
        
        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__("MainModel")
        self.__S__ = SpeakerNetModel(**kwargs)
        
        LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        self.__L__ = LossFunction(**kwargs)
    
    def forward(self, data, label=None):
        
        data = data.cuda()
        outp = self.__S__.forward(data)
        
        if label == None:
            return outp
        
        else:
            nloss = self.__L__.forward(outp, label)
            return nloss

class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, **kwargs):
        self.__model__ = speaker_model
        
        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__("Optimizer")
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)
        
        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__("Scheduler")
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)
        
        self.gpu = gpu
        
        assert self.lr_step in ["epoch", "iteration"]
    
    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====
    
    def train_network(self, loader, verbose, writer, it):
        
        self.__model__.train()
        
        stepsize = loader.batch_size
        
        counter = 0
        index   = 0
        loss    = 0
        
        epochsize = len(loader)
        
        for clean_magnitude_spectrogram, noisy_magnitude_spectrogram, clean_phase_spectrogram, noisy_phase_spectrogram, clean_waveform, noisy_waveform in loader:
            
            self.__model__.zero_grad()
            
            noisy_magnitude_spectrogram = noisy_magnitude_spectrogram.cuda()
            clean_magnitude_spectrogram = clean_magnitude_spectrogram.cuda()
            
            # noisy, clean MUST BE IN ORDER!
            nloss = self.__model__(noisy_magnitude_spectrogram, clean_magnitude_spectrogram)
            
            if verbose:
                writer.add_scalar(tag="Loss/train",
                                scalar_value=nloss,
                                global_step=(it*epochsize)+(counter*stepsize))
            
            nloss.backward()
            self.__optimizer__.step()

            loss    += nloss.detach().cpu().item()
            counter += 1
            index   += stepsize

            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
                sys.stdout.write("Loss {:f}".format(loss / counter))
                sys.stdout.flush()
            
            if self.lr_step == "iteration":
                self.__scheduler__.step()
        
        if self.lr_step == "epoch":
            self.__scheduler__.step()
            
        return (loss / counter)
        
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## evaluate metrics
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateMetrics(self, batch_size, nDataLoaderThread, sample_rate,
                        n_fft, win_length, hop_length, test_path, distributed, **kwargs):
        
        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        
        self.__model__.eval()
        
        ## Define metrics
        stoi  = ShortTimeObjectiveIntelligibility(fs=sample_rate, extended=False)
        # pesq  = PerceptualEvaluationSpeechQuality(fs=sample_rate, mode='wb')
        pesq  = pesq_batch
        sisnr = ScaleInvariantSignalNoiseRatio()
        sisdr = ScaleInvariantSignalDistortionRatio(zero_mean=True)
        
        ## Reset constants
        STOI_SUM    = 0
        PESQ_SUM    = 0
        SISNR_SUM   = 0
        SISDR_SUM   = 0
        
        counter     = 0
        length      = sample_rate * 3
        
        ## Define test data loader
        test_dataset = test_dataset_loader(test_path=test_path,
                                            n_fft=n_fft,
                                            win_length=win_length,
                                            hop_length=hop_length,
                                            window_fn=torch.hann_window,
                                            power=None,
                                            sample_rate=sample_rate)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None
        
        
        # voxceleb trainer setting: batch_size=1, drop_last=False
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nDataLoaderThread,
                                                  drop_last=True,
                                                  sampler=sampler)
        
        for clean_magnitude_spectrogram, noisy_magnitude_spectrogram, clean_phase_spectrogram, noisy_phase_spectrogram, clean_waveform, noisy_waveform in test_loader:
            
            clean_magnitude_spectrogram = clean_magnitude_spectrogram.cuda()
            noisy_magnitude_spectrogram = noisy_magnitude_spectrogram.cuda()
            
            with torch.no_grad():
                denoised_magnitude_spectrogram = self.__model__(noisy_magnitude_spectrogram)
                denoised_phase_spectrogram = noisy_phase_spectrogram
        
            ## Phasor to complex
            clean_spectrogram    = phasor_to_complex(clean_magnitude_spectrogram.cpu().squeeze(),
                                                         clean_phase_spectrogram.cpu().squeeze())
            # noisy_spectrogram    = phasor_to_complex(noisy_magnitude_spectrogram.cpu().squeeze(),
            #                                              noisy_phase_spectrogram.cpu().squeeze())
            denoised_spectrogram = phasor_to_complex(denoised_magnitude_spectrogram.cpu().squeeze(),
                                                         denoised_phase_spectrogram.cpu().squeeze())
            
            ## Spectrogram to waveform
            clean_waveform      = istft(n_fft, win_length, hop_length, torch.hann_window, True,    clean_spectrogram, length)
            # noisy_waveform      = istft(n_fft, win_length, hop_length, torch.hann_window, True,    noisy_spectrogram, length)
            denoised_waveform   = istft(n_fft, win_length, hop_length, torch.hann_window, True, denoised_spectrogram, length)
            
            ## Debug
            DEBUG = False
            if DEBUG:
                os.makedirs("./debug", exist_ok=True)
                for i in range(len(PESQ_LIST)):
                    if not isinstance(PESQ_LIST[i], float):
                        torchaudio.save(   "./debug/debug_clean.wav",    clean_waveform[i].unsqueeze(0), sample_rate, format="wav")
                        torchaudio.save("./debug/debug_denoised.wav", denoised_waveform[i].unsqueeze(0), sample_rate, format="wav")
                        return
            
            STOI_SUM    +=  stoi(denoised_waveform, clean_waveform).item()
            PESQ_LIST    = pesq(sample_rate, clean_waveform.numpy(), denoised_waveform.numpy(), mode='wb')
            PESQ_LIST    = [elem for elem in PESQ_LIST if isinstance(elem, float)]
            PESQ_SUM    += sum(PESQ_LIST) / len(PESQ_LIST)
            SISNR_SUM   += sisnr(denoised_waveform, clean_waveform).item()
            SISDR_SUM   += sisdr(denoised_waveform, clean_waveform).item()
            counter     += 1
        
        STOI  =  STOI_SUM / counter
        PESQ  =  PESQ_SUM / counter
        SISNR = SISNR_SUM / counter
        SISDR = SISDR_SUM / counter
        
        return STOI, PESQ, SISNR, SISDR
    
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## inference one test sample
    ## ===== ===== ===== ===== ===== ===== ===== =====
    
    def inference(self, sample_rate, n_fft, win_length, hop_length, test_path, **kwargs):
        
        self.__model__.eval()
        
        ## Define test data loader
        test_dataset = test_dataset_loader(test_path=test_path,
                                            n_fft=n_fft,
                                            win_length=win_length,
                                            hop_length=hop_length,
                                            window_fn=torch.hann_window,
                                            power=None,
                                            sample_rate=sample_rate)
        
        clean_magnitude_spectrogram, noisy_magnitude_spectrogram, clean_phase_spectrogram, noisy_phase_spectrogram, clean_waveform, noisy_waveform = test_dataset[0]
        
        ## set batch_size as 1
        clean_magnitude_spectrogram = clean_magnitude_spectrogram.unsqueeze(0)
        noisy_magnitude_spectrogram = noisy_magnitude_spectrogram.unsqueeze(0)
        clean_phase_spectrogram = clean_phase_spectrogram.unsqueeze(0)
        noisy_phase_spectrogram = noisy_phase_spectrogram.unsqueeze(0)
        
        clean_magnitude_spectrogram = clean_magnitude_spectrogram.cuda()
        noisy_magnitude_spectrogram = noisy_magnitude_spectrogram.cuda()
        
        with torch.no_grad():
                denoised_magnitude_spectrogram = self.__model__(noisy_magnitude_spectrogram)
                denoised_phase_spectrogram = noisy_phase_spectrogram
        
        ## Phasor to complex
        clean_spectrogram    = phasor_to_complex(clean_magnitude_spectrogram.cpu().squeeze(),
                                                     clean_phase_spectrogram.cpu().squeeze())
        noisy_spectrogram    = phasor_to_complex(noisy_magnitude_spectrogram.cpu().squeeze(),
                                                     noisy_phase_spectrogram.cpu().squeeze())
        denoised_spectrogram = phasor_to_complex(denoised_magnitude_spectrogram.cpu().squeeze(),
                                                     denoised_phase_spectrogram.cpu().squeeze())
        
        ## Spectrogram to waveform
        clean_waveform      = istft(n_fft, win_length, hop_length, torch.hann_window, True,    clean_spectrogram, sample_rate*3)
        noisy_waveform      = istft(n_fft, win_length, hop_length, torch.hann_window, True,    noisy_spectrogram, sample_rate*3)
        denoised_waveform   = istft(n_fft, win_length, hop_length, torch.hann_window, True, denoised_spectrogram, sample_rate*3)
        
        os.makedirs("./test", exist_ok=True)
        torchaudio.save("./test/test_clean.wav", clean_waveform.unsqueeze(0), sample_rate, format="wav")
        torchaudio.save("./test/test_noisy.wav", noisy_waveform.unsqueeze(0), sample_rate, format="wav")
        torchaudio.save("./test/test_denoised.wav", denoised_waveform.unsqueeze(0), sample_rate, format="wav")
        
        return
    
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## inference practical audio
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def practice(self, nDataLoaderThread, n_fft, sample_rate, **kwargs):
        
        self.__model__.eval()
        
        practice_dataset = practice_dataset_loader(**kwargs)
        practice_loader = torch.utils.data.DataLoader(practice_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=nDataLoaderThread,
                                                      drop_last=True,
                                                      sampler=None)
        
        inverse_spectrogram = transforms.InverseSpectrogram(n_fft=n_fft)
        
        counter     = 0
        length      = 24000
        
        mel_scale = transforms.MelScale(n_mels=80, sample_rate=8000, n_stft=129)
        
        def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
            if ax is None:
                _, ax = plt.subplots(1, 1)
            if title is not None:
                ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest", cmap='inferno')
        
        for idx, (noisy_magnitude_spectrogram, noisy_phase_spectrogram) in enumerate(practice_loader):
            noisy_waveform = []
            denoised_waveform = []
            
            noisy_magnitude_spectrogram = noisy_magnitude_spectrogram.cuda() # [1, 129, 188]
            
            with torch.no_grad():
                denoised_magnitude_spectrogram  = self.__model__(noisy_magnitude_spectrogram)
                denoised_phase_spectrogram     = noisy_phase_spectrogram
            
            noisy_magnitude_spectrogram_np = noisy_magnitude_spectrogram.squeeze().cpu().numpy()
            denoised_magnitude_spectrogram_np = denoised_magnitude_spectrogram.squeeze().cpu().numpy()
            
            # import pdb; pdb.set_trace();
            # noisy_magnitude_melspectrogram = mel_scale(noisy_magnitude_spectrogram.squeeze().cpu())
            # noisy_magnitude_melspectrogram_np = amp_to_db(noisy_magnitude_melspectrogram).numpy()
            # denoised_magnitude_melspectrogram = mel_scale(denoised_magnitude_spectrogram.squeeze().cpu())
            # denoised_magnitude_melspectrogram_np = amp_to_db(denoised_magnitude_melspectrogram).numpy()
            
            # plt.figure()
            # plt.imshow((noisy_magnitude_melspectrogram_np), aspect='auto', origin='lower', cmap='inferno')
            # plt.colorbar(format='%+2.0f dB')
            # plt.title(f'noisy MelSpectrogram {idx + 1}')
            # noisy_image_path = os.path.join('melspectrogram', f'noisy_melspectrogram_{idx + 1}.png')
            # plt.savefig(noisy_image_path)
            # plt.close()

            # plt.figure()
            # plt.imshow((denoised_magnitude_melspectrogram_np), aspect='auto', origin='lower', cmap='inferno')
            # plt.colorbar(format='%+2.0f dB')
            # plt.title(f'denoisedd MelSpectrogram {idx + 1}')
            # denoised_image_path = os.path.join('melspectrogram', f'denoised_melspectrogram_{idx + 1}.png')
            # plt.savefig(denoised_image_path)
            # plt.close()
            
            noisy_spectrogram   = phasor_to_complex(noisy_magnitude_spectrogram.cpu().squeeze(),   noisy_phase_spectrogram.cpu().squeeze()).unsqueeze(0)
            denoised_spectrogram = phasor_to_complex(denoised_magnitude_spectrogram.cpu().squeeze(), denoised_phase_spectrogram.cpu().squeeze()).unsqueeze(0)
            
            ## Spectrogram to waveform
            for ii in range(noisy_spectrogram.size(0)):
                noisy_waveform.append(inverse_spectrogram(noisy_spectrogram[ii].unsqueeze(0),    length))
                denoised_waveform.append(inverse_spectrogram(denoised_spectrogram[ii].unsqueeze(0), length))
            
            noisy_waveform = torch.stack(noisy_waveform, dim=0)
            denoised_waveform = torch.stack(denoised_waveform, dim=0)
            
            for ii in range(1):
                
                image_number = counter + ii
                
                # draw all plots in practice dataset
                # _, axes = plt.subplots(3, 1, sharex=True, sharey=True)
                
                # plot_waveform(noisy_waveform[ii], sample_rate, title="noisy", ax=axes[0])
                # plot_waveform(clean_waveform[ii], sample_rate, title="clean", ax=axes[1])
                # plot_waveform(denoised_waveform[ii], sample_rate, title="denoised", ax=axes[2])
                
                # plt.savefig("waveform/waveform{}.jpg".format(image_number))
                # plt.clf()
                
                # _, axes = plt.subplots(2, 1, sharex=True, sharey=True)
                # plot_spectrogram(noisy_magnitude_spectrogram_np, title="noisy", ax=axes[0])
                # plot_spectrogram(denoised_magnitude_spectrogram_np, title="denoised", ax=axes[1])
                # plt.savefig("spectrogram/spectrogram{}.jpg".format(image_number))
                # plt.clf()
                
                os.makedirs("audio/practice_noisy_audio/", exist_ok=True)
                os.makedirs("audio/practice_denoised_audio/", exist_ok=True)
                torchaudio.save("audio/practice_noisy_audio/audio{}.wav".format(image_number), noisy_waveform[ii], sample_rate)
                torchaudio.save("audio/practice_denoised_audio/audio{}.wav".format(image_number), denoised_waveform[ii], sample_rate)
                
            counter += 1
        
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        
        torch.save(self.__model__.module.state_dict(), path)
    
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====
    
    def loadParameters(self, path):
        
        self_state = self.__model__.module.state_dict()                     
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            
            # From here ~
            for name, param in loaded_state.items(): # [(key1, value1), (key2, value2), ...]
                new_name = "__S__." + name
                newdict[new_name] = param
                delete_list.append(name)
            
            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]
            # ~ to here, add prefix '__S__' to every keys in loaded_state __S__
            
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                # Remove prefix module from name
                name = name.replace("module.", "")
                
                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue
            
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            
            self_state[name].copy_(param)
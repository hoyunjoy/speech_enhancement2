import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
import sys, random
import time, itertools, importlib
import matplotlib.pyplot as plt
import librosa

from compute_metrics import compute_metrics
from pesq import pesq_batch
from models.discriminator import batch_pesq
from torchmetrics.audio import ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality, ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio
from torchaudio.utils import download_asset
from DatasetLoader2 import mag_pha_stft, mag_pha_istft
from torch.utils.tensorboard import SummaryWriter
from models.generator import phase_losses, pesq_score


def phasor_to_complex(mag, phase):
            real = mag * torch.cos(phase)
            imag = mag * torch.sin(phase)
            complex_value = torch.complex(real, imag)
            return complex_value

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
    
    # def forward(self, x, label=None):
    #     return self.module(x, label)
    def forward(self, x):
        return self.module(x)
    

class SpeakerNet(nn.Module):
    def __init__(self, model, **kwargs):
        super(SpeakerNet, self).__init__()
        
        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__("MainModel")
        self.__S__ = SpeakerNetModel(**kwargs)
    
    def forward(self, data):
        
        outp = self.__S__.forward(data)
        return outp

class ModelTrainer(object):
    def __init__(self,
                 speaker_model_1,
                 speaker_model_2,
                 optimizer_1,
                 optimizer_2,
                 scheduler_1,
                 scheduler_2,
                 trainfunc_1,
                 trainfunc_2,
                 gpu,
                 **kwargs):
        
        self.__model_1__ = speaker_model_1
        self.__model_2__ = speaker_model_2
        
        Optimizer_1 = importlib.import_module("optimizer." + optimizer_1).__getattribute__("Optimizer")
        self.__optimizer_1__ = Optimizer_1(self.__model_1__.parameters(), **kwargs)
        
        Optimizer_2 = importlib.import_module("optimizer." + optimizer_2).__getattribute__("Optimizer")
        self.__optimizer_2__ = Optimizer_2(self.__model_2__.parameters(), **kwargs)
        
        Scheduler_1 = importlib.import_module("scheduler." + scheduler_1).__getattribute__("Scheduler")
        self.__scheduler_1__, self.lr_step_1 = Scheduler_1(self.__optimizer_1__, **kwargs)
        
        Scheduler_2 = importlib.import_module("scheduler." + scheduler_2).__getattribute__("Scheduler")
        self.__scheduler_2__, self.lr_step_2 = Scheduler_2(self.__optimizer_2__, **kwargs)
        
        LossFunction_1 = importlib.import_module("loss." + trainfunc_1).__getattribute__("LossFunction")
        self.__lossfunction_1__ = LossFunction_1(**kwargs)
        
        LossFunction_2 = importlib.import_module("loss." + trainfunc_2).__getattribute__("LossFunction")
        self.__lossfunction_2__ = LossFunction_2(**kwargs)
        
        self.gpu = gpu
        
        assert self.lr_step_1 in ["epoch", "iteration"]
        assert self.lr_step_2 in ["epoch", "iteration"]
    
    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====
    
    def train_network(self,
                      loader,
                      verbose,
                      writer,
                      it,
                      n_fft,
                      hop_length,
                      win_length,
                      compress_factor,
                      **kwargs):
        
        self.__model_1__.train()
        self.__model_2__.train()
        
        steps = (it-1) * loader.__len__() # because it starts from 1
        
        for counter, (clean_wav, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha) in enumerate(loader):
            
            ## Move to GPU
            clean_wav = clean_wav.cuda()
            clean_mag = clean_mag.cuda()
            clean_pha = clean_pha.cuda()
            clean_com = clean_com.cuda()
            noisy_mag = noisy_mag.cuda()
            noisy_pha = noisy_pha.cuda()
            one_labels = torch.ones(loader.batch_size).cuda()
            
            ## Generate enhanced output
            enhan_mag, enhan_pha, enhan_com = self.__model_1__((noisy_mag, noisy_pha))
            enhan_wav = mag_pha_istft(enhan_mag,
                                         enhan_pha,
                                         n_fft,
                                         hop_length,
                                         win_length,
                                         compress_factor)
            clean_wav_list    = list(clean_wav.cpu().numpy())
            enhan_wav_list = list(enhan_wav.detach().cpu().numpy())
            batch_pesq_score  = batch_pesq(clean_wav_list, enhan_wav_list) # PESQ of enhanced output
            
            ##----------------------------------------------------------------##
            ## Discriminator
            self.__optimizer_2__.zero_grad()
            
            ## Metric of real data and generated data, respectively
            metric_r = self.__model_2__((clean_mag, clean_mag))
            metric_g = self.__model_2__((clean_mag, enhan_mag.detach()))
            
            ## Discriminator loss
            loss_disc_r = self.__lossfunction_2__(metric_r.flatten(), one_labels) # Order is not important because it's MSELoss
            if batch_pesq_score is not None:
                loss_disc_g = self.__lossfunction_2__(metric_g.flatten(), batch_pesq_score.cuda()) # Order is not important because it's MSELoss
            else:
                print('pesq is None')
                loss_disc_g = 0
            loss_disc_all = loss_disc_r + loss_disc_g
            
            loss_disc_all.backward()
            self.__optimizer_2__.step()
            ##----------------------------------------------------------------##



            ##----------------------------------------------------------------##
            ## Generator
            self.__optimizer_1__.zero_grad()
            
            ## Generator loss
            metric_g = self.__model_2__((clean_mag, enhan_mag)) # This should be included because of backward
            loss_gen_all, losses = self.__lossfunction_1__((enhan_wav, enhan_mag, enhan_pha, enhan_com, metric_g),
                                                   (clean_wav, clean_mag, clean_pha, clean_com, one_labels))
            
            loss_gen_all.backward()
            self.__optimizer_1__.step()
            ##----------------------------------------------------------------##
            
            
            
            ##----------------------------------------------------------------##            
            if verbose:
                writer.add_scalar(tag="Training/Generator Loss",
                                  scalar_value=loss_gen_all,
                                  global_step=steps)
                writer.add_scalar(tag="Training/Magnitude Loss",
                                  scalar_value=losses[0],
                                  global_step=steps)
                writer.add_scalar(tag="Training/Phase Loss",
                                  scalar_value=losses[1],
                                  global_step=steps)
                writer.add_scalar(tag="Training/Complex Loss",
                                  scalar_value=losses[2],
                                  global_step=steps)
                writer.add_scalar(tag="Training/Metric Loss",
                                  scalar_value=losses[3],
                                  global_step=steps)
                writer.add_scalar(tag="Training/Time Loss",
                                  scalar_value=losses[4],
                                  global_step=steps)
                writer.add_scalar(tag="Training/Discriminator Loss",
                                  scalar_value=loss_disc_all,
                                  global_step=steps)
                sys.stdout.write("\rEpoch {:d} Processing {:d} of {:d}: ".format(it, (counter+1), loader.__len__()))
                sys.stdout.flush()
            ##----------------------------------------------------------------##
            steps += 1
            
            if self.lr_step_1 == "iteration":
                self.__scheduler_1__.step()
            
            if self.lr_step_2 == "iteration":
                self.__scheduler_2__.step()
        
        if self.lr_step_1 == "epoch":
            self.__scheduler_1__.step()
        
        if self.lr_step_2 == "epoch":
            self.__scheduler_2__.step()
            
        return loss_gen_all, loss_disc_all
        
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Validate network
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def valid_network(self,
                      verbose,
                      writer,
                      it,
                      loader,
                      sample_rate,
                      n_fft,
                      win_length,
                      hop_length,
                      compress_factor,
                      **kwargs):
        
        self.__model_1__.eval() # Generator
        
        torch.cuda.empty_cache()
        wavs_r, wavs_g = [], []
        val_mag_err_tot = 0
        val_pha_err_tot = 0
        val_com_err_tot = 0
        
        with torch.no_grad():
            for counter, (clean_wav, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha) in enumerate(loader):
                clean_wav = clean_wav.cuda()
                clean_mag = clean_mag.cuda()
                clean_pha = clean_pha.cuda()
                clean_com = clean_com.cuda()
                
                enhan_mag, enhan_pha, enhan_com = self.__model_1__((noisy_mag.cuda(), noisy_pha.cuda()))
                enhan_wav = mag_pha_istft(enhan_mag,
                                         enhan_pha,
                                         n_fft,
                                         hop_length,
                                         win_length,
                                         compress_factor)
                wavs_r += torch.split(clean_wav, 1, dim=0) # [B, T] -> B개의 [1, T] 모양 tensor가 들어있는 tuple 
                wavs_g += torch.split(enhan_wav, 1, dim=0)
                
                val_mag_err_tot += F.mse_loss(clean_mag, enhan_mag).item()
                val_ip_err, val_gd_err, val_iaf_err = phase_losses(clean_pha, enhan_pha, n_fft)
                val_pha_err_tot += (val_ip_err + val_gd_err + val_iaf_err).item()
                val_com_err_tot += F.mse_loss(clean_com, enhan_com).item()
            
            val_mag_err = val_mag_err_tot / (counter+1) # because counter starts from 0
            val_pha_err = val_pha_err_tot / (counter+1)
            val_com_err = val_com_err_tot / (counter+1)
            val_pesq_score = pesq_score(wavs_r, wavs_g, sample_rate).item()
            
            print('Epoch : {:d}, PESQ Score: {:4.3f}'.format(it, val_pesq_score))
            writer.add_scalar(tag="Validation/PESQ Score",
                              scalar_value=val_pesq_score,
                              global_step=it)
            writer.add_scalar(tag="Validation/Magnitude Loss",
                              scalar_value=val_mag_err,
                              global_step=it)
            writer.add_scalar(tag="Validation/Phase Loss",
                              scalar_value=val_pha_err,
                              global_step=it)
            writer.add_scalar(tag="Validation/Complex Loss",
                              scalar_value=val_com_err,
                              global_step=it)
        
            
        
        # if distributed:
        #     rank = torch.distributed.get_rank()
        # else:
        #     rank = 0
        
        # self.__model_1__.eval()
        
        # ## Define metrics
        # stoi  = ShortTimeObjectiveIntelligibility(fs=sample_rate, extended=False)
        # # pesq  = PerceptualEvaluationSpeechQuality(fs=sample_rate, mode='wb')
        # pesq  = pesq_batch
        # sisnr = ScaleInvariantSignalNoiseRatio()
        # sisdr = ScaleInvariantSignalDistortionRatio(zero_mean=True)
        
        # ## Initialize constants
        # STOI_SUM    = 0
        # PESQ_SUM    = 0
        # SISNR_SUM   = 0
        # SISDR_SUM   = 0
        
        # ## Define test data loader
        # test_dataset = test_dataset_loader(test_path=test_path,
        #                                     n_fft=n_fft,
        #                                     win_length=win_length,
        #                                     hop_length=hop_length,
        #                                     window_fn=torch.hann_window,
        #                                     power=None,
        #                                     sample_rate=sample_rate,
        #                                     compress_factor=compress_factor)

        # if distributed:
        #     sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        # else:
        #     sampler = None
        
        
        # # Voxceleb trainer setting: batch_size=1, drop_last=False
        # test_loader = torch.utils.data.DataLoader(test_dataset,
        #                                           batch_size=batch_size*4,
        #                                           shuffle=False,
        #                                           num_workers=nDataLoaderThread,
        #                                           drop_last=True,
        #                                           sampler=sampler)
        
        # stepsize = test_loader.batch_size
        # index   = 0
        
        # for counter, (clean_wav, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha) in enumerate(test_loader):
            
        #     clean_wav = clean_wav.cuda() # [2, 32000]
        #     clean_mag = clean_mag.cuda() # [2, 201, 321]
        #     clean_pha = clean_pha.cuda() # [2, 201, 321]
        #     clean_com = clean_com.cuda() # [2, 201, 321, 2]
        #     noisy_mag = noisy_mag.cuda() # [2, 201, 321]
        #     noisy_pha = noisy_pha.cuda() # [2, 201, 321]
            
        #     with torch.no_grad():
        #         enhan_mag, enhan_pha, enhan_com = self.__model_1__((noisy_mag, noisy_pha))
            
        #     ## Get clean wav
        #     clean_wav    = clean_wav.squeeze(1).cpu() # [2, 32000]
            
        #     ## Get noisy wav
        #     noisy_wav = mag_pha_istft(noisy_mag.squeeze(1),
        #                               noisy_pha.squeeze(1),
        #                               n_fft,
        #                               hop_length,
        #                               win_length,
        #                               compress_factor)
        #     noisy_wav = noisy_wav.cpu() # [2, 32000]
            
        #     ## Get enhan wav
        #     enhan_wav = mag_pha_istft(enhan_mag.squeeze(1),
        #                               enhan_pha.squeeze(1),
        #                               n_fft,
        #                               hop_length,
        #                               win_length,
        #                               compress_factor)
        #     enhan_wav = enhan_wav.cpu() # [2, 32000]
            
        #     STOI_SUM    += stoi(enhan_wav, clean_wav).item()
        #     PESQ_LIST    = pesq(sample_rate, clean_wav.numpy(), enhan_wav.numpy(), mode='wb')
            
        #     ## Debug
        #     DEBUG = False
        #     if DEBUG:
        #         os.makedirs("./debug", exist_ok=True)
        #         for i in range(len(PESQ_LIST)):
        #             ## If there are abnormal pesq values in list, save them.
        #             if not isinstance(PESQ_LIST[i], float):
        #                 torchaudio.save("./debug/clean.wav", clean_wav[i].unsqueeze(0), sample_rate, format="wav")
        #                 torchaudio.save("./debug/enahn.wav", enhan_wav[i].unsqueeze(0), sample_rate, format="wav")
        #                 return
            
        #     ## Remove 'NoUtterancesErrors'
        #     PESQ_LIST    = [elem for elem in PESQ_LIST if isinstance(elem, float)]
        #     PESQ_SUM    += sum(PESQ_LIST) / len(PESQ_LIST)
        #     SISNR_SUM   += sisnr(enhan_wav, clean_wav).item()
        #     SISDR_SUM   += sisdr(enhan_wav, clean_wav).item()
            
        #     index   += stepsize
            
        #     if verbose:
        #         sys.stdout.write("\rProcessing {:d} of {:d}: ".format(index, test_loader.__len__() * test_loader.batch_size))
        
        # STOI  =  STOI_SUM / counter
        # PESQ  =  PESQ_SUM / counter
        # SISNR = SISNR_SUM / counter
        # SISDR = SISDR_SUM / counter
        
        # return STOI, PESQ, SISNR, SISDR
    
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Inference one test sample
    ## ===== ===== ===== ===== ===== ===== ===== =====
    
    def inference(self, loader, sample_rate, n_fft, win_length, hop_length, compress_factor, **kwargs):
        
        self.__model_1__.eval()
        
        loader_iter = iter(loader)
        batch = next(loader_iter)
        clean_wav, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = next(loader_iter) # To select the second batch
        
        clean_wav = clean_wav.cuda()
        clean_mag = clean_mag.cuda()
        clean_pha = clean_pha.cuda()
        clean_com = clean_com.cuda()
        noisy_mag = noisy_mag.cuda()
        noisy_pha = noisy_pha.cuda()
        
        with torch.no_grad():
            enhan_mag, enhan_pha, enhan_com = self.__model_1__((noisy_mag, noisy_pha))
        
        clean_wav = clean_wav.squeeze(1).cpu() # [1, 32000]
        clean_mag = clean_mag.squeeze(1).cpu() # [1, 201, 321]
        clean_pha = clean_pha.squeeze(1).cpu() # [1, 201, 321]
        clean_com = clean_com.squeeze(1).cpu() # [1, 201, 321, 2]
        noisy_mag = noisy_mag.squeeze(1).cpu() # [1, 201, 321]
        noisy_pha = noisy_pha.squeeze(1).cpu() # [1, 201, 321]
        enhan_mag = enhan_mag.squeeze(1).cpu() # [1, 201, 321]
        enhan_pha = enhan_pha.squeeze(1).cpu() # [1, 201, 321]
        
        ## Get clean wav
        clean_wav    = clean_wav.squeeze(1).cpu() # [1, 32000]
           
        ## Get noisy wav
        noisy_wav = mag_pha_istft(noisy_mag, noisy_pha, n_fft, hop_length, win_length, compress_factor)
        noisy_wav = noisy_wav.squeeze(1).cpu() # [1, 32000]
        
        ## Get enhan wav
        enhan_wav = mag_pha_istft(enhan_mag, enhan_pha, n_fft, hop_length, win_length, compress_factor)
        enhan_wav = enhan_wav.cpu() # [1, 32000]
        
        ## Save three waveforms
        dirname = "./test7"
        os.makedirs(dirname, exist_ok=True)
        torchaudio.save(dirname + "/clean.wav", clean_wav, sample_rate, format="wav")
        torchaudio.save(dirname + "/noisy.wav", noisy_wav, sample_rate, format="wav")
        torchaudio.save(dirname + "/enhan.wav", enhan_wav, sample_rate, format="wav")
        
        return
    
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate metrics
    ## ===== ===== ===== ===== ===== ===== ===== =====
    
    def evaluateMetrics(self, loader, sample_rate, n_fft, win_length, hop_length, compress_factor, **kwargs):
        
        metrics_total = np.zeros(6)
        num = len(loader)
        
        with torch.no_grad():
            for counter, (clean_wav, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha) in enumerate(loader):
                
                enhan_mag, enhan_pha, enhan_com = self.__model_1__((noisy_mag.cuda(), noisy_pha.cuda()))
                enhan_wav = mag_pha_istft(enhan_mag,
                                         enhan_pha,
                                         n_fft,
                                         hop_length,
                                         win_length,
                                         compress_factor)
                
                metrics = compute_metrics(clean_wav.squeeze(0).numpy(),
                                          enhan_wav.cpu().squeeze(0).numpy(),
                                          sample_rate,
                                          path=0)
                metrics = np.array(metrics)
                metrics_total += metrics
        
        metrics_avg = metrics_total / num
        print('pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2], 
              'covl: ', metrics_avg[3], 'ssnr: ', metrics_avg[4], 'stoi: ', metrics_avg[5])
        
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path, model):
        
        if model == "model_1":
            torch.save(self.__model_1__.module.state_dict(), path)
        elif model == "model_2":
            torch.save(self.__model_2__.module.state_dict(), path)
        else:
            ValueError(f"Unknown model {model}")
        
        return
    
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====
    
    def loadParameters(self, path, model):
        
        if model == "model_1":
            self_state = self.__model_1__.module.state_dict()
        elif model == "model_2":
            self_state = self.__model_2__.module.state_dict()
        else:
            ValueError(f"Unknown model {model}")
        
        # self_state = self.__model__.module.state_dict()                     
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
        
        return
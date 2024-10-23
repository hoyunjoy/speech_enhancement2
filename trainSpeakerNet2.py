import sys, time, os, argparse
import yaml
import numpy
import torch
import glob
import zipfile
import warnings
import datetime
from SpeakerNet2 import *
from DatasetLoader2 import *
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
import torch.distributed as dist
import torch.multiprocessing as mp
warnings.simplefilter("ignore")

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "SpeakerNet")

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')

## Data lodaer
parser.add_argument('--batch_size',        type=int,    default=100,     help='Batch size, defined as the number of classes per batch')
parser.add_argument('--nDataLoaderThread', type=int,    default=5,      help='Number of loader threads')
parser.add_argument('--seed',              type=int,    default=10,     help='Seed for the random number generator')
parser.add_argument('--sample_rate',       type=int,    default=16000,   help='Sample rate of the audio')

## STFT
parser.add_argument('--n_fft',          type=int,     default=400,         help='n_fft')
parser.add_argument('--win_length',     type=int,     default=400,         help='win_length')
parser.add_argument('--hop_length',     type=int,     default=100,         help='hop_length')

## Training details
parser.add_argument('--test_interval',  type=int,     default=5,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,     default=100,   help='Maximum number of epochs')
parser.add_argument('--trainfunc_1',      type=str,     default="",    help='Loss function to use')
parser.add_argument('--trainfunc_2',      type=str,     default="",    help='Loss function to use')

## Optimizer
parser.add_argument('--optimizer_1',      type=str,     default="adam",   help='Optimizer')
parser.add_argument('--optimizer_2',      type=str,     default="adam",   help='Optimizer')
parser.add_argument('--adam_b1',          type=float,   default=0.8,      help='adam_b1')
parser.add_argument('--adam_b2',          type=float,   default=0.99,     help='adam_b2')
parser.add_argument('--scheduler_1',      type=str,     default="steplr", help='Learning rate scheduler')
parser.add_argument('--scheduler_2',      type=str,     default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',               type=float,   default=0.001,    help='Initial learning rate')
parser.add_argument('--lr_decay',         type=float,   default=0.90,     help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',     type=float,   default=0.01,     help='Weight decay in the optimizer')

## Load and save
parser.add_argument('--initial_model_1',  type=str,     default="", help='Initial model weights, otherwise initialise with random weights')
parser.add_argument('--initial_model_2',  type=str,     default="", help='Initial model weights, otherwise initialise with random weights')
parser.add_argument('--save_path',        type=str,     default="", help='Path for model and logs')

## Training and evaluation data
parser.add_argument('--dataset',             type=str,     default="/mnt/lynx4/datasets/VOICE_DEMAND")
parser.add_argument('--train_clean_dir',     type=str,     default="/mnt/lynx4/datasets/VOICE_DEMAND/clean_trainset_28spk_wav_16k", help='Absolute path to the clean train set')
parser.add_argument('--train_noisy_dir',     type=str,     default="/mnt/lynx4/datasets/VOICE_DEMAND/noisy_trainset_28spk_wav_16k", help='Absolute path to the noisy train set')
parser.add_argument('--valid_clean_dir',     type=str,     default="/mnt/lynx4/datasets/VOICE_DEMAND/clean_testset_wav_16k",        help='Absolute path to the clean valid set')
parser.add_argument('--valid_noisy_dir',     type=str,     default="/mnt/lynx4/datasets/VOICE_DEMAND/noisy_testset_wav_16k",        help='Absolute path to the noisy valid set')

## Model
parser.add_argument('--model_1',          type=str,     default="",          help='Name of model definition')
parser.add_argument('--model_2',          type=str,     default="",          help='Name of model definition')

## For test only
parser.add_argument('--eval',           dest='eval',  action='store_true',   help='Eval only')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')

## MP-SENet
parser.add_argument('--dense_channel',      type=int,       default=64,     help='dense_channel')
parser.add_argument('--compress_factor',    type=float,     default=0.3,    help='compress_factor')
parser.add_argument('--num_tsconformers',   type=int,       default=4,      help='num_tsconformers')
parser.add_argument('--beta',               type=float,     default=2.0,    help='beta')

args = parser.parse_args()

## Find option type
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
            return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    
    for k, v in yaml_config.items():    # k: option name, v: option value
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)   # string to specific type that was found
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):
    
    args.gpu = gpu
    
    ## Load models
    model_1 = SpeakerNet(args.model_1, **vars(args)) # Generator
    model_2 = SpeakerNet(args.model_2, **vars(args)) # Discriminator
    
    if args.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        
        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        model_1.cuda(args.gpu)
        model_2.cuda(args.gpu)
        
        model_1 = torch.nn.parallel.DistributedDataParallel(model_1, device_ids=[args.gpu], find_unused_parameters=False)
        model_2 = torch.nn.parallel.DistributedDataParallel(model_2, device_ids=[args.gpu], find_unused_parameters=False)
        
        print('Loaded the model on GPU {:d}'.format(args.gpu))
    
    else:
        model_1 = WrappedModel(model_1).cuda(args.gpu)
        model_2 = WrappedModel(model_2).cuda(args.gpu)
    
    it_1 = 1
    it_2 = 1
    
    try:
        ## Write args to scorefile
        if args.gpu == 0:
            scorefile   = open(args.result_save_path + "/scores.txt", "a+")
            
        ## Initialize trainer and data loader
        train_dataset    = dataset_loader(clean_dir=args.train_clean_dir,
                                          noisy_dir=args.train_noisy_dir,
                                          n_fft=args.n_fft,
                                          win_length=args.win_length,
                                          hop_length=args.hop_length,
                                          window_fn=torch.hann_window,
                                          power=None,
                                          sample_rate=args.sample_rate,
                                          compress_factor=args.compress_factor)
        
        if args.distributed:
            train_sampler = DistributedSampler(train_dataset, seed=args.seed)
        else:
            generator       = torch.Generator().manual_seed(args.seed)
            train_sampler   = RandomSampler(train_dataset, generator=generator)
        
        train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=args.batch_size,
                        num_workers=args.nDataLoaderThread,
                        sampler=train_sampler,
                        pin_memory=True,
                        worker_init_fn=worker_init_fn,
                        drop_last=True)
        
        if args.gpu == 0:
            valid_dataset = dataset_loader(clean_dir=args.valid_clean_dir,
                                          noisy_dir=args.valid_noisy_dir,
                                          n_fft=args.n_fft,
                                          win_length=args.win_length,
                                          hop_length=args.hop_length,
                                          window_fn=torch.hann_window,
                                          power=None,
                                          sample_rate=args.sample_rate,
                                          compress_factor=args.compress_factor)
            
            valid_loader = torch.utils.data.DataLoader(
                            valid_dataset,
                            batch_size=1,
                            num_workers=1,
                            sampler=None,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn,
                            drop_last=True)
        
        trainer = ModelTrainer(model_1, model_2, **vars(args))
        
        ## Load model weights
        modelfiles_1 = glob.glob('%s/model_1/model0*.model'%args.model_save_path)
        modelfiles_2 = glob.glob('%s/model_2/model0*.model'%args.model_save_path)
        modelfiles_1.sort()
        modelfiles_2.sort()
        
        ## Load model_1 weights
        if(args.initial_model_1 != ""):
            trainer.loadParameters(args.initial_model_1, model="model_1")
            print("Model_1 {} loaded!".format(args.initial_model_1))
        elif len(modelfiles_1) >= 1:
            trainer.loadParameters(modelfiles_1[-1], model="model_1")
            print("Model_1 {} loaded from previous state!".format(modelfiles_1[-1]))
            it_1 = int(os.path.splitext(os.path.basename(modelfiles_1[-1]))[0][5:]) + 1
        
        ## Load model_2 weights
        if(args.initial_model_2 != ""):
            trainer.loadParameters(args.initial_model_2, model="model_2")
            print("Model_2 {} loaded!".format(args.initial_model_2))
        elif len(modelfiles_2) >= 1:
            trainer.loadParameters(modelfiles_2[-1], model="model_2")
            print("Model_2 {} loaded from previous state!".format(modelfiles_2[-1]))
            it_2 = int(os.path.splitext(os.path.basename(modelfiles_2[-1]))[0][5:]) + 1
        
        assert it_1 == it_2
        it = it_1
        
        ## Update scheduler state
        for ii in range(1, it):
            trainer.__scheduler_1__.step()
            trainer.__scheduler_2__.step()
        
        ## Evaluation code
        if args.eval == True:
            
            if args.gpu == 0:
                ## num of model_1 + num of model_2
                pytorch_total_params = sum(p.numel() for p in model_1.module.__S__.parameters()) + sum(p.numel() for p in model_2.module.__S__.parameters())
                
                print('Total parameters: ', pytorch_total_params)
                print('Test path: ', args.dataset)
            
            trainer.evaluateMetrics(valid_loader, **vars(args))
            trainer.inference(valid_loader, **vars(args))
            return
        
        ## Save training code and params
        if args.gpu == 0:
            pyfiles = glob.glob('./*.py')
            strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            
            zipf = zipfile.ZipFile(args.result_save_path+'/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
            for file in pyfiles:
                zipf.write(file)
            zipf.close()
            
            with open(args.result_save_path + '/run%s.cmd'%args.save_path.split(sep="/")[-1], 'w') as f:
                f.write('%s'%args)
                
        ## Writer for tensorboard
        writer = None
        if args.gpu == 0:
            writer = SummaryWriter('runs/%s'%args.save_path.split(sep="/")[-1])
        
        ## Core training script
        try:
            for it in range(it, args.max_epoch + 1):
                
                if args.distributed:
                    train_sampler.set_epoch(it)
                    
                clr_1 = [x['lr'] for x in trainer.__optimizer_1__.param_groups]
                clr_2 = [x['lr'] for x in trainer.__optimizer_2__.param_groups]
                
                train_loss_1, train_loss_2 = trainer.train_network(loader=train_loader,
                                                verbose=(args.gpu == 0),
                                                writer=writer,
                                                it=it,
                                                **vars(args))
                
                if args.gpu == 0:
                    
                    print('\n', time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, Gen Loss {:f}, Disc Loss {:f}, Gen LR {:f}, Disc LR {:f}".format(it, train_loss_1, train_loss_2, max(clr_1), max(clr_2)))
                    scorefile.write("Epoch {:d}, Gen Loss {:f}, Disc Loss {:f}, Gen LR {:f}, Disc LR {:f} \n".format(it, train_loss_1, train_loss_2, max(clr_1), max(clr_2)))
                
                if (it % args.test_interval == 0) or True: # set True when debugging
                    if args.gpu == 0:
                        trainer.saveParameters(args.model_save_path + "/model_1/model%09d.model"%it, model="model_1")
                        trainer.saveParameters(args.model_save_path + "/model_2/model%09d.model"%it, model="model_2")
                        trainer.valid_network((args.gpu == 0), writer, it, valid_loader, **vars(args))
                        # trainer.inference(valid_loader, **vars(args))
                        
                        scorefile.flush()
        
        finally:
            if args.gpu == 0:
                writer.close()

    finally:
        if args.gpu == 0:
            scorefile.close()

## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    args.model_save_path    = args.save_path + "/model"
    args.result_save_path   = args.save_path + "/result"
    
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.model_save_path + "/model_1", exist_ok=True)
    os.makedirs(args.model_save_path + "/model_2", exist_ok=True)
    os.makedirs(args.model_save_path + "/metrics", exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)
    
    n_gpus = torch.cuda.device_count()
    
    print('Python Version:', sys.version)
    print('Pytorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    args.batch_size = int(args.batch_size / n_gpus)
    print('Batch size per GPU:', args.batch_size)
    print('Save path:', args.save_path)
    
    
    
    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)
    
if __name__ == '__main__':
    main()
import sys, time, os, argparse
import yaml
import numpy
import torch
import glob
import zipfile
import warnings
import datetime
# from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
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
parser.add_argument('--trainfunc',      type=str,     default="",    help='Loss function to use')

## Optimizer
parser.add_argument('--optimizer',      type=str,     default="adam",   help='Optimizer')
parser.add_argument('--scheduler',      type=str,     default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float,   default=0.001,    help='Initial learning rate')
parser.add_argument('--lr_decay',       type=float,   default=0.90,     help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float,   default=0,        help='Weight decay in the optimizer')

## Load and save
parser.add_argument('--initial_model',  type=str,     default="", help='Initial model weights, otherwise initialise with random weights')
parser.add_argument('--save_path',      type=str,     default="", help='Path for model and logs')

## Training and evaluation data
parser.add_argument('--train_path',     type=str,     default="/mnt/lynx4/datasets/VOICE_DEMAND", help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,     default="/mnt/lynx4/datasets/VOICE_DEMAND", help='Absolute path to the test set')

## Model
parser.add_argument('--model',          type=str,     default="",          help='Name of model definition')

## For test only
parser.add_argument('--eval',           dest='eval',  action='store_true',   help='Eval only')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')

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
    s = SpeakerNet(**vars(args))
    
    if args.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        
        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)
        
        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=False)
        
        print('Loaded the model on GPU {:d}'.format(args.gpu))
    
    else:
        s = WrappedModel(s).cuda(args.gpu)
    
    it = 1
    
    try:
        ## Write args to scorefile
        if args.gpu == 0:
            scorefile   = open(args.result_save_path + "/scores.txt", "a+")
            
        ## Initialize trainer and data loader
        # practice_dataset = practice_dataset_loader(**vars(args))
        train_dataset    = train_dataset_loader(train_path=args.train_path,
                                                n_fft=args.n_fft,
                                                win_length=args.win_length,
                                                hop_length=args.hop_length,
                                                window_fn=torch.hann_window,
                                                power=None,
                                                sample_rate=args.sample_rate)
        
        test_dataset     = test_dataset_loader(test_path=args.test_path,
                                                n_fft=args.n_fft,
                                                win_length=args.win_length,
                                                hop_length=args.hop_length,
                                                window_fn=torch.hann_window,
                                                power=None,
                                                sample_rate=args.sample_rate)
        
        if args.distributed:
            train_sampler = DistributedSampler(train_dataset, seed=args.seed)
            test_sampler  = DistributedSampler(test_dataset, seed=args.seed)

        else:
            generator = torch.Generator().manual_seed(args.seed)
            train_sampler     = RandomSampler(train_dataset, generator=generator)
            test_sampler      = RandomSampler(test_dataset, generator=generator)
            # practice_sampler  = RandomSampler(practice_dataset, generator=generator)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.nDataLoaderThread,
            sampler=train_sampler,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )
        
        test_loader  = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.nDataLoaderThread,
            sampler=test_sampler,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )
        
        # practice_loader = torch.utils.data.DataLoader(
        #     practice_dataset,
        #     batch_size=1,
        #     num_workers=args.nDataLoaderThread,
        #     sampler=practice_sampler,
        #     pin_memory=False,
        #     worker_init_fn=worker_init_fn,
        #     drop_last=False,
        # )
        
        trainer = ModelTrainer(s, **vars(args))
        
        ## Load model weights
        modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
        modelfiles.sort()
        
        if(args.initial_model != ""):
            trainer.loadParameters(args.initial_model)
            print("Model {} loaded!".format(args.initial_model))
        elif len(modelfiles) >= 1:
            trainer.loadParameters(modelfiles[-1])
            print("Model {} loaded from previous state!".format(modelfiles[-1]))
            it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
        
        for ii in range(1, it):
            trainer.__scheduler__.step()
        
        ## Evaluation code - must run on single GPU
        if args.eval == True:
            
            if args.gpu == 0:
                pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())
                
                print('Total parameters: ', pytorch_total_params)
                print('Test path', args.test_path)
            
            STOI, PESQ, SISNR, SISDR = trainer.evaluateMetrics(**vars(args))
            
            if args.gpu == 0:
                print("STOI {:2.4f}, PSEQ {:2.4f}, SI-SNR {:2.4f}, SI-SDR {:2.4f}".format(STOI, PESQ, SISNR, SISDR))
                trainer.inference(**vars(args))
                # trainer.practice(**vars(args))
            
            return
        
        ## Save training code and params
        if args.gpu == 0:
            pyfiles = glob.glob('./*.py')
            strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            
            zipf = zipfile.ZipFile(args.result_save_path+'/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
            for file in pyfiles:
                zipf.write(file)
            zipf.close()
            
            with open(args.result_save_path + '/run%s.cmd'%strtime, 'w') as f:
                f.write('%s'%args)
                
        ## Writer for tensorboard
        writer = None
        if args.gpu == 0:
            writer = SummaryWriter('runs')
        
        ## Core training script
        try:
            for it in range(it, args.max_epoch + 1):
                
                if args.distributed:
                    train_sampler.set_epoch(it)
                    
                clr = [x['lr'] for x in trainer.__optimizer__.param_groups]
                
                train_loss = trainer.train_network(loader=train_loader,
                                                verbose=(args.gpu == 0),
                                                writer=writer,
                                                it=it)
                
                if args.gpu == 0:
                    
                    print('\n', time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, TLOSS {:f}, LR {:f}".format(it, train_loss, max(clr)))
                    scorefile.write("Epoch {:d}, TLOSS {:f}, LR {:f} \n".format(it, train_loss, max(clr)))
                
                if it % args.test_interval == 0:
                    
                    if args.gpu == 0:
                        
                        STOI, PESQ, SISNR, SISDR = trainer.evaluateMetrics(**vars(args))
                        print("STOI {:2.4f}, PSEQ {:2.4f}, SI-SNR {:2.4f}, SI-SDR {:2.4f}".format(STOI, PESQ, SISNR, SISDR))
                        
                        trainer.saveParameters(args.model_save_path + "/model%09d.model"%it)
                        
                        with open(args.model_save_path + "/model%09d.txt"%it, 'w') as metricsfile:
                            metricsfile.write('STOI {:2.4f}, PSEQ {:2.4f}, SI-SNR {:2.4f}, SI-SDR {:2.4f}'.format(STOI, PESQ, SISNR, SISDR))
                        
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
    os.makedirs(args.result_save_path, exist_ok=True)
    
    n_gpus = torch.cuda.device_count()
    
    print('Python Version:', sys.version)
    print('Pytorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:', args.save_path)
    
    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)
    
if __name__ == '__main__':
    main()
"""
Training a single model (student or teacher)
"""
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import numpy as np
import builtins
from learning_lib.loops import test_vanilla
from learning_lib.utils import f1, load_pretrained_weight
from data_lib.dataloader import get_histo_independent_dataloader
from model_lib.model_factory import load_model

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # basic
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--gpu_id', type=str, default='0,1,2', help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', default=12345, type=int, help='seed for initializing training. choices=[None, 0, 1],')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    # multiprocessing
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10004', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--deterministic', action='store_false', help='Make results reproducible, true by default')
    parser.add_argument('--skip_validation', action='store_false', help='Skip validation of teacher')

    parser.add_argument('--pretrained_folder', type=str, default='1', help='trial id')
    opt = parser.parse_args()

    opt.root_dir = f'./result'
    # set the path of model and tensorboard
    opt.model = opt.pretrained_folder.split('/')[0]
    folder_name = opt.pretrained_folder.split('/')[-1]
    opt.dataset = (folder_name.split('_')[0] + '_' + folder_name.split('_')[1]).lower()
    opt.n_cls = int(folder_name.split('_')[2][0])
    opt.learning_strategy = folder_name.split('_')[-1].lower()
    if opt.learning_strategy == 'lora':
        opt.model = f'{opt.model}_LORA8'

    if 'k19' in opt.dataset:
        opt.dataset = 'colon_kather19'
    elif 'kbsmc' in opt.dataset:
        opt.dataset = 'colon_kbsmc1'
    elif 'pcam' in opt.dataset:
        opt.dataset = 'breast_pcam'
    elif 'panda' in opt.dataset:
        opt.dataset = 'prostate_panda512'

    opt.pretrained_path = opt.root_dir + '/' + opt.pretrained_folder

    return opt


metricAcc_best_acc = 0
metricAcc_best_f1 = 0
metricAcc_best_epoch = 0
metricAcc_best_acc_test = 0
metricAcc_best_f1_test = 0

metricF1_best_acc = 0
metricF1_best_f1 = 0
metricF1_best_epoch = 0
metricF1_best_acc_test = 0
metricF1_best_f1_test = 0
total_time = time.time()

def main():
    opt = parse_option()
    # ASSIGN CUDA_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    opt.distributed = opt.world_size > 1 or opt.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        opt.world_size = ngpus_per_node * opt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(opt.gpu_id, ngpus_per_node, opt)


def main_worker(gpu, ngpus_per_node, opt):
    global metricAcc_best_acc, metricAcc_best_f1, metricAcc_best_epoch, metricAcc_best_acc_test, metricAcc_best_f1_test, \
           metricF1_best_acc, metricF1_best_f1, metricF1_best_epoch, metricF1_best_acc_test, metricF1_best_f1_test, \
           total_time
    #opt.batch_size=8
    opt.gpu = gpu
    opt.rank = 0
    opt.dist_backend = 'nccl'

    if opt.multiprocessing_distributed and opt.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.distributed:
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        if opt.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            opt.rank = opt.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

    #if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        # exp_name = opt.model_name.replace('_IS_224_BS256', '')
        # exp_name = exp_name.replace('_strictTrue_RA_CPU16_GPU3_seed12345_epoch50_trial_1_lr1e4', '')
        # exp_name = exp_name.replace('Transfer', 'LinearProbing')

    opt.seed = int(opt.trial)
    if opt.seed is not None:
        print('opt.deterministic',  opt.deterministic)
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        np.random.seed(opt.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)

    model = load_model(opt.model,  ckpt_path='Histo', n_cls=opt.n_cls)
    if 'lora' == opt.learning_strategy:
        print(f'{opt.model}  is LORA')
        state_dict = load_pretrained_weight(opt=opt, whole_weight_path='net_best_acc.pth', lora_only_weight_path='net_lora_best_acc.pth')
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f'{opt.model} is not LORA', )
        state_dict = load_pretrained_weight(opt=opt, whole_weight_path='net_best_acc.pth', lora_only_weight_path=None)
        model.load_state_dict(state_dict, strict=True)

    criterion = nn.CrossEntropyLoss()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif opt.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.gpu is not None:
            torch.cuda.set_device(opt.gpu)
            model.cuda(opt.gpu)
            opt.batch_size = int(opt.batch_size / ngpus_per_node)
            opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu])
        else:
            print('multiprocessing_distributed must be with a specifiec gpu id')
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        model = model.cuda(opt.gpu)
    else:
        criterion = criterion.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    model.eval()
    for n, p in model.named_parameters():
        p.requires_grad = False
    # dataloader
    loader_dict = {}

    if opt.dataset == 'colon_kather19':
        data_dict = {'colon_kather19': opt.n_cls, 'colon_kather16': opt.n_cls, 'colon_crc_tp': 5}
    elif opt.dataset == 'colon_kbsmc1':
        data_dict = {'colon_kbsmc1': 2, 'colon_kbsmc2': 2, 'colon_chaoyang': 2, 'colon_digest2019': 2}
    elif opt.dataset == 'breast_pcam':
        data_dict = {'breast_pcam': opt.n_cls, 'breast_bracs': opt.n_cls, 'breast_bach': opt.n_cls}
    elif opt.dataset == 'prostate_panda':
        data_dict = {'prostate_panda': opt.n_cls, 'prostate_aggc22': opt.n_cls, 'prostate_ubc': opt.n_cls}


    for dataname, nr_classes in data_dict.items():
        valid_loader, test_loader = get_histo_independent_dataloader(dataname=dataname,
                                                                     nr_classes=nr_classes,
                                                                     batch_size=opt.batch_size,
                                                                     num_workers=opt.num_workers,
                                                                     multiprocessing_distributed=opt.multiprocessing_distributed)
        if dataname == 'prostate_aggc22':
            loader_dict['prostate_aggc22_valid'] = valid_loader
        if test_loader is None:
            test_loader = valid_loader
        loader_dict[dataname] = test_loader

    for dataname, test_loader in loader_dict.items():
        try:
            nr_classes = data_dict[dataname]
        except:
            nr_classes = opt.n_cls
        metricAcc_best_acc_test, _, metricAcc_output_stat_test = test_vanilla(dataname, nr_classes, test_loader, model, criterion, opt, prefix='Test')
        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            metricAcc_best_f1_test = f1(metricAcc_output_stat_test['conf_mat'], nr_classes)
            state = {k: v for k, v in opt._get_kwargs()}
            # No. parameters(M)
            num_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
            state['Total params'] = num_params
            state['Total time'] = float('%.2f' % ((time.time() - total_time) / 3600.0))
            print(f' ** [Best Model ({opt.dataset})] {dataname} Acc@1 {metricAcc_best_acc_test:.3f} {dataname} F1 {metricAcc_best_f1_test:.3f} ')
            print(metricAcc_output_stat_test['conf_mat'])


if __name__ == '__main__':
    main()
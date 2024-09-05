"""
Training a single model (student or teacher)
"""
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import loralib as lora
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import time
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import pandas as pd
import numpy as np
import builtins
from learning_lib.loops import train_vanilla as train, validate_vanilla
from learning_lib.utils import save_dict_to_json, reduce_tensor, adjust_learning_rate, update_dict_to_json, f1, mark_model_trainable
from data_lib.dataloader import get_histo_dataloader
from model_lib.model_factory import load_model


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # basic
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', default=12345, type=int, help='seed for initializing training. choices=[None, 0, 1],')
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1.0e-4, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,40,60', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--cosine', action='store_true', help='using cosine annealingdist-url')

    # data_lib
    # data_lib and model
    parser.add_argument('--datafolder', type=str, default='prostate_hv', help='data_lib')
    parser.add_argument('--model', type=str, default='phikon_LORA1')
    parser.add_argument('--pretrain', type=str, default='Histo', help='Histo, None')
    parser.add_argument('--pre_strict', action='store_false', help='strict by default')
    parser.add_argument('--learning_strategy', type=str, default='linear', choices=['full_supervised', 'linear', 'full_ft', 'partial_ft', 'lora'])


    # Augment
    parser.add_argument('--aug_train', type=str, default='RA', help='aug_train')
    parser.add_argument('--crop', type=float, default=0.2, help='crop threshold for RandomResizedCrop')
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--n_cls', type=int, default=4, help='number of class')
    parser.add_argument('--skip_test:', action='store_true', help='strict by default')

    # multiprocessing
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10002', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--deterministic', action='store_false', help='Make results reproducible, true by default')
    parser.add_argument('--skip_validation', action='store_false', help='Skip validation of teacher')

    opt = parser.parse_args()

    opt.rank = 0
    opt.dist_backend = 'nccl'

    # set the path of model and tensorboard
    opt.model_path = f"./result/{opt.model.split('_')[0]}/"
    opt.pretrained_path = f'./pretrained/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    #opt.pretrain = opt.pretrained_path + opt.pretrain

    opt.model_name = f"{opt.model.split('_')[0]}_{opt.dataset}_{opt.n_cls}cls_{opt.learning_strategy}"

    opt.save_folder = os.path.join(opt.model_path, f'{opt.dataset}_{opt.n_cls}cls_{opt.learning_strategy}')
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


metricAcc_best_acc = 0
metricAcc_best_f1 = 0
metricAcc_best_epoch = 0
metricAcc_best_acc_test = 0
metricAcc_best_f1_test = 0
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
        # Since we have ngpus_per_node processes per node, the total world_size needs to be adjusted accordingly
        opt.world_size = ngpus_per_node * opt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(opt.gpu_id, ngpus_per_node, opt)


def main_worker(gpu, ngpus_per_node, opt):
    global metricAcc_best_acc, metricAcc_best_f1, metricAcc_best_epoch, metricAcc_best_acc_test, metricAcc_best_f1_test, total_time

    # Set device
    opt.gpu = int(gpu)

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
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url, world_size=opt.world_size,
                                rank=opt.rank)

    opt.seed = int(opt.trial)
    if opt.seed is not None:
        print('opt.deterministic', opt.deterministic)
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        np.random.seed(opt.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)

    # Set model, optimizer and loss function
    if opt.learning_strategy == 'fully_supervised':
        model = load_model(model_name=opt.model, ckpt_path='None', n_cls=opt.n_cls)
    else:
        model = load_model(model_name=opt.model, ckpt_path='Histo', n_cls=opt.n_cls)

    mark_model_trainable(model, learning_strategy=opt.learning_strategy, n_cls=opt.n_cls)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name}: requires_grad={param.requires_grad}')


    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
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
            if 'LORA' in opt.model:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu])
        else:
            print('multiprocessing_distributed must be with a specifiec gpu id')
            model.cuda()
            if 'LORA' in opt.model:
                model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        model = model.cuda(opt.gpu)
    else:
        criterion = criterion.cuda()
        if torch.cuda.device_count() > 1:
            if 'LORA' in opt.model:
                model = torch.nn.DataParallel(model, find_unused_parameters=True).cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    if not opt.deterministic:
        cudnn.benchmark = True

    # Set dataloader
    train_loader, val_loader, test_loader, train_sampler = get_histo_dataloader(
        opt=opt,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        multiprocessing_distributed=opt.multiprocessing_distributed
    )

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss, train_output_stat = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_loss]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_loss = reduced.tolist()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            train_f1 = f1(train_output_stat['conf_mat'], opt.n_cls)
            print(' * Epoch {}, Acc@1 {:.3f}, , F1 {:.3f}, Time {:.2f}'.format(epoch, train_acc, train_f1, time2 - time1))


        val_acc, val_loss, val_output_stat = validate_vanilla(val_loader, model, criterion, opt, prefix='Val')
        if test_loader is not None:
            test_acc, test_loss, test_output_stat = validate_vanilla(test_loader, model, criterion, opt, prefix='Test')

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            val_f1 = f1(val_output_stat['conf_mat'], opt.n_cls)
            if test_loader is not None:
                test_f1 = f1(test_output_stat['conf_mat'], opt.n_cls)
            else:
                test_acc, test_f1 = 0, 0

            #wandb_logger.log({'val_acc': val_acc, 'val_f1': val_f1, 'test_acc': test_acc, 'test_f1': test_f1}, commit=True)

            # save the best model
            if val_acc > metricAcc_best_acc:
                metricAcc_best_epoch = epoch
                metricAcc_best_acc, metricAcc_best_f1 = val_acc, val_f1
                metricAcc_best_acc_test, metricAcc_best_f1_test = test_acc, test_f1

                state = {'model': model.state_dict(),
                         'metricAcc_best_epoch': metricAcc_best_epoch,
                         'metricAcc_best_acc': metricAcc_best_acc,
                         'metricAcc_best_f1': metricAcc_best_f1,
                         'optimizer': optimizer.state_dict()
                         }

                save_file = os.path.join(opt.save_folder, 'net_best.pth')
                torch.save(state, save_file)
                print('saving the best model!')
                if 'LORA' in opt.model:
                    save_lora = os.path.join(opt.save_folder, 'net_lora_best.pth')
                    torch.save(lora.lora_state_dict(model), save_lora)
                    print('saving the best LORA model!')

            print(' ** Valid Acc@1 {:.3f} Valid F1 {:.4f} Test Acc@1 {:.3f} Test F1 {:.4f}'.format(val_acc, val_f1, test_acc, test_f1))
            print(' ** [Best Model (metric-Acc)] Valid Acc@1 {:.3f} Valid F1 {:.3f} Test Acc@1 {:.3f} Test F1 {:.3f} - Epoch {}'.format(
                metricAcc_best_acc, metricAcc_best_f1, metricAcc_best_acc_test, metricAcc_best_f1_test, metricAcc_best_epoch))

            valid_metrics = {'val_cf': pd.Series({'conf_mat': val_output_stat['conf_mat']}).to_json(orient='records'), 'val_loss': val_loss, 'val_acc': val_acc, 'test_acc': test_acc}
            update_dict_to_json(epoch, valid_metrics, os.path.join(opt.save_folder, "stat.json"))



    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        state = {k: v for k, v in opt._get_kwargs()}
        # No. parameters(M)
        num_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
        state['Total params'] = num_params
        state['Total time'] = float('%.2f' % ((time.time() - total_time) / 3600.0))
        params_json_path = os.path.join(opt.save_folder, "parameters.json")
        save_dict_to_json(state, params_json_path)

        print(' ** [Best Model (metric-Acc)] Valid Acc@1 {:.3f} Valid F1 {:.3f} Test Acc@1 {:.3f} Test F1 {:.3f} - Epoch {}'.format(
                metricAcc_best_acc, metricAcc_best_f1, metricAcc_best_acc_test, metricAcc_best_f1_test, metricAcc_best_epoch))


if __name__ == '__main__':
    main()
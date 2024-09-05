from __future__ import print_function, division
import sys
import time
import torch
# from learning_lib.utils import AverageMeter, accuracy, reduce_tensor, process_accumulated_output, accuracy_merge, process_accumulated_output_independent
from learning_lib.utils import *
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np

def f1(x, n_cls):
    "F1"
    f = 0
    for i in range(x.shape[0]):
        if x[i][i] == 0:
            f += 0
        else:
            f += (2 * x[i][i] / x[:, i].sum() * x[i][i] / x[i, :].sum()) / (
                    x[i][i] / x[:, i].sum() + x[i][i] / x[i, :].sum())
    return f / n_cls


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    infer_output = ['logit', 'true']
    accumulator = {metric: [] for metric in infer_output}
    # n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size
    n_batch = len(train_loader)

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        images, labels = batch_data

        # if opt.dali is None:
        #     images, labels = batch_data
        # else:
        #     images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # ===================forward=====================
        output = model(images)
        loss = criterion(output, labels)
        losses.update(loss.item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(output, labels, topk=(1,))
        top1.update(metrics[0].item(), images.size(0))
        batch_time.update(time.time() - end)
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accumulator['logit'].extend([output.cpu().detach().numpy()])
        accumulator['true'].extend([labels.cpu().detach().numpy()])
        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}'.format(
                epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                loss=losses, top1=top1))
            sys.stdout.flush()

    output_stat = process_accumulated_output(accumulator, opt.batch_size, opt.n_cls)
    return top1.avg, losses.avg, output_stat

def validate_vanilla(val_loader, model, criterion, opt, prefix='Test'):
    """validation"""

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    infer_output = ['logit', 'true']
    accumulator = {metric: [] for metric in infer_output}
    # switch to evaluate mode
    model.eval()

    # n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size
    n_batch = len(val_loader)

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            images, labels = batch_data
            # if opt.dali is None:
            #     images, labels = batch_data
            # else:
            #     images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()
            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1,))
            top1.update(metrics[0].item(), images.size(0))
            batch_time.update(time.time() - end)
            accumulator['logit'].extend([output.cpu().numpy()])
            accumulator['true'].extend([labels.cpu().numpy()])
            if idx % opt.print_freq == 0:
                print('{3}: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}'.format(
                    idx, n_batch, opt.gpu, prefix, batch_time=batch_time, loss=losses,
                    top1=top1))
    output_stat = process_accumulated_output(accumulator, opt.batch_size, opt.n_cls)

    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, losses.count]).to(opt.gpu)
        cf_metrics = torch.tensor(output_stat['conf_mat']).to(opt.gpu)

        total_metrics = reduce_tensor(total_metrics, 1)  # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        cf_metrics = reduce_tensor(cf_metrics, 1)

        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        stat = {'acc': ret[0], 'conf_mat': cf_metrics.cpu().numpy()}

        return ret[0], ret[1], stat

    return top1.avg, losses.avg, output_stat



def test_vanilla(dataname, nr_classes, val_loader, model, criterion, opt, prefix='Test'):
    """test"""

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    infer_output = ['logit', 'true']
    accumulator = {metric: [] for metric in infer_output}
    # switch to evaluate mode
    model.eval()

    # n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size
    n_batch = len(val_loader)

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            images, labels = batch_data
            # if opt.dali is None:
            #     images, labels = batch_data
            # else:
            #     images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()
            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            if dataname == 'colon_chaoyang' and opt.n_cls == 4:
                metrics = accuracy_merge(output, labels, topk=(1,))
            else:
                metrics = accuracy(output, labels, topk=(1,))

            top1.update(metrics[0].item(), images.size(0))
            batch_time.update(time.time() - end)
            accumulator['logit'].extend([output.cpu().numpy()])
            accumulator['true'].extend([labels.cpu().numpy()])
            if idx % opt.print_freq == 0:
                print('{3}: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}'.format(
                    idx, n_batch, opt.gpu, prefix, batch_time=batch_time, loss=losses,
                    top1=top1))
    output_stat = process_accumulated_output_testversion(accumulator, dataname, nr_classes, opt)

    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, losses.count]).to(opt.gpu)
        cf_metrics = torch.tensor(output_stat['conf_mat']).to(opt.gpu)

        total_metrics = reduce_tensor(total_metrics, 1)  # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        cf_metrics = reduce_tensor(cf_metrics, 1)

        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        stat = {'acc': ret[0], 'conf_mat': cf_metrics.cpu().numpy()}

        return ret[0], ret[1], stat

    return top1.avg, losses.avg, output_stat
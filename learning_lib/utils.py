from __future__ import print_function

import json
import os
import re
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import confusion_matrix
import argparse

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def update_fewshot(self, val, init, alpha=0.2):
        self.val = val
        if init:
            self.avg = val
        else:
            self.avg = alpha * val + (1 - alpha) * self.avg


def process_accumulated_output(output, batch_size, nr_classes):
    #
    def uneven_seq_to_np(seq):
        # print(seq)
        # print((len(seq) - 1) )
        # print(len(seq[-1]) )
        item_count = batch_size * (len(seq) - 1) + len(seq[-1])
        cat_array = np.zeros((item_count,) + seq[0][0].shape, seq[0].dtype)
        # BUG: odd len even
        if len(seq) < 2:
            return seq[0]
        for idx in range(0, len(seq) - 1):
            cat_array[idx * batch_size:
                      (idx + 1) * batch_size] = seq[idx]
        cat_array[(idx + 1) * batch_size:] = seq[-1]
        return cat_array

    proc_output = dict()
    true = uneven_seq_to_np(output['true'])
    # threshold then get accuracy
    logit = uneven_seq_to_np(output['logit'])

    pred = np.argmax(logit, axis=-1)
    # pred_c = [covert_dict[pred_c[idx]] for idx in range(len(pred_c))]
    acc = np.mean(pred == true)
    #print('acc', acc)
    # print(classification_report(true, pred_c, labels=[0, 1, 2, 3]))
    # confusion matrix
    conf_mat = confusion_matrix(true, pred, labels=np.arange(nr_classes))
    proc_output.update(acc=acc, conf_mat=conf_mat,)
    return proc_output


def process_accumulated_output_testversion(output, dataname, nr_classes, opt):
    def uneven_seq_to_np(seq):
        # print(seq)
        # print((len(seq) - 1) )
        # print(len(seq[-1]) )
        item_count = opt.batch_size * (len(seq) - 1) + len(seq[-1])
        cat_array = np.zeros((item_count,) + seq[0][0].shape, seq[0].dtype)
        # BUG: odd len even
        if len(seq) < 2:
            return seq[0]
        for idx in range(0, len(seq) - 1):
            cat_array[idx * opt.batch_size:
                      (idx + 1) * opt.batch_size] = seq[idx]
        cat_array[(idx + 1) * opt.batch_size:] = seq[-1]
        return cat_array

    proc_output = dict()
    true = uneven_seq_to_np(output['true'])
    # threshold then get accuracy
    logit = uneven_seq_to_np(output['logit'])

    pred = np.argmax(logit, axis=-1)
    if dataname == 'colon_chaoyang' and opt.n_cls == 4:
        pred[pred == 1] = 0
        pred[pred == 2] = 1
        pred[pred == 3] = 1
    # pred_c = [covert_dict[pred_c[idx]] for idx in range(len(pred_c))]
    acc = np.mean(pred == true)
    # print('acc', acc)
    # print(classification_report(true, pred_c, labels=[0, 1, 2, 3]))
    # confusion matrix
    conf_mat = confusion_matrix(true, pred, labels=np.arange(nr_classes))
    proc_output.update(acc=acc, conf_mat=conf_mat, )
    return proc_output


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim = 1, largest = True, sorted = True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_merge(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        pred[pred == 1] = 0
        pred[pred == 2] = 1
        pred[pred == 3] = 1
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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


def adjust_learning_rate(epoch, opt, optimizer):
    lr = opt.learning_rate
    if opt.cosine:
        eta_min = lr * (opt.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / opt.epochs)) / 2

    else:
        steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
        if steps > 0:
            lr = lr * (opt.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_dict_to_json(epoch, d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    if not os.path.isfile(json_path):
        with open(json_path, 'w') as f:
            # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
            d = {k: v for k, v in d.items()}
            s = {epoch:d}
            json.dump(s, f, indent=4)

    with open(json_path) as json_file:
        json_data = json.load(json_file)

    d = {k: v for k, v in d.items()}
    current_epoch_dict = {epoch: d}
    json_data.update(current_epoch_dict)

    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


def reduce_tensor(tensor, world_size = 1, op='avg'):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size > 1:
        rt = torch.true_divide(rt, world_size)
    return rt


def get_unfreeze_encoder_params(model, freeze_ratio=0.5):
    unfreeze_list = []

    for n, p in model.named_parameters():
        if 'phikon.embeddings.' in n:
            pattern = r'((?:.+)?embeddings)'
            unfreeze_list.append(re.search(pattern, n).group(0))
        elif 'phikon.encoder.layer.' in n:
            pattern = r'((?:.+)?layer\.\d+\.)'
            unfreeze_list.append(re.search(pattern, n).group(0))
        elif 'blocks' in n:
            pattern = r'((?:.+)?blocks\.\d+\.)'
            unfreeze_list.append(re.search(pattern, n).group(0))
        elif 'patch_embed' in n:
            unfreeze_list.append('patch_embed')
        elif n in ['cls_token', 'pos_embed', 'uni.cls_token', 'uni.pos_embed']:
            unfreeze_list.append(n)
        elif n in ['norm.weight', 'norm.bias']:
            unfreeze_list.append('norm')

    unfreeze_list = list(OrderedDict.fromkeys(unfreeze_list))
    unfreeze_list = unfreeze_list[int(len(unfreeze_list) * freeze_ratio):]
    return unfreeze_list


def mark_model_trainable(model, learning_strategy, n_cls) -> None:
    if learning_strategy.lower() in ['none', 'full_ft', 'fully_supervised']:
        return

    if learning_strategy.lower == 'linear':
         for n, p in model.named_parameters():
             p.requires_grad = False
    elif learning_strategy.lower() == 'lora':
        for n, p in model.named_parameters():
            if 'lora' not in n:
                p.requires_grad = False
    elif learning_strategy.lower() == 'partial_ft':
        for n, p in model.named_parameters():
            p.requires_grad = False

        unfreeze_params_list = get_unfreeze_encoder_params(model)
        block_num = 0
        for n, p in model.named_parameters():
            rename = None
            if 'phikon.embeddings.' in n:
                pattern = r'((?:.+)?embeddings)'
                rename = re.search(pattern, n).group(0)
            elif 'phikon.encoder.layer.' in n:
                pattern = r'((?:.+)?layer\.\d+\.)'
                rename = re.search(pattern, n).group(0)
            elif 'blocks' in n:
                block_num = int(n.split('blocks.')[-1][0])
                pattern = r'((?:.+)?blocks\.\d+\.)'
                rename = re.search(pattern, n).group(0)
            elif ('layers.' in n) and ('.downsample.' in n):
                rename = '.'.join(n.split('.')[:2]) + f'.blocks.{block_num}.'
            elif 'patch_embed' in n:
                rename = 'patch_embed'
            elif n in ['cls_token', 'pos_embed']:
                rename = n
            elif n in ['norm.weight', 'norm.bias']:
                rename = 'norm'

            if (rename is None) or (rename in unfreeze_params_list):
                p.requires_grad = True
            else:
                p.requires_grad = False


    for n, p in model.named_parameters():
        if ('classifier' in n or 'head' in n or 'fc' in n) and p.shape[0] == n_cls:
            p.requires_grad = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name}: requires_grad={param.requires_grad}')

    return


def load_pretrained_weight(opt, whole_weight_path, lora_only_weight_path=None):
    state_dict = OrderedDict()
    if lora_only_weight_path:
        lora_state_dict = torch.load(os.path.join(opt.pretrained_path, lora_only_weight_path))
        if 'model' in lora_state_dict.keys():
            lora_state_dict = lora_state_dict['model']
        for k, v in lora_state_dict.items():
            k = k.replace('module.', '')
            state_dict[k] = v

        whole_state_dict = torch.load(os.path.join(opt.pretrained_path, whole_weight_path))
        if 'model' in whole_state_dict.keys():
            whole_state_dict = whole_state_dict['model']
        for k, v in whole_state_dict.items():
            k = k.replace('module.', '')
            if ('classifier' in k or 'head' in k or 'fc' in k) and v.shape[0] == opt.n_cls:
                state_dict[k] = v
    else:
        whole_state_dict = torch.load(os.path.join(opt.pretrained_path, whole_weight_path))
        if 'model' in whole_state_dict.keys():
            whole_state_dict = whole_state_dict['model']
        for k, v in whole_state_dict.items():
            k = k.replace('module.', '')
            state_dict[k] = v

    return state_dict


def load_pretrained_weight_fewshot(pretrained_path):
    state_dict = OrderedDict()
    whole_state_dict = torch.load(pretrained_path)
    if 'model' in whole_state_dict.keys():
        whole_state_dict = whole_state_dict['model']
    for k, v in whole_state_dict.items():
        k = k.replace('module.', '')
        state_dict[k] = v

    return state_dict


# def get_model_dir(args: argparse.Namespace, seed: int):
#     model_type = args.method if args.episodic_training else 'standard'
#     train = "train={}".format('_'.join(args.train_sources))
#     valid = "valid={}".format('_'.join(args.val_sources))
#     return os.path.join(args.ckpt_path,
#                         train,
#                         valid,
#                         f'method={model_type}',
#                         f'pretrained={args.pretrained}',
#                         f'image_size={args.image_size}',
#                         f'model={args.model}',
#                         f'seed={seed}')

def save_checkpoint(state, folder, filename='model_best.pth.tar'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
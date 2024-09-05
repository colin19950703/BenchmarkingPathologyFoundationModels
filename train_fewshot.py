import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=FutureWarning)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import argparse
import random
import time
from typing import Dict


import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from fewshot_lib.config import Split
from fewshot_lib.dataloder_fewshot import get_fewshot_dataloader
from fewshot_lib.methods import __dict__ as all_methods
from learning_lib.utils import AverageMeter, save_checkpoint, load_pretrained_weight_fewshot, f1
from model_lib.model_factory import load_model


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training')
    # DATA:
    parser.add_argument('--image_size', type=int, default=224, help='Images will be resized to this value')
    parser.add_argument('--train_sources', nargs='+', default=['colon_crc_tp'], help='Which data_lib to use')
    parser.add_argument('--val_sources', nargs='+', default=['colon_kather19'], help='Which data_lib to use')
    parser.add_argument('--test_sources', nargs='+', default=['breakhis'], help='Which data_lib to use')
    parser.add_argument('--train_transforms', nargs='+', default=['random_resized_crop', 'random_flip', 'jitter', 'to_tensor', 'normalize'], help='Transforms applied to training data')
    parser.add_argument('--test_transforms', nargs='+', default=['resize', 'center_crop', 'to_tensor', 'normalize'], help='Transforms applied to test data')
    parser.add_argument('--data_path', type=str, default='./datafolder/converted_data/', help='Path to the data')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints', help='Path to save checkpoints')
    parser.add_argument('--res_path', type=str, default='results', help='Path to save results')
    parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the data')
    # MODEL
    parser.add_argument('--model', type=str, default='ctranspath', help='Model architecture')
    parser.add_argument('--use_fc', type=bool, default=True, help='Whether to use fully connected layer')
    parser.add_argument('--pretrained', type=str, default='Histo', help='Whether to use pretrained model')
    # TRAINING
    parser.add_argument('--seeds', type=int, default=2021, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--train_freq', type=int, default=50, help='Frequency of training iterations')
    parser.add_argument('--train_iter', type=int, default=50000, help='Number of training iterations')
    parser.add_argument('--loss', type=str, default='_CrossEntropy', help='Loss function')
    parser.add_argument('--focal_gamma', type=float, default=3.0, help='Gamma parameter for focal loss')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    # VALIDATION
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size for validation')
    parser.add_argument('--val_iter', type=int, default=250, help='Number of validation iterations')
    parser.add_argument('--val_freq', type=int, default=1000, help='Frequency of validation')
    # TEST
    parser.add_argument('--test_batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--test_iter', type=int, default=1000, help='Number of testing iterations')
    parser.add_argument('--simu_params', nargs='+',
                        default=['train_sources', 'val_sources', 'test_sources', 'arch', 'image_size', 'pretrained',
                                 'num_support', 'seed'], help='Simulation parameters')

    # AUGMENTATIONS:
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--cutmix_prob', type=float, default=1.0)
    parser.add_argument('--augmentation', type=str, default='none')
    # OPTIM:
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.1)

    # EPISODES
    parser.add_argument('--num_ways', type=int, default=4, help='Set it if you want a fixed # of ways per task')
    parser.add_argument('--num_support', type=int, default=5, help='Set it if you want a fixed # of support samples per class')
    parser.add_argument('--num_query', type=int, default=15, help='Set it if you want a fixed # of query samples per class')
    parser.add_argument('--min_ways', type=int, default=2, help='Minimum # of ways per task')
    parser.add_argument('--max_ways_upper_bound', type=int, default=10, help='Maximum # of ways per task')
    parser.add_argument('--max_num_query', type=int, default=10, help='Maximum # of query samples')
    parser.add_argument('--max_support_set_size', type=int, default=100, help='Maximum # of support samples')
    parser.add_argument('--min_examples_in_class', type=int, default=0, help='Classes that have less samples will be skipped')
    parser.add_argument('--max_support_size_contrib_per_class', type=int, default=10, help='Maximum # of support samples per class')
    parser.add_argument('--min_log_weight', type=float, default=-0.69314718055994529, help='Do not touch, used to randomly sample support set')
    parser.add_argument('--max_log_weight', type=float, default=0.69314718055994529, help='Do not touch, used to randomly sample support set')
    parser.add_argument('--ignore_bilevel_ontology', type=bool, default=True)

    parser.add_argument('--method', type=str, default='SimpleShot')
    parser.add_argument('--freeze_encoder', action='store_true', help='freeze the encoder')
    parser.add_argument('--freeze_ratio', type=float, default=0.0, help='freeze the encoder')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    opt = parser.parse_args()

    model_name = {opt.model.split('_')[0].lower()}
    opt.model_dir = f"checkpoints/{model_name}/{model_name}_{opt.num_ways}ways_{opt.num_support}shots_{opt.num_query}query_{opt.method}"
    #opt.model_dir = f"result/{model_name}/{model_name}_{opt.num_ways}ways_{opt.num_support}shots_{opt.num_query}query_{opt.method}"

    return opt

def main(opt):

    if opt.seeds:
        opt.seed = opt.seeds
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        random.seed(opt.seed)

    # ============ Device ================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ============ Data loaders =========
    train_loader, num_classes = get_fewshot_dataloader(opt=opt,
                                                       sources=opt.train_sources,
                                                       batch_size=opt.batch_size,
                                                       split=Split["TRAIN"])

    val_loader, num_classes_val = get_fewshot_dataloader(opt=opt,
                                                         sources=opt.val_sources,
                                                         batch_size=opt.val_batch_size,
                                                         split=Split["VALID"])

    print(f"=> There are {num_classes} classes in the train datasets")
    print(f"=> There are {num_classes_val} classes in the validation datasets")

    # ============ Model and optim ================
    if 'True' == opt.pretrained or 'Histo' in opt.pretrained:
        model = load_model(opt.model, 'Histo', opt.num_ways)
        print("Histopathological pretrained weights loaded")
    else:
        model = load_model(opt.model, 'Histo', opt.num_ways)
        state_dict = load_pretrained_weight_fewshot(pretrained_path=opt.pretrained)
        if 'head.weight' in state_dict:
            del state_dict['head.weight']
        if 'head.bias' in state_dict:
            del state_dict['head.bias']
        if 'phikon.head.weight' in state_dict:
            del state_dict['phikon.head.weight']
        if 'phikon.head.bias' in state_dict:
            del state_dict['phikon.head.bias']
        if 'uni.head.weight' in state_dict:
            del state_dict['uni.head.weight']
        if 'uni.head.bias' in state_dict:
            del state_dict['uni.head.bias']
        model.load_state_dict(state_dict, strict=False)
        print("Custom pretrained weights loaded")

    # ============ Training method ================
    print(f"=> Using {opt.method} method")
    if opt.method.lower() == 'baseline' or opt.method.lower() == 'baselineplusplus':
        feat, _ = model(torch.randn(1, 3, opt.image_size, opt.image_size), is_feat=True)
        method = all_methods[opt.method](opt=opt, feature_dim=feat.shape[-1], finetune_all_layers=True)
        if torch.cuda.device_count() > 1:
            method = torch.nn.DataParallel(method)
            print("Multi-GPU")
        method = method.to(device)
    else:
        method = all_methods[opt.method](opt=opt)

    # Apply DataParallel to use multiple GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, opt.train_iter, eta_min=1e-9)

    # ============ Prepare metrics ================
    metrics: Dict[str, torch.tensor] = {"train_loss": torch.zeros(int(opt.train_iter / opt.train_freq)).type(torch.float32),
                                        "train_acc": torch.zeros(int(opt.train_iter / opt.train_freq)).type(torch.float32),
                                        "val_acc": torch.zeros(int(opt.train_iter / opt.val_freq)).type(torch.float32),
                                        "val_loss": torch.zeros(int(opt.train_iter / opt.val_freq)).type(torch.float32),
                                        "test_acc": torch.zeros(int(opt.train_iter / opt.val_freq)).type(torch.float32),
                                        "test_loss": torch.zeros(int(opt.train_iter / opt.val_freq)).type(torch.float32),
                                        }
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    best_episode_iter, best_val_acc, best_val_f1 = 0, 0., 0.

    # ============ Training loop ============
    model.train()
    tqdm_bar = tqdm(train_loader, total=opt.train_iter, ascii=True)
    iter = 0
    for data in tqdm_bar:
        iter+=1
        if iter >= opt.train_iter:
            break

        # ============ Make a training iteration ============
        t0 = time.time()
        support, query, support_labels, target = data
        support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
        query, target = query.to(device), target.to(device, non_blocking=True)
        loss, preds_q = method(x_s=support,
                               x_q=query,
                               y_s=support_labels,
                               y_q=target,
                               model=model)  # [batch, q_shot]

        # Perform optim
        if opt.method.lower() != 'baselineplusplus' and opt.method.lower() != 'baseline':
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        # Log metrics
        train_loss.update_fewshot(loss.mean().detach(), iter == 0)
        train_acc.update_fewshot((preds_q == target).float().mean(), iter == 0)
        batch_time.update_fewshot(time.time() - t0, iter == 0)

        if iter % opt.train_freq == 0:
            tqdm_bar.set_description('Time {batch_time.val:.3f} ({batch_time.avg:.3f}) Loss {loss.val:.4f} ({loss.avg:.4f}) Acc {acc.val:.4f} ({acc.avg:.4f})'.format(
                batch_time=batch_time,
                loss=train_loss,
                acc=train_acc))

            for k in metrics:
                if 'train' in k:
                    metrics[k][int(iter / opt.train_freq)] = eval(k).avg

        # ============ Evaluation ============
        if iter % opt.val_freq == 0:
            val_episode_acc, val_episode_f1, val_episode_loss = evaluate(val_loader, model, method, device, opt)
            if val_episode_acc > best_val_acc:
                best_episode_iter = iter
                best_val_acc = val_episode_acc
                best_val_f1 = val_episode_f1
                best_weights = model.state_dict().copy()

                state = {'model': best_weights,
                         'best_episode_iter': best_episode_iter,
                         'best_val_acc': best_val_acc,
                         'best_val_f1': best_val_f1,
                         'optimizer': optimizer.state_dict()}
                save_checkpoint(state=state, folder=opt.model_dir, filename='net_best_acc.pth')
                print('saving the best acc model!')

            print(' ** Valid Acc@1 {:.3f} Valid F1 {:.4f}'.format(val_episode_acc, val_episode_f1))
            print(' ** [Best Model] Valid Acc@1 {:.3f} Valid F1 {:.3f} - Episode(Iter) {}'.format(best_val_acc, best_val_f1, best_episode_iter))

def evaluate(loader, model, method, device, opt):
    print('Starting validation ...')
    model.eval()
    method.eval()

    tqdm_eval_bar = tqdm(loader, total=opt.val_iter, ascii=True)
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    val_f1 = AverageMeter()

    for j, data in enumerate(tqdm_eval_bar):
        support, query, support_labels, query_labels = data
        support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
        query, query_labels = query.to(device), query_labels.to(device, non_blocking=True)

        loss, query_pred = method(x_s=support,
                                  x_q=query,
                                  y_s=support_labels,
                                  y_q=query_labels,
                                  model=model)

        valid_episode_acc = (query_pred == query_labels).float().mean() * 100.
        query_pred, query_labels = query_pred.detach().cpu().numpy(), query_labels.detach().cpu().numpy()
        valid_episode_cm = confusion_matrix(query_labels[0], query_pred[0], labels=np.arange(opt.num_ways))
        try:
            valid_episode_f1 = f1(valid_episode_cm, opt.num_ways).mean()
        except:
            valid_episode_f1 = f1(valid_episode_cm, opt.num_ways)

        val_acc.update_fewshot(valid_episode_acc, False)
        val_f1.update_fewshot(valid_episode_f1, False)
        val_loss.update_fewshot(loss.mean().detach(), False)
        tqdm_eval_bar.set_description(f'Val Acc@1 {val_acc.avg:.3f} Val F1 {val_f1.avg:.3f} Val Loss {val_loss.avg:.3f}')

        if j >= opt.val_iter:
            break

    model.train()
    method.train()

    return val_acc.avg, val_f1.avg, val_loss.avg


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

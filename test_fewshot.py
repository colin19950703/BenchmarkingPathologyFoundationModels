import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=FutureWarning)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from fewshot_lib.config import Split
from fewshot_lib.dataloder_fewshot import get_fewshot_dataloader
from fewshot_lib.methods import __dict__ as all_methods
from learning_lib.utils import AverageMeter, load_pretrained_weight_fewshot, f1
from model_lib.model_factory import load_model
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training')
    # DATA:
    parser.add_argument('--image_size', type=int, default=224, help='Images will be resized to this value')
    parser.add_argument('--train_sources', nargs='+', default=['colon_crc_tp'], help='Which data_lib to use')
    parser.add_argument('--val_sources', nargs='+', default=['colon_kather19'], help='Which data_lib to use')
    #parser.add_argument('--test_sources', nargs='+', default=['colon_kbsmc', 'etc_lc25000', 'gastric_kbsmc', 'prostate_panda', 'breast_bach', 'breast_breakhis'], help='Which data_lib to use')
    parser.add_argument('--test_sources', nargs='+',default=['colon_kbsmc', 'etc_lc25000', 'gastric_kbsmc', 'prostate_panda', 'breast_bach'], help='Which data_lib to use')
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
    parser.add_argument('--simu_params', nargs='+', default=['train_sources', 'val_sources', 'test_sources', 'arch', 'image_size', 'pretrained', 'num_support', 'seed'], help='Simulation parameters')
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
    parser.add_argument('--num_support', type=int, default=5, help='Set it if you wanxt a fixed # of support samples per class')
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

    parser.add_argument('--method', type=str, default='ProtoNet')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    opt = parser.parse_args()

    model_name = opt.model.split('_')[0].lower()
    opt.model_dir = f"result/{model_name}/{model_name}_{opt.num_ways}ways_{opt.num_support}shots_{opt.num_query}query_{opt.method}"
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
    val_loader, num_classes_val = get_fewshot_dataloader(opt=opt,
                                                         sources=opt.val_sources,
                                                         batch_size=opt.val_batch_size,
                                                         split=Split["VALID"])

    test_dict = {'test_source': [], 'test_loader': [], 'num_classes_test': []}
    for source in opt.test_sources:
        loader, num_classes = get_fewshot_dataloader(opt=opt,
                                                     sources=[source],
                                                     batch_size=opt.val_batch_size,
                                                     split=Split["TEST"])
        test_dict['test_source'].append(source)
        test_dict['test_loader'].append(loader)
        test_dict['num_classes_test'].append(num_classes)


    #  If you want to get the total number of classes (i.e from combined datasets)
    print(f"=> There are {num_classes_val} classes in the validation datasets")
    for idx, test_source in enumerate(test_dict['test_source']):
        print(f"=> There are {test_dict['num_classes_test'][idx]} classes in the {test_dict['test_source'][idx]} datasets")

    # ============ Model and optim ================
    if 'True' == opt.pretrained or 'Histo' == opt.pretrained:
        model = load_model(opt.model, 'Histo', opt.num_ways)
        print("Histo pretrained weights loaded")
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
        method = all_methods[opt.method](opt=opt, feature_dim=feat.shape[-1], finetune_all_layers=False)
        if torch.cuda.device_count() > 1:
            method = torch.nn.DataParallel(method)
            print("Multi-GPU")
        method = method.to(device)
    else:
        method = all_methods[opt.method](opt=opt)

    # Apply DataParallel to use multiple GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print("Multi-GPU")

    model = model.to(device)
    model.eval()
    # ============ Evaluation loop ============

    prefix = 'Validation'
    valid_episode_acc, valid_episode_f1, valid_episode_loss = evaluate(val_loader, model, method, device, opt, prefix=prefix)
    print(' ** Valid Acc@1 {:.3f} Valid F1 {:.4f}'.format(valid_episode_acc, valid_episode_f1))
    valid_metrics = {'val_acc': valid_episode_acc, 'val_f1': valid_episode_f1, 'val_loss': valid_episode_loss}

    prefix = 'Testing'
    for idx, test_source in enumerate(test_dict['test_source']):
        test_loader = test_dict['test_loader'][idx]
        test_episode_acc, test_episode_f1, test_episode_loss = evaluate(test_loader, model, method, device, opt, prefix=prefix)
        print(' ** Test Acc@1 {:.3f} Test F1 {:.4f}'.format(test_episode_acc, test_episode_f1))
        test_metrics = {f"test_{test_dict['test_source'][idx]}_acc": test_episode_acc,
                        f"test_{test_dict['test_source'][idx]}_f1": test_episode_f1,
                        f"test_{test_dict['test_source'][idx]}_loss": test_episode_loss}


def evaluate(loader, model, method, device, opt, prefix='Validation'):
    print(f'Starting {prefix} ...')
    model.eval()
    method.eval()
    #print(len(loader))
    if 'valid' in prefix.lower():
        tqdm_eval_bar = tqdm(loader, total=opt.val_iter, ascii=True)
    else:
        tqdm_eval_bar = tqdm(loader, total=opt.test_iter, ascii=True)
    eval_loss = AverageMeter()
    eval_acc = AverageMeter()
    eval_f1 = AverageMeter()

    for j, data in enumerate(tqdm_eval_bar):
        support, query, support_labels, query_labels = data
        support, support_labels = support.to(device), support_labels.to(device, non_blocking=True)
        query, query_labels = query.to(device), query_labels.to(device, non_blocking=True)

        loss, query_pred = method(x_s=support,
                                  x_q=query,
                                  y_s=support_labels,
                                  y_q=query_labels,
                                  model=model)  # [batch, q_shot]

        valid_episode_acc = (query_pred == query_labels).float().mean() * 100.
        query_pred, query_labels = query_pred.detach().cpu().numpy(), query_labels.detach().cpu().numpy()
        valid_episode_cm = confusion_matrix(query_labels[0], query_pred[0], labels=np.arange(opt.num_ways))
        try:
            valid_episode_f1 = f1(valid_episode_cm, opt.num_ways).mean()
        except:
            valid_episode_f1 = f1(valid_episode_cm, opt.num_ways)

        eval_acc.update_fewshot(valid_episode_acc, False)
        eval_f1.update_fewshot(valid_episode_f1, False)
        eval_loss.update_fewshot(loss.mean().detach(), False)
        tqdm_eval_bar.set_description(
            f'{prefix} Acc@1 {eval_acc.avg:.3f} {prefix} F1 {eval_f1.avg:.3f} {prefix} Loss {eval_loss.avg:.3f}')


        if j >= opt.val_iter and 'valid' in prefix.lower():
            break

        if j >= opt.test_iter and 'test' in prefix.lower():
            break

    return eval_acc.avg, eval_f1.avg, eval_loss.avg


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

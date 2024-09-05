import torch
from collections import OrderedDict
from model_lib import timm
import warnings
from model_lib.CTranspath.ctran import ctranspath, ctranspath_LORA8
from model_lib.Lunit.dino import lunit_dino16, lunit_dino16_LORA8
from model_lib.Phikon.builder import phikon_base, phikon_LORA8
from model_lib.UNI.builder import uni_base, uni_LORA8
import os

#from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate, update_dict_to_json, load_pretrained_weights, load_pretrained_weights_teacher
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

def loader(model_name, ckpt_path, n_cls):
    if ckpt_path == 'Histo':
        pretrained = True
    else:
        pretrained = False

    model_name = model_name.lower()

    if model_name in 'ctranspath':
        model = ctranspath(num_classes=n_cls, pretrained=pretrained)
    elif model_name == 'ctranspath_lora8':
        model = ctranspath_LORA8(num_classes=n_cls, pretrained=pretrained)
    elif model_name == 'lunit16' or model_name == 'lunit':
        model = lunit_dino16(num_classes=n_cls, pretrained=pretrained)
    elif model_name == 'lunit16_lora8' or model_name == 'lunit_lora8':
        model = lunit_dino16_LORA8(num_classes=n_cls, pretrained=pretrained)
    elif model_name in 'phikon':
        model = phikon_base(num_classes=n_cls, pretrained=pretrained)
    elif model_name == 'phikon_lora8':
        model = phikon_LORA8(num_classes=n_cls, pretrained=pretrained)
    elif model_name in 'uni':
        model = uni_base(num_classes=n_cls, pretrained=pretrained)
    elif model_name == 'uni_lora8':
        model = uni_LORA8(num_classes=n_cls, pretrained=pretrained)
    else:
        raise NotImplementedError(model_name)

    return model
def load_model(model_name, ckpt_path, n_cls):
    model = loader(model_name, ckpt_path, n_cls)

    if ckpt_path and ckpt_path != 'Init' and ckpt_path != 'Histo' and ckpt_path != 'None':
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if 'model' in state_dict.keys():
                state_dict = state_dict['model']
            encoder_state_dict = OrderedDict()
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                encoder_state_dict[k] = v
            if model_name in ['ctranspath', 'lunit']:
                try:
                    encoder_state_dict.pop('classifier_.1.weight')
                    encoder_state_dict.pop('classifier_.1.bias')
                    # model.load_state_dict(encoder_state_dict, strict=False)
                    msg = model.load_state_dict(state_dict, strict=False)
                except:
                    encoder_state_dict.pop('head.weight')
                    encoder_state_dict.pop('head.bias')
                    msg = model.load_state_dict(state_dict, strict=False)
            else:
                msg = model.load_state_dict(state_dict, strict=False)
                # model.load_state_dict(encoder_state_dict, strict=True)
            print(msg)
            #assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
            print("=> loaded pre-trained model '{}'".format(ckpt_path))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_path))
    else:
        print("=> loaded pre-trained weight")

    return model

def load_fewshot_model(model_name, ckpt_path, n_cls):
    model = loader(model_name, ckpt_path, n_cls)

    if ckpt_path not in ['ImageNet', 'Init', 'Histo']:
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if 'model' in state_dict.keys():
                state_dict = state_dict['model']
            encoder_state_dict = OrderedDict()
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                encoder_state_dict[k] = v
            try:
                msg = model.load_state_dict(encoder_state_dict, strict=False)
            except:
                if 'phikon' in model_name:
                    encoder_state_dict.pop('phikon.head.weight')
                    encoder_state_dict.pop('phikon.head.bias')
                else:
                    encoder_state_dict.pop('head.weight')
                    encoder_state_dict.pop('head.bias')
                msg = model.load_state_dict(encoder_state_dict, strict=False)
            print(msg)
            print("=> loaded pre-trained model '{}'".format(ckpt_path))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_path))
    return model
def _test():
    import torch
    import os
    import torch.nn as nn
    import argparse

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser('argument for training')

    #basic
    parser.add_argument('--model_name', type=str, default='effiB0', help='print frequency')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'
    #print('num of cuda', torch.cuda.device_count())

    model_name = args.model_name
    if '224' in model_name:
        dummpy_input = torch.randn(16, 3, 224, 224)
    elif '384' in model_name:
        dummpy_input = torch.randn(16, 3, 384, 384)
    else:
        dummpy_input = torch.randn(16, 3, 512, 512)

    n_cls = 7
    net = load_model(model_name, 'ImageNet', n_cls, strict=False, gpu='0,1', multiprocessing_distributed=False)
    net = nn.DataParallel(net).cuda()


    # y_class = net(dummpy_input.cuda())
    # print(f'{model_name}', y_class.size())
    cnt=0
    for name, param in net.named_parameters():
        # if 'head' in name:
        #     param.requires_grad = True
        if ('classifier' in name or 'head' in name or 'fc' in name) and param.shape[0] == n_cls:
            if param.shape[0] == n_cls:
                print(model_name, name, )
                cnt+=1

    if cnt == 0:
        print(model_name, 'None')

if __name__ == '__main__':
    _test()
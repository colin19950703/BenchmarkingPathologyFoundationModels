import os
import csv
from glob import glob
import random
from collections import Counter
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision import transforms
from torchvision.utils import make_grid
import pandas as pd

def print_number_of_sample(train_set=[], valid_set=[], test_set=[], test2_set=None):
    def fill_empty_label(label_dict):
        for i in range(max(label_dict.keys()) + 1):
            if label_dict[i] != 0:
                continue
            else:
                label_dict[i] = 0
        return dict(sorted(label_dict.items()))

    train_label = [train_set[i][1] for i in range(len(train_set))]
    if len(train_label) != 0:
        d = Counter(train_label)
        d = fill_empty_label(d)
        print("train", d)
        train_label = [d[key] for key in d.keys()]
    else:
        print("train : None", )

    valid_label = [valid_set[i][1] for i in range(len(valid_set))]
    if len(valid_label) != 0:
        d = Counter(valid_label)
        d = fill_empty_label(d)
        print("val", d)
        valid_label = [d[key] for key in d.keys()]
    else:
        print("valid : None", )

    test_label = [test_set[i][1] for i in range(len(test_set))]
    if len(test_label) != 0:
        d = Counter(test_label)
        d = fill_empty_label(d)
        print("test", d)
        test_label = [d[key] for key in d.keys()]
    else:
        print("test : None", )

    if test2_set is not None:
        test2_label = [test2_set[i][1] for i in range(len(test2_set))]
        if len(test2_label) != 0:
            d = Counter(test2_label)
            d = fill_empty_label(d)
            print("test2", d)
            test2_label = [d[key] for key in d.keys()]
        else:
            print("test2 : None", )


#Breast
def prepare_breast_pcam_data(data_root_dir='datafolder/raw_data/breast_pcam', nr_classes=2):
    def load_data_info(base_path, folder_name):
        pathname = f'{base_path}/{folder_name}/*.jpg'
        file_list = glob(pathname, recursive=True)
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]

        return list(zip(file_list, label_list))

    train_set = load_data_info(base_path=data_root_dir, folder_name='Train')
    valid_set = load_data_info(base_path=data_root_dir, folder_name='Valid')
    test_set = load_data_info(base_path=data_root_dir, folder_name='Test')

    print_number_of_sample(train_set, valid_set, test_set)

    return train_set, valid_set, test_set


def prepare_breast_bach_data(data_root_dir='datafolder/raw_data/breast_bach', nr_classes=2):
    def load_data_info(pathname, gt_list, nr_classes, parse_label=True, label_value=0):
        file_list = glob(pathname)
        if parse_label:
            label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            #label_list = [int(label_dict[file_path.split('/')[-2]]) for file_path in file_list]
        else:
            label_list = [label_value] * len(file_list)

        label_list = [gt_list[i] for i in label_list]
        list_out = list(zip(file_list, label_list))
        list_out = [list_out[i] for i in range(len(list_out)) if list_out[i][1] < nr_classes]

        return list_out

    gt_list = {0: 0,
               1: 0,
               2: 1,
               3: 1
               }

    train_set = load_data_info('%s/train/*/*.jpg' % data_root_dir, gt_list=gt_list, nr_classes=nr_classes)
    valid_set = load_data_info('%s/val/*/*.jpg' % data_root_dir, gt_list=gt_list, nr_classes=nr_classes)
    test_set = load_data_info('%s/test/*/*.jpg' % data_root_dir, gt_list=gt_list, nr_classes=nr_classes)

    print_number_of_sample(test_set=train_set + valid_set + test_set)
    return None, None, train_set+valid_set+test_set

def prepare_breast_bracs_data(data_root_dir='datafolder/raw_data/breast_bracs', nr_classes=2):
    def load_data_info(pathname, gt_list, nr_classes, parse_label=True, label_value=0):
        file_list = glob(pathname)
        if parse_label:
            label_list = [int(gt_list[file_path.split('/')[-2]]) for file_path in file_list]
        else:
            label_list = [label_value] * len(file_list)

        list_out = list(zip(file_list, label_list))
        list_out = [list_out[i] for i in range(len(list_out)) if list_out[i][1] < nr_classes]
        return list_out

    gt_list = {'0_N': 0,
               '1_PB': 0,
               '2_UDH': 0,
               '3_FEA': 2,
               '4_ADH': 2,
               '5_DCIS': 1,
               '6_IC': 1,
               }

    train_set = load_data_info('%s/train/*/*.png' % data_root_dir, gt_list=gt_list, nr_classes=nr_classes)
    valid_set = load_data_info('%s/val/*/*.png' % data_root_dir, gt_list=gt_list, nr_classes=nr_classes)
    test_set = load_data_info('%s/test/*/*.png' % data_root_dir, gt_list=gt_list, nr_classes=nr_classes)

    print_number_of_sample(test_set=train_set+valid_set+test_set)
    return None, None, train_set+valid_set+test_set

#Colon Subtyping
def prepare_colon_kather19_data(data_root_dir='datafolder/raw_data/colon_kather19', nr_classes=7):
    def load_data_info(pathname, covert_dict):
        file_list = glob(pathname)
        #datafolder/raw_data/colon_Kather19_Norm/NCT-CRC-HE-100K/BACK
        label_list = [int(covert_dict[file_path.split('/')[-2]][1]) for file_path in file_list]

        return list(zip(file_list, label_list))

    val_ratio = 0.3
    #
    """
    Cls     K19         -   K16
    0   <-  TUM(0)      -   TUMOR(0)   : tumor epithelium
    1   <-  STR(1)      -   STROMA(1)  : simple stroma
    X   <-              -   COMP(3)    : complex stroma
    1   <-  MUS(2)      -              : muscle
    3   <-  LYM(3)      -   LYMPHO(4)  : lymphocytes
    4   <-  DEB(4)      -   DEBRIS(5)  : debris / necrosis
    4   <-  MUC(5)      -              : mucus
    6   <-  NORM(6)     -   MUCOSA(6)  : normal mucosal glands
    7   <-  ADI(7)      -   AIDPOSE(7) : adipose tissue
    8   <-  BACK(8)     -   EMPTY(8)   : background
    """

    const_kather19 = {
        'TUM': ('TUM', 0), 'STR': ('STR', 1), 'MUS': ('MUS', 1), 'LYM': ('LYM', 2),
        'DEB': ('DEB', 3), 'MUC': ('MUC', 3), 'NORM': ('NORM', 4), 'ADI': ('ADI', 5),
        'BACK': ('BACK', 6),
    }

    k19_path = f'{data_root_dir}/norm/'

    k19_test_path = f'{data_root_dir}/external/'
    data_list = load_data_info(pathname=f'{k19_path}/*/*/*.tif', covert_dict=const_kather19,)
    random.Random(5).shuffle(data_list)
    train_set = data_list[int(val_ratio * len(data_list)):]
    valid_set = data_list[:int(val_ratio / 2 * len(data_list))]
    test_set = load_data_info(pathname=f'{k19_test_path}/*/*/*.tif', covert_dict=const_kather19)

    print_number_of_sample(train_set, valid_set, test_set)

    return train_set, valid_set, test_set


def prepare_colon_kather16_data(data_root_dir='datafolder/raw_data/colon_kahter16_colornorm_macenkoNormed', nr_classes=7):
    def load_data_info(root_dir, pathname, covert_dict):
        file_list = glob(pathname)
        COMPLEX_list = glob(f'{root_dir}/03_COMPLEX/*.tif')
        file_list = [elem for elem in file_list if elem not in COMPLEX_list]
        label_list = [int(covert_dict[file_path.split('/')[-2]][1]) for file_path in file_list]
        return list(zip(file_list, label_list))

    # const_kather19 = {
    #     'TUM': ('TUM', 0), 'STR': ('STR', 1), 'MUS': ('MUS', 2), 'LYM': ('LYM', 3),
    #     'DEB': ('DEB', 4), 'MUC': ('MUC', 5), 'NORM': ('NORM', 6), 'ADI': ('ADI', 7),
    #     'BACK': ('BACK', 8),
    # }
    # DEB: debris/necrosis 2 + MUC: mucus 4
    # STR: simple stroma 7 +  MUS: muscle 4

    # we don't use the complex stroma here
    # const_kather16 = {
    #     '07_ADIPOSE': ('07_ADIPOSE', 0), '08_EMPTY': ('08_EMPTY', 1), '05_DEBRIS': ('05_DEBRIS', 2),
    #     '04_LYMPHO': ('04_LYMPHO', 3), '06_MUCOSA': ('06_MUCOSA', 6), '02_STROMA': ('02_STROMA', 7),
    #     '01_TUMOR': ('01_TUMOR', 8)
    # }
    val_ratio = 0.3
    """
    Cls     K19         -   K16
    0   <-  TUM(0)      -   TUMOR(0)   : tumor epithelium
    1   <-  STR(1)      -   STROMA(1)  : simple stroma
    X   <-              -   COMP(3)    : complex stroma
    1   <-  MUS(2)      -              : muscle
    3   <-  LYM(3)      -   LYMPHO(4)  : lymphocytes
    4   <-  DEB(4)      -   DEBRIS(5)  : debris / necrosis
    4   <-  MUC(5)      -              : mucus
    6   <-  NORM(6)     -   MUCOSA(6)  : normal mucosal glands
    7   <-  ADI(7)      -   AIDPOSE(7) : adipose tissue
    8   <-  BACK(8)     -   EMPTY(8)   : background
    """
    const_kather16 = {
        '01_TUMOR': ('01_TUMOR', 0), '02_STROMA': ('02_STROMA', 1), '04_LYMPHO': ('04_LYMPHO', 2),
        '05_DEBRIS': ('05_DEBRIS', 3), '06_MUCOSA': ('06_MUCOSA', 4), '07_ADIPOSE': ('07_ADIPOSE', 5),
        '08_EMPTY': ('08_EMPTY', 6)
    }

    data_list = load_data_info(root_dir=data_root_dir, pathname=f'{data_root_dir}/*/*.tif', covert_dict=const_kather16)
    random.Random(5).shuffle(data_list)
    # k16 for train, validation, test
    train_set = data_list[int(val_ratio * len(data_list)):]
    valid_set = data_list[:int(val_ratio / 2 * len(data_list))]
    test_set = data_list[int(val_ratio / 2 * len(data_list)):int(val_ratio * len(data_list))]

    print_number_of_sample(train_set, valid_set, test_set)

    return None, None, train_set+valid_set+test_set


def prepare_colon_crc_tp_data(data_root_dir='datafolder/raw_data/colon_crc_tp', nr_classes=7):
    def load_data_info(root_dir, covert_dict):
        pathname = f'{root_dir}/*/*.png'
        file_list = glob(pathname)

        label_list = [int(covert_dict[file_path.split('/')[-2]][1]) for file_path in file_list]
        #print(Counter(label_list))
        out_list = list(zip(file_list, label_list))
        out_list = [elem for elem in out_list if elem[1] != 9]
        return out_list

    val_ratio = 0.3

    """
    Cls    | K19     | K16        | CRC-TP       
    0   <- | TUM(0)  | TUMOR(0)   | Tumor          : tumor epithelium
    1   <- | STR(1)  | STROMA(1)  | Stroma         : simple stroma
    X   <- |         | COMP(3)    | Complex Stroma : complex stroma
    1   <- | MUS(2)  |            | Muscle         : muscle
    3   <- | LYM(3)  | LYMPHO(4)  | Inflammatory   : lymphocytes
    4   <- | DEB(4)  | DEBRIS(5)  | Debris         : debris / necrosis
    4   <- | MUC(5)  |            |                : mucus
    6   <- | NORM(6) | MUCOSA(6)  | Benign         : normal mucosal glands
    7   <- | ADI(7)  | AIDPOSE(7) |                : adipose tissue
    8   <- | BACK(8) | EMPTY(8)   |                : background
    """

    const_crctp_to_kather19 = {
        'Tumor': ('TUM', 0), 'Stroma': ('STR', 1), 'Muscle': ('MUS', 1), 'Inflammatory': ('LYM', 2),
        'Debris': ('DEB', 3), 'Benign': ('NORM', 4), 'Complex Stroma': ('CSTR', 9),

    }

    data_root_dir_train = f'{data_root_dir}/Training'
    data_root_dir_test = f'{data_root_dir}/Testing'
    data_list = load_data_info(root_dir=data_root_dir_train, covert_dict=const_crctp_to_kather19)
    test_set = load_data_info(root_dir=data_root_dir_test, covert_dict=const_crctp_to_kather19)

    random.Random(5).shuffle(data_list)
    train_set = data_list[int(val_ratio * len(data_list)):]
    valid_set = data_list[:int(val_ratio / 2 * len(data_list))]

    print_number_of_sample(train_set, valid_set, test_set)
    return None, None, train_set+valid_set+test_set


#Colon Cacner Detection
def prepare_colon_kbsmc1_data(data_root_dir='datafolder/raw_data/colon_kbsmc', nr_classes=4):
    def load_data_info(pathname, gt_list, parse_label=True, label_value=0):
        file_list = glob(pathname)
        cancer_test = False
        if cancer_test:
            file_list_bn = glob(pathname.replace('*.jpg', '*0.jpg'))
            file_list = [elem for elem in file_list if elem not in file_list_bn]
            label_list = [int(file_path.split('_')[-1].split('.')[0])-1 for file_path in file_list]
        else:
            if parse_label:
                label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            else:
                label_list = [label_value for file_path in file_list]

        label_list = [gt_list[i] for i in label_list]
        #print(Counter(label_list))
        return list(zip(file_list, label_list))

    data_root_dir = f'{data_root_dir}/colon1'
    gt_list = {0: 0, 1: 0, 2: 1, 3: 1}

    set_1010711 = load_data_info('%s/1010711/*.jpg' % data_root_dir, gt_list)
    set_1010712 = load_data_info('%s/1010712/*.jpg' % data_root_dir, gt_list)
    set_1010713 = load_data_info('%s/1010713/*.jpg' % data_root_dir, gt_list)
    set_1010714 = load_data_info('%s/1010714/*.jpg' % data_root_dir, gt_list)
    set_1010715 = load_data_info('%s/1010715/*.jpg' % data_root_dir, gt_list)
    set_1010716 = load_data_info('%s/1010716/*.jpg' % data_root_dir, gt_list)
    wsi_00016 = load_data_info('%s/wsi_00016/*.jpg' % data_root_dir, gt_list, parse_label=True, label_value=0)  # benign exclusively
    wsi_00017 = load_data_info('%s/wsi_00017/*.jpg' % data_root_dir, gt_list, parse_label=True, label_value=0)  # benign exclusively
    wsi_00018 = load_data_info('%s/wsi_00018/*.jpg' % data_root_dir, gt_list, parse_label=True, label_value=0)  # benign exclusively

    train_set = set_1010711 + set_1010712 + set_1010713 + set_1010715 + wsi_00016
    valid_set = set_1010716 + wsi_00018
    test_set = set_1010714 + wsi_00017

    print_number_of_sample(train_set, valid_set, test_set)
    return train_set, valid_set, test_set


def prepare_colon_kbsmc2_data(data_root_dir='datafolder/raw_data/colon_kbsmc', nr_classes=4):
    def load_data_info(pathname, gt_list, nr_classes, parse_label=True, label_value=0):
        file_list = glob(pathname)
        cancer_test = False
        if cancer_test:
            file_list_bn = glob(pathname.replace('*.png', '*0.png'))
            file_list = [elem for elem in file_list if elem not in file_list_bn]
            label_list = [int(file_path.split('_')[-1].split('.')[0])-1 for file_path in file_list]
        else:
            if parse_label:
                label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            else:
                label_list = [label_value for file_path in file_list]
            label_list = [gt_list[i] for i in label_list]
            list_out = list(zip(file_list, label_list))
            list_out = [list_out[i] for i in range(len(list_out)) if list_out[i][1] < nr_classes]
            #print(Counter(label_list))
            return list_out

        #print(Counter(label_list))=
        return list(zip(file_list, label_list))
    data_root_dir = f'{data_root_dir}/colon2/'
    gt_list = {0: 3,  # "BN", #0
               1: 0,  # "TLS", #0
               2: 0,  # "TW", #2
               3: 1,  # "TM", #3
               4: 1,  # "TP", #4
               }

    test_set = load_data_info('%s/*/*/*.png' % data_root_dir, gt_list, nr_classes=nr_classes)

    print_number_of_sample(test_set=test_set)
    return None, None, test_set


def prepare_colon_digest2019_data(data_root_dir='datafolder/raw_data/colon_digestpath2019/', nr_classes=2):
    def load_data_info(pathname, parse_label=True, label_value=0):
        file_list = glob(pathname)
        if parse_label:
            label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        else:
            label_list = [label_value for file_path in file_list]

        return list(zip(file_list, label_list))

    train_set = load_data_info('%s/Train/*/*.jpg' % data_root_dir)
    valid_set = load_data_info('%s/Valid/*/*.jpg' % data_root_dir)
    test_set = load_data_info('%s/Test/*/*.jpg' % data_root_dir)

    print_number_of_sample(test_set=train_set+valid_set+test_set)
    return train_set, valid_set, test_set


def prepare_colon_chaoyang_data(data_root_dir='datafolder/raw_data/colon_chaoyang', nr_classes=2):
    def load_data_info(base_path, folder_name, gt_list):
        pathname = f'{base_path}/{folder_name}/*.JPG'
        file_list = glob(pathname, recursive=True)
        label_list = [int(file_path.split('-')[-1].split('.')[0]) for file_path in file_list]
        label_list = [gt_list[i] for i in label_list]
        list_out = list(zip(file_list, label_list))
        list_out = [list_out[i] for i in range(len(list_out)) if list_out[i][1] < nr_classes]
        return list_out

    gt_list = {0: 0,  # "BN", #0
               1: 3,  # "TLS", #0
               2: 1,  # "TW", #2
               3: 1,  # "TM", #3
               }

    train_set = load_data_info(base_path=data_root_dir, folder_name='train', gt_list=gt_list)
    valid_set = load_data_info(base_path=data_root_dir, folder_name='test', gt_list=gt_list)
    test_set = []

    print_number_of_sample(test_set=train_set+valid_set+test_set)
    return None, None, train_set+valid_set+test_set


#Prostate Gleason Grading
def prepare_prostate_panda_data(data_root_dir='datafolder/raw_data/prostate_panda', nr_classes=4):
    def load_data_info(pathname, parse_label=True, label_value=0):
        file_list = glob(pathname)
        if parse_label:
            label_list = [int(file_path.split('_')[-3]) - 2 for file_path in file_list]
        else:
            label_list = [label_value for file_path in file_list]
        return list(zip(file_list, label_list))

    #assert fold_idx < 3, "Currently only support 5 fold, each fold is 1 TMA"
    # 1000 ~ 6158
    # data_root_dir = '/data2/trinh/data/patches_data/PANDA/PANDA_512/PANDA_RA_patch/'
    train_set_1 = load_data_info('%s/1*/*.png' % data_root_dir)
    train_set_2 = load_data_info('%s/2*/*.png' % data_root_dir)
    train_set_3 = load_data_info('%s/3*/*.png' % data_root_dir)
    train_set_4 = load_data_info('%s/4*/*.png' % data_root_dir)
    train_set_5 = load_data_info('%s/5*/*.png' % data_root_dir)
    train_set_6 = load_data_info('%s/6*/*.png' % data_root_dir)

    train_set = train_set_1 + train_set_2 + train_set_4 + train_set_6
    valid_set = train_set_3
    test_set = train_set_5

    print_number_of_sample(train_set, valid_set, test_set)

    return train_set, valid_set, test_set


def prepare_prostate_ubc_data(data_root_dir='datafolder/raw_data/prostate_ubc', nr_classes=4): #TODO IS IT MATCHED?
    """
    prostate_miccai_2019_patches_690_80_step05
    class 0: 1811
    class 2: 7037
    class 3: 11431
    class 4: 292
    1811 BN, 7037 grade 3, 11431 grade 4, and 292 grade 5
    """

    def _split(dataset):  # train val test 80/10/10
        train, rest = train_test_split(dataset, train_size=0.8, shuffle=False, random_state=42)
        valid, test = train_test_split(rest, test_size=0.5, shuffle=False, random_state=42)
        return train, valid, test

    files = glob(f"{data_root_dir}/*/*.jpg")

    data_class0 = [data for data in files if int(data.split("_")[-1].split(".")[0]) == 0]
    data_class2 = [data for data in files if int(data.split("_")[-1].split(".")[0]) == 2]
    data_class3 = [data for data in files if int(data.split("_")[-1].split(".")[0]) == 3]
    data_class4 = [data for data in files if int(data.split("_")[-1].split(".")[0]) == 4]

    train_data0, validation_data0, test_data0 = _split(data_class0)
    train_data2, validation_data2, test_data2 = _split(data_class2)
    train_data3, validation_data3, test_data3 = _split(data_class3)
    train_data4, validation_data4, test_data4 = _split(data_class4)

    label_dict = {0: 0, 2: 1, 3: 2, 4: 3}

    train_path = train_data0 + train_data2 + train_data3 + train_data4
    valid_path = (validation_data0 + validation_data2 + validation_data3 + validation_data4)
    test_path = test_data0 + test_data2 + test_data3 + test_data4

    train_label = [int(path.split(".")[0][-1]) for path in train_path]
    valid_label = [int(path.split(".")[0][-1]) for path in valid_path]
    test_label = [int(path.split(".")[0][-1]) for path in test_path]

    test_label = [label_dict[k] for k in test_label]
    train_label = [label_dict[k] for k in train_label]
    valid_label = [label_dict[k] for k in valid_label]
    train_set = list(zip(train_path, train_label))
    valid_set = list(zip(valid_path, valid_label))
    test_set = list(zip(test_path, test_label))

    print_number_of_sample(test_set=train_set + valid_set + test_set)
    return None, None, train_set+valid_set+test_set


def prepare_prostate_kbsmc_data(data_root_dir='datafolder/raw_data/prostate_kbsmc', nr_classes=4):
    def load_data_info(pathname):
        file_list = glob(pathname)
        label_list = [convert_dict[file_path.split('/')[-2]] for file_path in file_list]
        #print(Counter(label_list))
        return list(zip(file_list, label_list))

    convert_dict = {'grade0': 0, 'grade3': 1, 'grade4': 2, 'grade5': 3}

    test_set = []

    test_set += load_data_info('%s/*/grade0/*.jpg' % data_root_dir)
    test_set += load_data_info('%s/*/grade3/*.jpg' % data_root_dir)
    test_set += load_data_info('%s/*/grade4/*.jpg' % data_root_dir)
    test_set += load_data_info('%s/*/grade5/*.jpg' % data_root_dir)
    # Counter({0: 2381})
    # Counter({1: 10815})
    # Counter({2: 7504})
    # Counter({3: 144})

    print_number_of_sample(test_set=test_set)
    return None, None, test_set


def prepare_prostate_aggc22_data(data_root_dir='datafolder/raw_data/prostate_aggc22', nr_classes=4):
    def load_data_info_from_df(df, gt_list, root_dir, nr_claases):
        file_list = []
        for idx in range(len(df)):
            folder_name = df.iloc[idx]['Name']
            subset_name = "_".join(folder_name.split('_')[:-1]) + "_image"
            if subset_name.split("_")[0] == 'Subset3':
                for scanner in ['Akoya', 'Philips', 'KFBio', 'Leica', 'Olympus', 'Zeiss']:
                    pathname = glob(
                        f"{data_root_dir}/{subset_name}/{scanner}/{folder_name}_{scanner}/*.jpg")
                    file_list.extend(pathname)
            else:
                pathname = glob(f'{root_dir}/{subset_name}/{folder_name}/*.jpg')
                file_list.extend(pathname)

        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        label_list = [gt_list[i] for i in label_list]
        list_out = list(zip(file_list, label_list))
        list_out = [list_out[i] for i in range(len(list_out)) if list_out[i][1] < nr_claases]

        return list_out

    gt_train_local = {1: 4,  # "Stroma",
                      2: 0,  # "Normal",
                      3: 1,  # "Grade3",
                      4: 2,  # "Grade4",
                      5: 3,  # "Grade5",
                      }

    data_root_dir = f'{data_root_dir}/'
    csv_path = f'{data_root_dir}/AGGC2022.csv'
    csv_subset = pd.read_csv(csv_path)
    train_df = csv_subset[csv_subset.TASK == 'Train']
    valid_df = csv_subset[csv_subset.TASK == 'Valid']
    test_df = csv_subset[csv_subset.TASK == 'Test']

    train_set = load_data_info_from_df(train_df, gt_train_local, data_root_dir, nr_classes)
    valid_set = load_data_info_from_df(valid_df, gt_train_local, data_root_dir, nr_classes)
    test_set = load_data_info_from_df(test_df, gt_train_local, data_root_dir, nr_classes)

    print_number_of_sample(test_set=train_set + valid_set + test_set)
    return None, None, train_set+valid_set+test_set


def main():
    # Breast
    print("Preparing breast_pcam data...")
    prepare_breast_pcam_data()

    print("Preparing breast_bach data...")
    prepare_breast_bach_data()

    print("Preparing breast_bracs data...")
    prepare_breast_bracs_data()

    # Colon Subtyping
    print("Preparing colon_kather19 data...")
    prepare_colon_kather19_data()

    print("Preparing colon_kather16 data...")
    prepare_colon_kather16_data()

    print("Preparing colon_crc_tp data...")
    prepare_colon_crc_tp_data()

    print("Preparing colon_kbsmc1 data...")
    prepare_colon_kbsmc1_data()

    print("Preparing colon_kbsmc2 data...")
    prepare_colon_kbsmc2_data()

    print("Preparing colon_digest2019 data...")
    prepare_colon_digest2019_data()

    print("Preparing colon_chaoyang data...")
    prepare_colon_chaoyang_data()

    # Prostate Gleason Grading
    print("Preparing prostate_panda data...")
    prepare_prostate_panda_data()

    print("Preparing prostate_ubc data...")
    prepare_prostate_ubc_data()

    print("Preparing prostate_kbsmc data...")
    prepare_prostate_kbsmc_data()

    print("Preparing prostate_aggc22 data...")
    prepare_prostate_aggc22_data()

if __name__ == "__main__":
    main()


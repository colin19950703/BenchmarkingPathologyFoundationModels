"""
get data loaders
"""
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from data_lib.RandAugment import rand_augment_transform
from PIL import Image
import data_lib.datalist as datalist


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class DatasetSerial(data.Dataset):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, pair_list, transform=None, target_transform=None):
        self.pair_list = pair_list
        self.transform = transform
        self.target_transform = target_transform
        self.num = self.__len__()

    def __getitem__(self, index):
        """
        Args:
            index (int): index
        Returns:
            tuple: (image, index, ...)
        """
        path, target = self.pair_list[index]
        image = pil_loader(path)

        # # image
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image

        return img, target



    def __len__(self):
        return len(self.pair_list)

def  get_histo_dataloader(opt, batch_size=128, num_workers=16, multiprocessing_distributed=False):
    """
    Data Loader for imagenet
    """
    def identity(img, **__):
        return img
    # transform

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms_list, valid_transforms_list = [], []
    train_transforms_list.extend([transforms.Resize((opt.image_size, opt.image_size))])
    valid_transforms_list.extend([transforms.Resize((opt.image_size, opt.image_size))])

    if opt.aug_train == 'RA':
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=100,
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )

        train_transforms_list.extend([
            # transforms.RandomResizedCrop(opt.image_size, scale=(opt.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            transforms.ToTensor(),
            normalize,
        ])

    valid_transforms_list.extend([
        transforms.ToTensor(),
        normalize
    ])

    train_transform = transforms.Compose(train_transforms_list)
    val_transform = transforms.Compose(valid_transforms_list)
    train_dataset, train_sampler, train_loader = None, None, None
    val_dataset, val_sampler, val_loader = None, None, None
    test_dataset, test_sampler, test_loader = None, None, None
    train_pairs, valid_pairs, test_pairs = getattr(datalist, f'prepare_{opt.dataset}_data')(nr_classes=opt.n_cls)

    train_dataset = DatasetSerial(train_pairs, transform=train_transform)
    val_dataset = DatasetSerial(valid_pairs, transform=val_transform)
    if test_pairs:
        test_dataset = DatasetSerial(test_pairs, transform=val_transform)

    if multiprocessing_distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        if test_dataset:
            test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=num_workers,
                              pin_memory=True,
                              sampler=train_sampler)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            sampler=val_sampler)

    if test_dataset:
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 sampler=test_sampler)



    print('train images: {}'.format(len(train_dataset)))
    print('val images: {}'.format(len(val_dataset)))
    if test_dataset:
        print('test images: {}'.format(len(test_dataset)))

    return train_loader, val_loader, test_loader, train_sampler


def get_histo_independent_dataloader(dataname, nr_classes, resize=True, image_size=224, aug_train='RA', batch_size=128, num_workers=16, multiprocessing_distributed=False):
    """
    Data Loader for imagenet
    """
    def identity(img, **__):
        return img
    # transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    valid_transforms_list = []

    valid_transforms_list.extend([transforms.Resize((image_size, image_size))])
    valid_transforms_list.extend([transforms.ToTensor(), normalize])
    val_transform = transforms.Compose(valid_transforms_list)

    val_dataset, val_sampler, val_loader = None, None, None
    test_dataset, test_sampler, test_loader = None, None, None

    _, valid_pairs, test_pairs = getattr(datalist, f'prepare_{dataname}_data')(nr_classes=nr_classes)

    if valid_pairs:
        val_dataset = DatasetSerial(valid_pairs, transform=val_transform)
        if multiprocessing_distributed:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader = DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 sampler=val_sampler)
        print('val images: {}'.format(len(val_dataset)))

    if test_pairs:
        test_dataset = DatasetSerial(test_pairs, transform=val_transform)
        if multiprocessing_distributed:
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 sampler=test_sampler)
        print('test images: {}'.format(len(test_dataset)))

    return val_loader, test_loader
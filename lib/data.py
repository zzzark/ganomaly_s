"""
LOAD DATA from file.
"""

import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder


##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    splits = ['train', 'test']
    drop_last_batch = {'train': True, 'test': False}
    shuffle = {'train': True, 'test': True}
    transform = transforms.Compose([transforms.Resize(opt.isize),
                                    transforms.CenterCrop(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    dataset = {x: ImageFolder(os.path.join(opt.dataroot, x), transform) for x in splits}
    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                 batch_size=opt.batchsize,
                                                 shuffle=shuffle[x],
                                                 num_workers=int(opt.workers),
                                                 drop_last=drop_last_batch[x],
                                                 worker_init_fn=(None if opt.manualseed == -1
                                                                 else lambda x: np.random.seed(opt.manualseed)))
                  for x in splits}
    return dataloader


def set_dataset(opt, i_latent, o_latent, labels, proportion=0.8):
    """

    Args:
        opt:
        i_latent:
        o_latent:
        labels:
        proportion:

    Returns:

    """
    i_latent = i_latent.cpu().reshape(len(i_latent), 1, int(opt.nz ** 0.5), -1)
    o_latent = o_latent.cpu().reshape(len(o_latent), 1, int(opt.nz ** 0.5), -1)
    labels = labels.cpu()

    if 0 < proportion < 1:
        prop = int(len(labels) // (1 / proportion))

    splits = ['i_train', 'i_test', 'o_train', 'o_test']
    drop_last_batch = {'i_train': True, 'i_test': False, 'o_train': True, 'o_test': False}
    shuffle = {'i_train': True, 'i_test': False, 'o_train': True, 'o_test': False}

    dataset = {'i_train': TensorDataset(i_latent[:prop], labels[:prop]),
               'o_train': TensorDataset(o_latent[:prop], labels[:prop]),
               'i_test': TensorDataset(i_latent[prop:], labels[prop:]),
               'o_test': TensorDataset(o_latent[prop:], labels[prop:]),
               }

    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                 batch_size=opt.batchsize,
                                                 shuffle=shuffle[x],
                                                 num_workers=int(opt.workers),
                                                 drop_last=drop_last_batch[x],
                                                 worker_init_fn=(None if opt.manualseed == -1
                                                                 else lambda x: np.random.seed(opt.manualseed)),
                                                 )
                  for x in splits}
    return dataloader

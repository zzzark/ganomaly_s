# -*- coding: utf-8 -*-
# @Time : 2020/3/16 下午10:01
# @Author : Yulin Liu
# @Site : 
# @File : train_z.py
# @Software: PyCharm

from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Ganomaly


def train():
    """ Training
    """

    ##
    # ARGUMENTS
    opt = Options().parse()
    ##
    # LOAD DATA
    dataloader = load_data(opt)
    ##
    # LOAD MODEL
    model = Ganomaly(opt, dataloader)
    ##
    # MODEL TEST
    res = model.test()

    model.z_train()




    print('AUC:%f\n' % res['AUC'])


if __name__ == '__main__':
    train()

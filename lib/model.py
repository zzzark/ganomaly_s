"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
import threading
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from lib.networks import NetG, NetD, NetC, weights_init
from lib.visualizer import Visualizer
from lib.loss import l1_loss, l2_loss, l3_loss
from lib.evaluate import evaluate

from lib.data import set_dataset


class BaseModel():
    """ Base Model for ganomaly
    """

    def __init__(self, opt, dataloader):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
        self.sqrtnz = int(self.opt.nz ** 0.5)

    ##
    def set_input(self, input: torch.Tensor):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])
                self.visualizer.save_fixed_real_s(self.fixed_input)

    def z_set_input(self, net, input: torch.Tensor):

        with torch.no_grad():
            if net == 'i':
                self.i_input.resize_(input[0].size()).copy_(input[0])
                self.i_gt.resize_(input[1].size()).copy_(input[1])
                self.i_label.resize_(input[1].size())
            if net == 'o':
                self.o_input.resize_(input[0].size()).copy_(input[0])
                self.o_gt.resize_(input[1].size()).copy_(input[1])
                self.o_label.resize_(input[1].size())

    ##
    def seed(self, seed_value):
        """ Seed 
        
        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_enc', self.err_g_enc.item())])

        return errors

    ##
    def z_get_errors(self, net):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """
        if net == 'i':
            errors = OrderedDict([
                ('err_i', self.err_i.item())])
        if net == 'o':
            errors = OrderedDict([
                ('err_o', self.err_o.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data
        # point
        fixed_reals = self.fixed_input.data
        # point
        return reals, fakes, fixed, fixed_reals

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD.pth' % (weight_dir))

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        if self.opt.strengthen:
            self.netd.train()  ## point
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            # self.optimize()
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                # point
                reals, fakes, fixed, fixed_reals = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed, fixed_reals)

        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch + 1, self.opt.niter))
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()
            res = self.test()
            if res['AUC'] > best_auc:
                best_auc = res['AUC']
                self.save_weights(self.epoch)
            self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name)

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """

        if self.opt.strengthen:
            self.netg.eval()
        with torch.no_grad():

            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32,
                                         device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,
                                         device=self.device)
            self.latent_i = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)
            self.latent_o = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)
            self.last_feature = torch.zeros(size=(
                len(self.dataloader['test'].dataset),
                list(self.netd.children())[0][-3].out_channels,
                list(self.netd.children())[0][-3].kernel_size[0],
                list(self.netd.children())[0][-3].kernel_size[1]
            ), dtype=torch.float32, device=self.device)

            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)
                _, features = self.netd(self.input)

                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                time_o = time.time()

                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(
                    error.size(0))
                self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.gt.reshape(
                    error.size(0))
                self.latent_i[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_i.reshape(
                    error.size(0), self.opt.nz)
                self.latent_o[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_o.reshape(
                    error.size(0), self.opt.nz)
                self.last_feature[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = features.reshape(
                    error.size(0),
                    list(self.netd.children())[0][-3].out_channels,
                    list(self.netd.children())[0][-3].kernel_size[0],
                    list(self.netd.children())[0][-3].kernel_size[1])

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _, _ = self.get_current_images()  # point add attribute fixed_real
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i + 1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i + 1), normalize=True)
            """
            data=[]
            feature = self.last_feature.cpu().numpy().reshape(self.last_feature.size()[0], -1)
            label = self.gt_labels.cpu().numpy().reshape(self.last_feature.size()[0], -1)

            features_dir = './features'
            file_name = 'features_map.csv'
            feature_path = os.path.join(features_dir, file_name + '.txt')
            import pandas as pd
            feature.tolist()
            label.tolist()
            test = pd.DataFrame(data=feature)
            test.to_csv("./1.csv", mode='a+', index=None, header=None)
            test = pd.DataFrame(data=label)
            test.to_csv("./2.csv", mode='a+', index=None, header=None)
            print('END')"""

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                    torch.max(self.an_scores) - torch.min(self.an_scores))

            # auc, eer = roc(self.gt_labels, self.an_scores)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])

            if self.opt.strengthen and self.opt.phase == 'test':
                # self.visualizer.display_scores_histo(self.epoch, self.an_scores, self.gt_labels)
                # self.visualizer.display_feature(self.last_feature, self.gt_labels)
                t0 = threading.Thread(target=self.visualizer.display_scores_histo,
                                      name='histogram ',
                                      args=(self.epoch, self.an_scores, self.gt_labels))
                t0.start()
                if self.opt.strengthen > 1:
                    t1 = threading.Thread(target=self.visualizer.display_feature,
                                          name='t-SNE visualizer',
                                          args=(self.last_feature, self.gt_labels))
                    t2 = threading.Thread(target=self.visualizer.display_latent,
                                          name='latent LDA visualizer',
                                          args=(self.latent_i, self.latent_o, self.gt_labels, 9, 1000, True))
                    t1.start()
                    t2.start()

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            if self.opt.classifier:
                self.z_dataloader = set_dataset(self.opt, self.latent_i, self.latent_o, self.gt_labels)
            return performance

    ##
    def z_save_weights(self, epoch):
        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netc_i.state_dict()},
                   '%s/netC_i.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netc_o.state_dict()},
                   '%s/netC_o.pth' % (weight_dir))

    ##
    def z_train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(">> Training model classifier")
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.z_train_one_epoch()
            res = self.z_test()
            if res['AUC'] > best_auc:
                best_auc = res['AUC']
                self.z_save_weights(self.epoch)
            self.visualizer.print_current_performance(res, best_auc)
        print(">> Training model %s.[Done]" % self.name)

    ##
    def z_train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netc_i.train()
        self.netc_o.train()
        epoch_iter = 0
        for i_data in tqdm(self.z_dataloader['i_train'], leave=False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.z_set_input('i', i_data)
            # self.optimize()
            self.z_optimize_params('i')

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.z_get_errors('i')
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.z_dataloader['i_train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            # if self.total_steps % self.opt.save_image_freq == 0:
            #     # point
            #     reals, fakes, fixed, fixed_reals = self.get_current_images()
            #     self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
            #     if self.opt.display:
            #         self.visualizer.display_current_images(reals, fakes, fixed, fixed_reals)

        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch + 1, self.opt.niter))

    ##
    def z_test(self):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """

        self.netd.eval()
        self.netc_i.eval()
        self.netc_o.eval()
        with torch.no_grad():

            # Load the weights of netg and netd.
            if self.opt.z_load_weights:
                d_path = "./output/{}/{}/train/weights/netD.pth".format(self.name.lower(), self.opt.dataset)
                i_path = "./output/{}/{}/train/weights/netC_i.pth".format(self.name.lower(), self.opt.dataset)
                o_path = "./output/{}/{}/train/weights/netC_o.pth".format(self.name.lower(), self.opt.dataset)
                d_pretrained_dict = torch.load(d_path)['state_dict']
                i_pretrained_dict = torch.load(i_path)['state_dict']
                o_pretrained_dict = torch.load(o_path)['state_dict']

                try:
                    self.netd.load_state_dict(d_pretrained_dict)
                    self.netc_i.load_state_dict(i_pretrained_dict)
                    self.netc_o.load_state_dict(o_pretrained_dict)
                except IOError:
                    raise IOError("net weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.i_scores = torch.zeros(size=(len(self.z_dataloader['i_test'].dataset),), dtype=torch.float32,
                                         device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.z_dataloader['i_test'].dataset),), dtype=torch.long,
                                         device=self.device)
            self.latent_i = torch.zeros(size=(len(self.z_dataloader['i_test'].dataset), self.opt.nz),
                                        dtype=torch.float32,
                                        device=self.device)
            self.latent_o = torch.zeros(size=(len(self.z_dataloader['i_test'].dataset), self.opt.nz),
                                        dtype=torch.float32,
                                        device=self.device)
            self.last_feature = torch.zeros(size=(
                len(self.dataloader['test'].dataset),
                list(self.netd.children())[0][-3].out_channels,
                list(self.netd.children())[0][-3].kernel_size[0],
                list(self.netd.children())[0][-3].kernel_size[1]
            ), dtype=torch.float32, device=self.device)

            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, i_data in enumerate(self.z_dataloader['i_test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.z_set_input('i', i_data)
                i_class = self.netc_i(self.i_input)
                o_class = self.netc_o(self.o_input)
                print("classifier of netc:", i_class, " ", o_class)
                self.fake, latent_i, latent_o = self.netg(self.input)
                _, features = self.netd(self.input)

                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                time_o = time.time()

                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(
                    error.size(0))
                self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.gt.reshape(
                    error.size(0))
                self.latent_i[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_i.reshape(
                    error.size(0), self.opt.nz)
                self.latent_o[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent_o.reshape(
                    error.size(0), self.opt.nz)
                self.last_feature[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = features.reshape(
                    error.size(0),
                    list(self.netd.children())[0][-3].out_channels,
                    list(self.netd.children())[0][-3].kernel_size[0],
                    list(self.netd.children())[0][-3].kernel_size[1])

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _, _ = self.get_current_images()  # point add attribute fixed_real
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i + 1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i + 1), normalize=True)
            """
            data=[]
            feature = self.last_feature.cpu().numpy().reshape(self.last_feature.size()[0], -1)
            label = self.gt_labels.cpu().numpy().reshape(self.last_feature.size()[0], -1)

            features_dir = './features'
            file_name = 'features_map.csv'
            feature_path = os.path.join(features_dir, file_name + '.txt')
            import pandas as pd
            feature.tolist()
            label.tolist()
            test = pd.DataFrame(data=feature)
            test.to_csv("./1.csv", mode='a+', index=None, header=None)
            test = pd.DataFrame(data=label)
            test.to_csv("./2.csv", mode='a+', index=None, header=None)
            print('END')"""

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                    torch.max(self.an_scores) - torch.min(self.an_scores))

            # auc, eer = roc(self.gt_labels, self.an_scores)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])

            if self.opt.strengthen and self.opt.phase == 'test':
                # self.visualizer.display_scores_histo(self.epoch, self.an_scores, self.gt_labels)
                # self.visualizer.display_feature(self.last_feature, self.gt_labels)
                t0 = threading.Thread(target=self.visualizer.display_scores_histo,
                                      name='histogram ',
                                      args=(self.epoch, self.an_scores, self.gt_labels))
                t0.start()
                if self.opt.strengthen > 1:
                    t1 = threading.Thread(target=self.visualizer.display_feature,
                                          name='t-SNE visualizer',
                                          args=(self.last_feature, self.gt_labels))
                    t2 = threading.Thread(target=self.visualizer.display_latent,
                                          name='latent LDA visualizer',
                                          args=(self.latent_i, self.latent_o, self.gt_labels, 9, 1000, True))
                    t1.start()
                    t2.start()

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            if self.opt.classifier:
                self.z_dataloader = set_dataset(self.opt, self.latent_i, self.latent_o, self.gt_labels)
            return performance


##
class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self):
        return 'Ganomaly'

    def __init__(self, opt, dataloader):
        super(Ganomaly, self).__init__(opt, dataloader)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)
        self.netc_i = NetC(self.opt).to(self.device)
        self.netc_o = NetC(self.opt).to(self.device)
        self.netc_i.apply(weights_init)
        self.netc_o.apply(weights_init)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")
        if self.opt.z_resume != '':
            print("\nLoading pre-trained z_networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.z_resume, 'netC_i.pth'))['epoch']
            self.netc_i.load_state_dict(torch.load(os.path.join(self.opt.z_resume, 'netC_i.pth'))['state_dict'])
            self.netc_o.load_state_dict(torch.load(os.path.join(self.opt.z_resume, 'netC_o.pth'))['state_dict'])
            print("\tDone.\n")

        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                 device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize),
                                       dtype=torch.float32, device=self.device)
        self.real_label = torch.ones(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Initialize input tensors for classifier
        self.i_input = torch.empty(size=(self.opt.batchsize, 1, self.sqrtnz, self.sqrtnz), dtype=torch.float32,
                                 device=self.device)
        self.o_input = torch.empty(size=(self.opt.batchsize, 1, int(self.opt.nz ** 0.5), int(self.opt.nz ** 0.5)),
                                   dtype=torch.float32,
                                   device=self.device)
        self.i_gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.i_label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.netc_i.train()
            self.netc_o.train()
            self.optimizer_i = optim.Adam(self.netc_i.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_o = optim.Adam(self.netc_o.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)

    ##
    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

    ##
    def backward_g(self):
        """ Backpropagate through netG
        """
        self.err_g_adv = self.l_adv(self.netd(self.input)[1], self.netd(self.fake)[1])
        self.err_g_con = self.l_con(self.fake, self.input)
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv * self.opt.w_adv + \
                     self.err_g_con * self.opt.w_con + \
                     self.err_g_enc * self.opt.w_enc
        self.err_g.backward(retain_graph=True)

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()

    ##
    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        self.netd.apply(weights_init)
        if (self.opt.strengthen != 1): print('   Reloading net d')

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        self.forward_d()

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # netd
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d.item() < 1e-5: self.reinit_d()

    ##
    def save_weights_z(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netC_i.state_dict()},
                   '%s/netC_i.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netC_o.state_dict()},
                   '%s/netC_o.pth' % (weight_dir))

    def forward_i(self):
        """ Forward propagate through netC_i
        """
        self.pred_abn_i = self.netc_i(self.i_input)

    def forward_o(self):
        """ Forward propagate through netC_o
        """
        self.pred_abn_o = self.netc_o(self.o_input)

    def backward_i(self):
        """ Backpropagate through netC_i
        """
        # Real - Fake Loss
        self.err_i = self.l_bce(self.pred_abn_i, self.real_label)

        # NetD Loss & Backward-Pass
        self.err_i.backward()

    def backward_o(self):
        """ Backpropagate through netC_o
        """
        # Real - Fake Loss
        self.err_o = self.l_bce(self.pred_abn_o, self.real_label)

        # NetD Loss & Backward-Pass
        self.err_o.backward()

    def reinit_i(self):
        """ Re-initialize the weights of netC_i
        """
        self.netc_i.apply(weights_init)
        if (self.opt.strengthen != 1): print('   Reloading net i')

    def reinit_o(self):
        """ Re-initialize the weights of netC_o
        """
        self.netc_o.apply(weights_init)
        if (self.opt.strengthen != 1): print('   Reloading net o')

    def z_optimize_params(self, net):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        if net == 'i':
            # Forward-pass
            self.forward_i()

            # Backward-pass
            # netc_i
            self.optimizer_i.zero_grad()
            self.backward_i()
            self.optimizer_i.step()

            if self.err_i.item() < 1e-5: self.reinit_i()
        if net == 'o':
            # Forward-pass
            self.forward_o()

            # Backward-pass
            # netc_o
            self.optimizer_o.zero_grad()
            self.backward_o()
            self.optimizer_o.step()

            if self.err_i.item() < 1e-5: self.reinit_o()


#
# class Classifier(Ganomaly):
#     """Classifier Class
#     """
#
#
#     @property
#     def name(self):
#         return 'Classifier'
#
#     def __init__(self, opt, dataloader):
#         super(Ganomaly, self).__init__(opt, dataloader)
#
#         self.z_dataloader = dataloader
#
#         self.netc_i = NetC(self.opt).to(self.device)
#         self.netc_o = NetC(self.opt).to(self.device)
#         self.netc_i.apply(weights_init)
#         self.netc_o.apply(weights_init)
#
#         if self.opt.isTrain:
#             self.netc_i.train()
#             self.netc_o.train()
#             self.optimizer_i = optim.Adam(self.netc_i.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
#             self.optimizer_o = optim.Adam(self.netc_o.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
#
#     def save_weights_z(self, epoch):
#         """Save netG and netD weights for the current epoch.
#
#         Args:
#             epoch ([int]): Current epoch number.
#         """
#
#         weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
#         if not os.path.exists(weight_dir): os.makedirs(weight_dir)
#
#         torch.save({'epoch': epoch + 1, 'state_dict': self.netC_i.state_dict()},
#                    '%s/netC_i.pth' % (weight_dir))
#         torch.save({'epoch': epoch + 1, 'state_dict': self.netC_o.state_dict()},
#                    '%s/netC_o.pth' % (weight_dir))
#
#     def forward_i(self):
#         """ Forward propagate through netC_i
#         """
#         self.pred_abn_i = self.netc_i(self.input)
#
#     def forward_o(self):
#         """ Forward propagate through netC_o
#         """
#         self.pred_abn_o = self.netc_o(self.input)
#
#     def backward_i(self):
#         """ Backpropagate through netC_i
#         """
#         # Real - Fake Loss
#         self.err_i = self.l_bce(self.pred_abn_i, self.real_label)
#
#         # NetD Loss & Backward-Pass
#         self.err_i.backward()
#
#     def backward_o(self):
#         """ Backpropagate through netC_o
#         """
#         # Real - Fake Loss
#         self.err_o = self.l_bce(self.pred_abn_o, self.real_label)
#
#         # NetD Loss & Backward-Pass
#         self.err_o.backward()
#
#     def reinit_i(self):
#         """ Re-initialize the weights of netC_i
#         """
#         self.netc_i.apply(weights_init)
#         if (self.opt.strengthen != 1): print('   Reloading net i')
#
#     def reinit_o(self):
#         """ Re-initialize the weights of netC_o
#         """
#         self.netc_o.apply(weights_init)
#         if (self.opt.strengthen != 1): print('   Reloading net o')
#
#     def optimize_params(self):
#         """ Forwardpass, Loss Computation and Backwardpass.
#         """
#         # Forward-pass
#         self.forward_i()
#         self.forward_o()
#
#         # Backward-pass
#         # netc_i
#         self.optimizer_i.zero_grad()
#         self.backward_i()
#         self.optimizer_i.step()
#
#         # netc_o
#         self.optimizer_o.zero_grad()
#         self.backward_o()
#         self.optimizer_o.step()
#         if self.err_i.item() < 1e-5: self.reinit_i()
#         if self.err_i.item() < 1e-5: self.reinit_o()
#
# ##
#     def z_train_one_epoch(self):
#         """ Train the model for one epoch.
#         """
#
#         self.netc_i.train()
#         self.netc_o.train()
#         epoch_iter = 0
#         for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
#             self.total_steps += self.opt.batchsize
#             epoch_iter += self.opt.batchsize
#
#             self.set_input(data)
#             # self.optimize()
#             self.optimize_params()
#
#             if self.total_steps % self.opt.print_freq == 0:
#                 errors = self.get_errors()
#                 if self.opt.display:
#                     counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
#                     self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)
#
#             if self.total_steps % self.opt.save_image_freq == 0:
#                 # point
#                 reals, fakes, fixed, fixed_reals = self.get_current_images()
#                 self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
#                 if self.opt.display:
#                     self.visualizer.display_current_images(reals, fakes, fixed, fixed_reals)
#
#         print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch + 1, self.opt.niter))
#         # self.visualizer.print_current_errors(self.epoch, errors)

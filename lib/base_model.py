# @Author : Ruikun Zheng
# @Description: Move the class `BaseModel` in `model.py` to `base_model.py`

import os
import threading
import time
from collections import OrderedDict

import numpy as np
import torch.utils.data
import torchvision.utils as vutils
from tqdm import tqdm

from lib.data import set_dataset
from lib.evaluate import evaluate
from lib.visualizer import Visualizer


# noinspection PyUnresolvedReferences,PyMethodMayBeStatic,PyUnboundLocalVariable,PyAttributeOutsideInit
class BaseModel:
    """ Base Model for ganomaly
    """

    def __init__(self, opt, dataloader):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initialize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
        self.sqrtnz = int(self.opt.nz ** 0.5)

    def set_input(self, input_: torch.Tensor):
        """ Set input and ground truth

        Args:
            input_ (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input_[0].size()).copy_(input_[0])
            self.gt.resize_(input_[1].size()).copy_(input_[1])
            self.label.resize_(input_[1].size())

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input_[0].size()).copy_(input_[0])
                self.visualizer.save_fixed_real_s(self.fixed_input)

    def z_set_input(self, net, input_: torch.Tensor):

        with torch.no_grad():
            if net == 'i':
                self.i_input.resize_(input_[0].size()).copy_(input_[0])
                self.i_gt.resize_(input_[1].size()).copy_(input_[1])
                self.i_label.resize_(input_[1].size())
                self.i_real_label.resize_(input_[1].size()).copy_(input_[1])
            if net == 'o':
                self.o_input.resize_(input_[0].size()).copy_(input_[0])
                self.o_gt.resize_(input_[1].size()).copy_(input_[1])
                self.o_label.resize_(input_[1].size())
                self.o_real_label.resize_(input_[1].size()).copy_(input_[1])

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

    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG.pth' % weight_dir)
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD.pth' % weight_dir)

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
            self.d_pred = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32,
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
                d_pred, features = self.netd(self.input)

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
                self.d_pred[i * self.opt.batchsize: i * self.opt.batchsize + d_pred.size(0)] = d_pred.reshape(
                    d_pred.size(0))
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
            test.to_csv("./feature.csv", mode='a+', index=None, header=None)
            test = pd.DataFrame(data=label)
            test.to_csv("./label.csv", mode='a+', index=None, header=None)
            print('END')
            """

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

    def z_save_weights(self, epoch):
        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netc_i.state_dict()},
                   '%s/netC_i.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netc_o.state_dict()},
                   '%s/netC_o.pth' % (weight_dir))

    def z_train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best = 0

        # Train for niter epochs.
        print(">> Training model classifier")
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.z_train_one_epoch()
            res = self.z_test()
            if res[self.opt.z_metric] > best:
                best = res[self.opt.z_metric]
                self.z_save_weights(self.epoch)
            self.visualizer.print_current_performance(res, best, self.opt.z_metric)
        self.visualizer.record_best(best, self.opt.z_metric, self.opt.abnormal_class, self.opt.manualseed, 'ganomaly_s')
        print(">> Training model %s.[Done]" % self.name)

    def z_train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netc_i.train()
        self.netc_o.train()
        epoch_iter = 0

        for data in tqdm(self.z_dataloader['i_train'], leave=False, total=len(self.z_dataloader['i_train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.z_set_input('i', data)
            # self.optimize()
            self.z_optimize_params('i')

        for data in tqdm(self.z_dataloader['o_train'], leave=False, total=len(self.z_dataloader['o_train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.z_set_input('o', data)
            # self.optimize()
            self.z_optimize_params('o')

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
            self.i_pred = torch.zeros(size=(len(self.z_dataloader['i_test'].dataset),), dtype=torch.float32,
                                      device=self.device)
            self.o_pred = torch.zeros(size=(len(self.z_dataloader['o_test'].dataset),), dtype=torch.float32,
                                      device=self.device)
            self.i_gt_labels = torch.zeros(size=(len(self.z_dataloader['i_test'].dataset),), dtype=torch.long,
                                           device=self.device)
            self.o_gt_labels = torch.zeros(size=(len(self.z_dataloader['o_test'].dataset),), dtype=torch.long,
                                           device=self.device)

            self.times = []
            self.total_steps = 0
            epoch_iter = 0

            for i, data in enumerate(self.z_dataloader['i_test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.z_set_input('i', data)
                i_pred = self.netc_i(self.i_input)
                time_o = time.time()

                self.i_pred[i * self.opt.batchsize: i * self.opt.batchsize + i_pred.size(0)] = i_pred.reshape(
                    i_pred.size(0))
                self.i_gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + i_pred.size(0)] = self.i_gt.reshape(
                    self.i_gt.size(0))

                self.times.append(time_o - time_i)

            for i, data in enumerate(self.z_dataloader['o_test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.z_set_input('o', data)
                o_pred = self.netc_o(self.o_input)

                time_o = time.time()

                self.o_pred[i * self.opt.batchsize: i * self.opt.batchsize + o_pred.size(0)] = o_pred.reshape(
                    o_pred.size(0))
                self.o_gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + o_pred.size(0)] = self.o_gt.reshape(
                    self.o_gt.size(0))

                self.times.append(time_o - time_i)

                # Save test images.

            # print(auprc(self.i_gt_labels.cpu(), self.i_pred.cpu()))
            # print((self.i_gt_labels.cpu()[:10], self.i_pred.cpu())[:10])

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # auc, eer = roc(self.gt_labels, self.an_scores)

            self.pred_c = self.i_pred.cpu() * self.opt.w_i + \
                          self.o_pred.cpu() * self.opt.w_o

            # print(self.pred_c[:5])
            # print(self.i_gt_labels[:5])

            scores = evaluate(self.o_gt_labels.cpu(), self.pred_c, self.opt.z_metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), (self.opt.z_metric, scores)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.z_dataloader['i_test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance


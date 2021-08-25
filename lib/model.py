"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from lib.base_model import BaseModel
from lib.loss import l2_loss
from lib.networks import NetG, NetD, NetC, weights_init


# noinspection PyUnresolvedReferences,PyMethodMayBeStatic,PyUnboundLocalVariable,PyAttributeOutsideInit
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
        if self.opt.classifier:
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
        self.o_gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.i_label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.o_label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.i_real_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32,
                                        device=self.device)
        self.o_real_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32,
                                        device=self.device)

        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            if self.opt.classifier:
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
        self.err_g = self.err_g_adv * self.opt.w_adv + self.err_g_con * self.opt.w_con + self.err_g_enc * self.opt.w_enc
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
        if self.opt.strengthen != 1: print('   Reloading net d')

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
                   '%s/netC_i.pth' % weight_dir)
        torch.save({'epoch': epoch + 1, 'state_dict': self.netC_o.state_dict()},
                   '%s/netC_o.pth' % weight_dir)

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
        self.err_i = self.l_bce(self.pred_abn_i, self.i_real_label)

        # NetD Loss & Backward-Pass
        self.err_i.backward()

    def backward_o(self):
        """ Backpropagate through netC_o
        """
        # Real - Fake Loss
        self.err_o = self.l_bce(self.pred_abn_o, self.o_real_label)

        # NetD Loss & Backward-Pass
        self.err_o.backward()

    def reinit_i(self):
        """ Re-initialize the weights of netC_i
        """
        self.netc_i.apply(weights_init)
        if self.opt.strengthen != 1: print('   Reloading net i')

    def reinit_o(self):
        """ Re-initialize the weights of netC_o
        """
        self.netc_o.apply(weights_init)
        if self.opt.strengthen != 1: print('   Reloading net o')

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

            if self.err_o.item() < 1e-5: self.reinit_o()


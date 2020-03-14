""" This file contains Visualizer class based on Facebook's visdom.

Returns:
    Visualizer(): Visualizer class to display plots and images
"""

##
import os
import time
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import plotly.figure_factory as ff


##
class Visualizer():
    """ Visualizer wrapper based on Visdom.

    Returns:
        Visualizer: Class file.
    """

    # pylint: disable=too-many-instance-attributes
    # Reasonable.

    ##
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.win_size = 256
        self.name = opt.name
        self.opt = opt
        if self.opt.display:
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port)

        # --
        # Dictionaries for plotting data and results.
        self.plot_data = None
        self.plot_res = None

        # --
        # Path to train and test directories.
        self.img_dir = os.path.join(opt.outf, opt.name, 'train', 'images')
        self.tst_img_dir = os.path.join(opt.outf, opt.name, 'test', 'images')
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.tst_img_dir):
            os.makedirs(self.tst_img_dir)
        # --
        # Log file.
        self.log_name = os.path.join(opt.outf, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    ##
    @staticmethod
    def normalize(inp):
        """Normalize the tensor

        Args:
            inp ([FloatTensor]): Input tensor

        Returns:
            [FloatTensor]: Normalized tensor.
        """
        return (inp - inp.min()) / (inp.max() - inp.min() + 1e-5)

    ##
    def plot_current_errors(self, epoch, counter_ratio, errors):
        """Plot current errros.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            errors (OrderedDict): Error for the current epoch.
        """

        if not hasattr(self, 'plot_data') or self.plot_data is None:
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss'
            },
            win=5
        )

    ##
    def plot_performance(self, epoch, counter_ratio, performance):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            performance (OrderedDict): Performance for the current epoch.
        """
        if not hasattr(self, 'plot_res') or self.plot_res is None:
            self.plot_res = {'X': [], 'Y': [], 'legend': list(performance.keys())}
        self.plot_res['X'].append(epoch + counter_ratio)
        self.plot_res['Y'].append([performance[k] for k in self.plot_res['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_res['X'])] * len(self.plot_res['legend']), 1),
            Y=np.array(self.plot_res['Y']),
            opts={
                'title': self.name + 'Performance Metrics',
                'legend': self.plot_res['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Stats'
            },
            win=6
        )

    ##
    def print_current_errors(self, epoch, errors):
        """ Print current errors.

        Args:
            epoch (int): Current epoch.
            errors (OrderedDict): Error for the current epoch.
            batch_i (int): Current batch
            batch_n (int): Total Number of batches.
        """
        # message = '   [%d/%d] ' % (epoch, self.opt.niter)
        message = '   Loss: [%d/%d] ' % (epoch, self.opt.niter)
        for key, val in errors.items():
            message += '%s: %.3f ' % (key, val)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    ##
    def print_current_performance(self, performance, best):
        """ Print current performance results.

        Args:
            performance ([OrderedDict]): Performance of the model
            best ([int]): Best performance.
        """
        message = '   '
        for key, val in performance.items():
            message += '%s: %.3f ' % (key, val)
        message += 'max AUC: %.3f' % best

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    ##
    def display_current_images(self, reals, fakes, fixed):
        """ Display current images.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        reals = self.normalize(reals.cpu().numpy())
        fakes = self.normalize(fakes.cpu().numpy())
        fixed = self.normalize(fixed.cpu().numpy())
        fixed_reals = self.normalize(self.fixed_input.cpu().numpy())

        self.vis.images(reals, win=1, opts={'title': 'Reals'})
        self.vis.images(fakes, win=2, opts={'title': 'Fakes'})
        self.vis.images(fixed, win=3, opts={'title': 'Fixed'})
        self.vis.images(fixed_reals, win=4, opts={'title': 'fixed_reals'})

    ##point
    def display_current_images(self, reals, fakes, fixed, fixed_reals):
        """ Display current images.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
            fixed_reals ([FloatTensor]): Fixed real Image
        """
        reals = self.normalize(reals.cpu().numpy())
        fakes = self.normalize(fakes.cpu().numpy())
        fixed = self.normalize(fixed.cpu().numpy())
        fixed_reals = self.normalize(fixed_reals.cpu().numpy())

        self.vis.images(reals, win=1, opts={'title': 'Reals'})
        self.vis.images(fakes, win=2, opts={'title': 'Fakes'})
        self.vis.images(fixed, win=3, opts={'title': 'Fixed'})
        self.vis.images(fixed_reals, win=4, opts={'title': 'fixed_reals'})

    def display_curent_images_info(self, fix_fake_label, fix_real_label):

        pass

    def display_current_images_test(self, reals, fakes, fixed, fixed_reals):
        """ Display current images.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
            fixed_reals ([FloatTensor]): Fixed real Image
        """
        reals = self.normalize(reals.cpu().numpy())
        fakes = self.normalize(fakes.cpu().numpy())
        fixed = self.normalize(fixed.cpu().numpy())
        fixed_reals = self.normalize(fixed_reals.cpu().numpy())

        self.vis.images(reals, win=1, opts={'title': 'Reals'})
        self.vis.images(fakes, win=2, opts={'title': 'Fakes'})
        self.vis.images(fixed, win=3, opts={'title': 'Fixed'})
        self.vis.images(fixed_reals, win=4, opts={'title': 'fixed_reals'})

    def save_current_images(self, epoch, reals, fakes, fixed):
        """ Save images for epoch i.

        Args:
            epoch ([int])        : Current epoch
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        vutils.save_image(reals, '%s/reals.png' % self.img_dir, normalize=True)
        vutils.save_image(fakes, '%s/fakes.png' % self.img_dir, normalize=True)
        vutils.save_image(fixed, '%s/fixed_fakes_%03d.png' % (self.img_dir, epoch + 1), normalize=True)

    def save_fixed_real_s(self, fixed_reals):
        vutils.save_image(fixed_reals, '%s/fixed_reals.png' % (self.img_dir), normalize=True)

    def display_scores_histo(self, epoch, scores, labels, win=7):
        """
        Display Histogram of the scores for both normal and abnormal test samples

        :param epoch:
        :param scores:
        :param labels:
        :return:
        """
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        abn_score = []
        nor_score = []

        for i, score in enumerate(scores, 0):
            if labels[i] == 1:
                abn_score.append(score)
            elif labels[i] == 0:
                nor_score.append(score)

        hist_data = [abn_score, nor_score]
        group_labels = ['Abnormal', 'Normal']

        fig = ff.create_distplot(hist_data, group_labels, bin_size=0.04)
        self.vis.plotlyplot(fig, win=win)

    def display_feature(self, features, labels, alg='t-SNE', win=8, iter=1000, message=False):
        """

        :param features: feature maps
        :param labels: labels
        :param alg:
        :param win: display window_id
        :param iter:  quantity of marker

        :return: None
        """
        labelss = labels[:iter].cpu().numpy()
        labels = [i + 1 for i in labelss]
        features = features[:iter].cpu().numpy().reshape(iter, -1)

        if alg == 't-SNE':
            from sklearn.manifold import TSNE

            tsne = TSNE(n_components=3, perplexity=40, learning_rate=140, n_iter=1000)
            if message: print("Start calculating t-SNE")
            tsne.fit_transform(features)
            if message: print("Finish calculation, Printing visual image")

            self.vis.scatter(X=tsne.embedding_, Y=labels, win=win, opts=dict(
                markersize=1,
                legend=['Normal', 'Abnormal']
            ))

    def display_latent(self, latent_is, latent_os, labels, win=9, iter=1000, message=False, n_components=2):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        if message:print('latent visualizer loading data')
        latent_i = latent_is[:iter].cpu().numpy()
        latent_o = latent_os[:iter].cpu().numpy()
        latent = np.concatenate([latent_i, latent_o])
        label = labels[:iter].cpu().numpy() + np.ones(1000)
        label = np.concatenate([label, label+np.ones(1000)*2])

        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(latent, label)
        X_new = lda.transform(latent)

        self.vis.scatter(X=X_new, Y=label, win=win, opts=dict(
            markersize=3,
            legend=['Normal_i', 'Abnormal_i', 'Normal_o', 'Abnormal_o']
        ))

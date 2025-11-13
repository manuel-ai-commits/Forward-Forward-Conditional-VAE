import math
import os

import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_image_generation_metrics import get_inception_score_and_fid

from src import utils, Layer_cnn, Layer_fc


class FFCCVAE(torch.nn.Module):
    """The model trained with Forward-Forward (FF)."""

    def __init__(self, opt):
        super(FFCCVAE, self).__init__()

        self.opt = opt

        if self.opt.device == "mps":
            torch.set_num_threads(8)

        self.bp = self.opt.model.bp
        self.bp_ff = self.opt.model.bp_ff
        if self.bp and self.bp_ff:
            raise ValueError("Both Backpropagation and Backpropagation with Forward-Forward loss are activated. Please choose one.")
        
        # Initial settings
        self.batch_size = self.opt.input.batch_size
        self.dataset = self.opt.input.dataset
        self.loss = self.opt.FFCCVAE.loss
        self.n_batches_vis = 10
        
        self.z_vis_list = []
        self.label_vis_list = []
        self.show_iters = 800
        self.ilt = self.opt.FFCCVAE.ilt
        
        # Model settings
        self.enc_channel_list = self.opt.FFCCVAE.enc_channel_list
        self.dec_channel_list = self.opt.FFCCVAE.dec_channel_list
        self.n_groups = self.opt.FFCCVAE.n_groups
        self.ClassGroups = self.opt.FFCCVAE.classgroups
        self.cfse = self.opt.FFCCVAE.CFSE
        self.maxpool = False
        self.beta = self.opt.FFCCVAE.beta
        self.latent_dim  = self.opt.FFCCVAE.latent_dim
        self.latent_shape = self.opt.FFCCVAE.latent_shape
        self.batchnorm_dec = self.opt.FFCCVAE.batchnorm_dec
        self.batchnorm_enc = self.opt.FFCCVAE.batchnorm_enc
        self.relu_dec = self.opt.FFCCVAE.relu_dec
        self.relu_enc = self.opt.FFCCVAE.relu_enc
        self.enc_kernel = self.opt.FFCCVAE.enc_kernel
        self.dec_kernel = self.opt.FFCCVAE.dec_kernel
        self.train_dec = self.opt.FFCCVAE.train_dec

        self.enc_model = nn.ModuleList()
        self.dec_model = nn.ModuleList()

        self._dataset_setup()

        # Dynamically add layers
        self.dims = [self.CNN_l1_dims]
        self.image_size = self.CNN_l1_dims[1]

        # setup the Encoder CNN layers
        self._enc_setup()
        # setup the Latent CNN layers
        self._latent_setup()
        # setup the Decoder CNN layers
        self._dec_setup()

        # Initialize the weights
        self._init_weights()

        

    def _init_weights(self):
        
        for m in self.enc_model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        for m in self.dec_model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def _dataset_setup(self):
         ## MNIST ##
        if self.opt.input.dataset == 'mnist':
            self.n_classes = 10

            if self.ilt == 'Fast':
                self.start_end = [[0, 3], [1, 4], [2, 5], [3, 6], [4, 20], [5, 20]]
            elif self.ilt == "Acc":
                # self.start_end = [[0, 6], [0, 11], [0, 16], [0, 21], [0, 20], [0, 20]]
                self.start_end = [[0, 2], [0, 3], [0, 4], [0, 5], [0, 20], [0, 20]]
            else:
                self.start_end = [[0, 50], [0, 100], [0, 150], [0, 200], [0, 250], [0, 300]]
            self.CNN_l1_dims = [1, 28, 28]  # Grayscale images, 28x28
            self.class_names= ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        ## Fashion-MNIST ##
        elif self.opt.input.dataset == 'fmnist':
            self.n_classes = 10
            if self.ilt == 'Fast':
                self.start_end = [[0, 7], [1, 10], [2, 13], [3, 16], [4, 30], [5, 40]]
            elif self.ilt == "Acc":
                self.start_end = [[0, 10], [0, 15], [0, 19], [0, 23], [0, 36], [0, 50]]
                # self.start_end = [[0, 6], [0, 9], [0, 11], [0, 14], [0, 30], [0, 40]]
            else:
                self.start_end = [[0, 10], [0, 20], [0, 30], [0, 40], [0, 50], [0, 60]]
            self.CNN_l1_dims = [1, 28, 28]
            self.class_names= ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

        ## CIFAR-10 ##
        elif self.opt.input.dataset == 'cifar10' or self.opt.input.dataset == 'cifar100':
            self.n_classes = 10 if self.opt.input.dataset == 'cifar10' else 100

            if self.ilt == 'Fast':
                self.start_end = [[0, 11], [2, 18], [4, 26], [6, 32], [8, 36], [10, 50]]
            elif self.ilt == "Acc":
                self.start_end = [[0, 11], [0, 16], [0, 21], [0, 25], [0, 36], [0, 50]]
                # self.start_end = [[0, 30], [0, 30], [30, 60], [30, 60], [60, 85], [60, 85], [85, 100], [85, 100]]
            else:
                self.start_end = [[0, 100], [0, 150], [0, 200], [0, 250], [0, 300], [0, 350]]
            self.CNN_l1_dims = [3, 32, 32]  # RGB images, 32x32
            self.class_names= ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
            self.path_fid_ref = os.path.join(os.getcwd(), 'fid_references', 'cifar10_train.npz')

        ## GTSRB ##
        elif self.opt.input.dataset == 'GTSRB':
            self.n_classes = len(self.opt.input.classes_allowed)

            if self.ilt == 'Fast':
                self.start_end = [[0, 11], [2, 18], [4, 26], [6, 32], [8, 36], [10, 50]]
            elif self.ilt == "Acc":
                self.start_end = [[0, 11], [0, 16], [0, 21], [0, 25], [0, 36], [0, 50]]
                # self.start_end = [[0, 30], [0, 30], [30, 60], [30, 60], [60, 85], [60, 85], [85, 100], [85, 100]]
            else:
                self.start_end = [[0, 100], [0, 150], [0, 200], [0, 250], [0, 300], [0, 350]]
            self.CNN_l1_dims = [3, 64, 64]  # RGB images, 30x30
            self.class_names= ["20", "30", "50", "60", "70", "80", "80 lifted", "100", "120", "no overtaking"]
            self.path_fid_ref = os.path.join(os.getcwd(), 'fid_references', 'GTSRB_train10.npz')
        
        ## SVHN ##
        elif self.opt.input.dataset == 'svhn':
            if self.ilt == 'Fast':
                self.start_end = [[0, 10], [2, 16], [4, 24], [6, 30], [8, 40], [10, 50]]
            elif self.ilt == "Acc":
                self.start_end = [[0, 10], [0, 15], [0, 20], [0, 25], [0, 35], [0, 50]]
            else:
                self.start_end = [[0, 10], [0, 20], [0, 30], [0, 40], [0, 50], [0, 60]]
            self.CNN_l1_dims = [3, 32, 32]  # SVHN images are RGB, 32x32
        else:
            raise ValueError(f"Unknown dataset: {self.opt.input.dataset}")
        
        return None

    def _enc_setup(self):
        for i, out_channels in enumerate(self.enc_channel_list):
            if self.ClassGroups:
                #ClassGroup case for CIFAR-100
                self.kernel = self.enc_kernel[i]
                if i % 2 == 1 and self.cfse:
                    group = self.n_groups[i]
                else:
                    group = 1

                if self.n_classes == self.n_groups[i]:
                    class_groups = None
                else:
                    class_groups = int(self.n_classes/self.n_groups[i])

                in_channels = self.dims[-1][0] # [[1, 28, 28]]
                layer = Layer_cnn.Conv_Layer(
                    self.dims[-1], opt = self.opt, in_channels = in_channels, out_channels = out_channels, 
                    num_classes = self.n_groups[i], act_fn = self.relu_enc[i],
                    kernel_size = self.kernel["kernel_size"], stride = self.kernel["stride"],
                    padding = self.kernel["padding"], maxpool = self.maxpool, batchnorm = self.batchnorm_enc[i],
                    groups = group, droprate = 0, loss_criterion = self.loss, ClassGroups = class_groups
                ).to(self.opt.device)
                self.enc_model.append(layer)
                self.dims.append(layer.next_dims)
                
                    
            else:
                class_groups = None

                self.kernel = self.enc_kernel[i]
                # if CSFE is activated, the group is the number of classes
                if i % 2 == 1 and self.cfse:
                    group = self.n_classes
                else:
                    group = 1

                in_channels = self.dims[-1][0] # [[1, 28, 28]]
                layer = Layer_cnn.Conv_Layer(
                                self.dims[-1], opt= self.opt, in_channels = in_channels,
                                out_channels = out_channels, num_classes = self.n_classes, act_fn = self.relu_enc[i],
                                kernel_size = self.kernel["kernel_size"], stride = self.kernel["stride"], 
                                padding = self.kernel["padding"], maxpool = self.maxpool, batchnorm = self.batchnorm_enc[i],
                                groups = group, droprate = 0, loss_criterion = self.loss, ClassGroups = class_groups
                                ).to(self.opt.device)
                self.enc_model.append(layer)
                self.dims.append(layer.next_dims)
                

    def _latent_setup(self):
        # Layer for latent space
        
        self.fc = Layer_fc.FC_LayerCW(
            self.enc_channel_list[-1]*self.latent_shape[0]* self.latent_shape[1], 1024, 
            relu = True, dropout = False, normalize =False, batchnorm =True
        ).to(self.opt.device)
        # self.fc_mu = nn.Linear(self.enc_channel_list[-1]*4, self.latent_dim)
        self.fc_mu_var = Layer_fc.FC_LayerCW(
            1024, 2*self.latent_dim, 
            relu = False, dropout = False, normalize =False, batchnorm =False
        ).to(self.opt.device)

        self.decoder_input_0 = Layer_fc.FC_LayerCW(
            self.latent_dim+self.n_classes, 1024, 
            relu = True, dropout = False, normalize =False, batchnorm =True
        ).to(self.opt.device)
        self.decoder_input_1 = Layer_fc.FC_LayerCW(
            1024, self.enc_channel_list[-1] * self.latent_shape[0]* self.latent_shape[0],
            relu = True, dropout = False, normalize = False, batchnorm = True
        ).to(self.opt.device)
        # self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.dims = [self.enc_model[-1].next_dims]

    def _dec_setup(self):
                
        

        for i, out_channels in enumerate(self.dec_channel_list):
            if self.ClassGroups:
                #ClassGroup case for CIFAR-100
                self.kernel = self.dec_kernel[i]
                if i % 2 == 1 and self.cfse and i!=len(self.dec_channel_list)-1:
                    group = self.n_groups[i]
                else:
                    group = 1

                if self.n_classes == self.n_groups[i]:
                    class_groups = None
                else:
                    class_groups = int(self.n_classes/self.n_groups[i])

                in_channels = self.dims[-1][0]
                layer = Layer_cnn.Conv_Layer_transpose(
                    self.dims[-1], opt= self.opt, in_channels=in_channels, out_channels=out_channels, num_classes=self.n_groups[i], act_fn=self.relu_dec[i],
                    kernel_size=self.kernel["kernel_size"], stride = self.kernel["stride"], padding=self.kernel["padding"], output_padding=self.kernel["output_padding"] , maxpool=self.maxpool,
                    batchnorm=self.batchnorm_dec[i],  groups=group, droprate=0, loss_criterion=self.loss, ClassGroups=  class_groups
                ).to(self.opt.device)
                self.dec_model.append(layer)
                self.dims.append(layer.next_dims)
            else:
                class_groups = None
                self.kernel = self.dec_kernel[i]
                # if CSFE is activated, the group is the number of classes
                if i % 2 == 1 and self.cfse and i!=len(self.dec_channel_list)-1:
                    group = self.n_classes
                else:
                    group = 1
                
                
            
                in_channels = self.dims[-1][0] # [[1, 28, 28]]
                layer = Layer_cnn.Conv_Layer_transpose(
                    self.dims[-1], opt= self.opt, in_channels= in_channels, out_channels= out_channels, num_classes= self.n_classes, act_fn= self.relu_dec[i],
                    kernel_size= self.kernel["kernel_size"], stride= self.kernel["stride"], padding= self.kernel["padding"], output_padding= self.kernel["output_padding"] ,
                    maxpool= self.maxpool, batchnorm= self.batchnorm_dec[i], groups= group, droprate=0, loss_criterion= self.loss, ClassGroups= class_groups
                ).to(self.opt.device)
                self.dec_model.append(layer)
                self.dims.append(layer.next_dims)

        self.classification_loss = nn.CrossEntropyLoss()

    def _N_classes(self):
        if self.opt.input.dataset == "cifar10" or self.opt.input.dataset == "mnist" or self.opt.input.dataset == "senti" or self.opt.input.dataset == "fmnist":
            return 10
        elif self.opt.input.dataset == "cifar100":
            return 100
        elif self.opt.input.dataset == "GTSRB":
            return len(self.opt.input.classes_allowed)

    def _calc_sparsity(self, z):
        return None

    def elbo_loss(self, recon_x, x, mu, log_var):
        """
        ELBO Optimization objective for gaussian posterior
        (reconstruction term + regularization term)
        """
        # MSE
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # CROSS ENTROPY
        # x = x * 255
        # x.data = x.data.int().long().view(-1)
        # # print(recon_x.shape)
        # recon_x = recon_x.permute(0, 2, 3, 4, 1)  # N * C * W * H
        # # print(recon_x.shape)
        # recon_x = recon_x.contiguous().view(-1, 256)
        # recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')


        # https://arxiv.org/abs/1312.6114 (Appendix B)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # Or this
        # kld_loss = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var) 
        # kld_loss = torch.sum(kld_loss).mul_(-0.5)

        # Return the combined loss (reconstruction + regularization)
        # print(f"MSE: {MSE}, KLD: {kld_loss}")
        return recon_loss/ x.size(0), kld_loss / x.size(0)
        # return -torch.mean(elbo)
    

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1)

        Params:
            mu (Tensor): Mean of Gaussian latent variables [B x D]
            logvar (Tensor): log-Variance of Gaussian latent variables [B x D]

        Returns: 
            z (Tensor) [B x D]
        """

        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z = eps.mul(sigma).add_(mu)

        return z
    
    ### CVAE MIRROR DECODER ###
    def _encoder_CVAE_mirror(self, inputs, labels= None, scalar_outputs = None, epoch = None, mode = "train", label = None):
        z = inputs
                
        if mode == "train":
            y = labels.long()
            y_l = y.clone()
            y_n = y[torch.randperm(z.size(0))]

            enc_out = []
            enc_out.append(z)

            start_end = self.start_end

            y = y.view(z.shape[0], 1)
            y_n = y_n.view(z.shape[0], 1)
            # positive
            onehot_y = torch.zeros((z.shape[0], self.n_classes), requires_grad=False, device= self.opt.device)
            onehot_y.scatter_(1, y, 1)
            # negative
            onehot_y_neg = torch.zeros((z.shape[0], self.n_classes), requires_grad=False, device= self.opt.device)
            onehot_y_neg.scatter_(1, y_n, 1)

            onehot_conv_y = onehot_y.view(z.shape[0], 1, 1, self.n_classes)*torch.ones((z.shape[0], z.shape[1], z.shape[2], self.n_classes), device= self.opt.device)
            h_pos = torch.cat((z, onehot_conv_y), dim=3)
            # print(h_pos.shape)
            for i, convlayer in enumerate(self.enc_model):
                s, e = start_end[i]
                if epoch in list(range(s, e)) and not self.bp:
                    # print("Shape before layer",i, "encoder", h_pos.shape)
                    # print("Shape before layer",i, "encoder", y.shape)
                    h_pos, ff_loss = convlayer.forward_forward(h_pos, y_l)
                    # print(h_pos.shape)
                    # print("Shape after layer",i, "encoder", h_pos.shape)
                    scalar_outputs["Loss"] += ff_loss
                    if not self.bp_ff:
                        h_pos.detach()
                    enc_out.append(h_pos[:,:,:,:h_pos.shape[2]].clone())
                    # print(enc_out[-1].shape)
                else:
                    h_pos = convlayer.forward(h_pos)
                    enc_out.append(h_pos[:,:,:,:h_pos.shape[2]].clone().detach())
                    # print("Shape after layer",i, "encoder", h_pos.shape)
            
            

            enc_out.reverse()

            return h_pos, y, y_n, onehot_y, onehot_y_neg, enc_out, scalar_outputs

        elif mode== "test":
            activations = []

            if label is not None:
                # Ensure the label is valid
                
                if not (0 <= label < self.n_classes):
                    raise ValueError(f"Invalid label: {label}. Must be between 0 and {self.n_classes - 1}.")
                
                label = torch.full((z.shape[0], 1), label, dtype=torch.long,requires_grad=False, device=self.opt.device)
                label= label.long()
                label = label.view(z.shape[0], 1)
                
                # Generate one-hot for the selected label
                onehot_y = torch.zeros((z.shape[0], self.n_classes), requires_grad=False, device=self.opt.device)
                onehot_y.scatter_(1, label, 1)
            else:
                # No label (all zeros)
                # onehot_y = torch.zeros((z.shape[0], self.n_classes),requires_grad=False, device=self.opt.device)
                # Neutral labels (all 0.1)
                onehot_y = torch.full((z.shape[0], self.n_classes), 0.1, requires_grad=False, device=self.opt.device)

            # Reshape one-hot encoding for concatenation
            onehot_conv_y = onehot_y.view(z.shape[0], 1, 1, self.n_classes)
            onehot_conv_y = onehot_conv_y * torch.ones((z.shape[0], z.shape[1], z.shape[2], self.n_classes), device=self.opt.device)

            # Concatenate one-hot encoding with latent representation
            h_pos = torch.cat((z, onehot_conv_y), dim=3)
            
            for i, convlayer in enumerate(self.enc_model):
                h_pos = convlayer.forward(h_pos)
                activations.append(h_pos.view(h_pos.size(0), -1) )
                # print("Shape after layer",i, "encoder", h_pos.shape)

            return h_pos, onehot_y, activations, scalar_outputs
        

        

    def _latent_CVAE_mirror(self, h_pos, onehot_y, scalar_outputs, y= None, y_n= None, onehot_y_neg= None, activations = None, label = None, mode = "train"):
        # print("Shape before latent space", h_pos.shape)
        if mode == "train":
            
            if not self.bp:
                # print("Shape before latent space", h_pos.shape)
                h_pos = utils.overlay_y_on_x3d(h_pos, y)  # overlay label on smoothed layer and first linear relu
                h_neg = utils.overlay_y_on_x3d(h_pos, y_n)

                # Flatten
                # print("Shape before latent space", h_pos.shape)
                # print("Shape before latent space", h_pos.shape)
                h_pos, h_neg, nn_loss = self.fc.forward_forward(h_pos, h_neg, y)
                scalar_outputs["Loss"] += nn_loss
                if not self.bp_ff:
                    h_pos.detach()
                    h_neg.detach()
            else:
                h_pos = self.fc.forward(h_pos)


            mu_var = self.fc_mu_var.forward(h_pos)
            # split x in half
            mu = mu_var[:, :self.latent_dim]
            # sigma shouldn't be negative
            log_var = mu_var[:, self.latent_dim:]

            h_pos = self.reparameterize(mu, log_var)
            h_neg = torch.cat((h_pos, onehot_y_neg), dim=1)
            h_pos = torch.cat((h_pos, onehot_y), dim=1)
            # print("Shape after latent space", h_pos.shape)



            return h_pos, h_neg, mu, log_var, scalar_outputs

        elif mode == "test":

            if label is not None:
                y = torch.full((h_pos.size(0),), label, dtype=torch.long, device=h_pos.device)
                h_pos = utils.overlay_y_on_x3d(h_pos, y)
                h_pos = self.fc.forward(h_pos)
                activations.append(h_pos)
            
            else:
                h_pos = self.fc.forward(h_pos)
                
            mu_var = self.fc_mu_var.forward(h_pos)
            # split x in half
            mu = mu_var[:, :self.latent_dim]
            # sigma shouldn't be negative
            log_var = mu_var[:, self.latent_dim:]
            h_pos = self.reparameterize(mu, log_var)
            h_pos = torch.cat((h_pos, onehot_y), dim=1)
            # print("Shape after latent space", h_pos.shape)

            return h_pos, mu, log_var, activations, scalar_outputs

        
    def _decoder_CVAE_mirror(self, h_pos,  y=None, y_n=None, mu=None, log_var=None, enc_list=None, h_neg = None, scalar_outputs=None, epoch=None, mode = "train"):
        
        j=0
        if mode == "train":
            if not self.bp:
                utils.overlay_y_on_x(h_pos, y)  
                utils.overlay_y_on_x(h_neg, y_n)
                h_pos, h_neg, nn_loss= self.decoder_input_0.forward_forward(h_pos, h_neg, y)
                scalar_outputs["Loss"] += nn_loss
                if not self.bp_ff:
                    h_pos.detach()
                    h_neg.detach()


                h_pos = self.decoder_input_1.forward(h_pos)
                h_pos = h_pos.view(-1, self.enc_channel_list[-1], self.latent_shape[0], self.latent_shape[0])
                # Loss over the decoder fcs\
                rec_loss, kld_loss = self.elbo_loss(h_pos, enc_list[j], mu, log_var)
                scalar_outputs["Loss"] += rec_loss + kld_loss* self.beta
                if not self.bp_ff:
                    h_pos.detach()
                
            else:
                h_pos = self.decoder_input_0.forward(h_pos)
                h_pos = self.decoder_input_1.forward(h_pos)
                h_pos = h_pos.view(-1, self.enc_channel_list[-1], self.latent_shape[0], self.latent_shape[0])
            j+=1
            start_end = self.start_end
            
            for i, convlayer in enumerate(self.dec_model):
                s, e = start_end[i]
                if (epoch in list(range(s, e)) or epoch is None) and not self.bp:
                    h_pos = convlayer.forward(h_pos)
                    rec_loss, kld_loss = self.elbo_loss(h_pos, enc_list[j], mu, log_var)
                    scalar_outputs["Loss"] += rec_loss + kld_loss * self.beta  # Includes in backprop
                    if not self.bp_ff:
                        h_pos.detach()
                else:
                    # Not propagating loss for this layer, computing for information purposes
                    h_pos = convlayer.forward(h_pos)
                    rec_loss, kld_loss = self.elbo_loss(h_pos, enc_list[j], mu, log_var)
                    if i == len(self.dec_model)-1:
                        scalar_outputs["Loss"] += rec_loss + kld_loss * self.beta
                scalar_outputs["MSE_loss"] = rec_loss
                scalar_outputs["KLD_loss"] = kld_loss
                j+=1
        elif mode == "test":
            h_pos = self.decoder_input_0.forward(h_pos)
            h_pos = self.decoder_input_1.forward(h_pos)
            h_pos = h_pos.view(-1, self.enc_channel_list[-1], self.latent_shape[0], self.latent_shape[0])
            for i, convlayer in enumerate(self.dec_model):
                h_pos = convlayer.forward(h_pos)
                # print("Shape after layer",i, "decoder", h_pos.shape)
            
        
        
        
        if scalar_outputs is None:
            return h_pos
        else:
            return h_pos, scalar_outputs
        
    
    def forward(self, inputs, labels, epoch):
        scalar_outputs = {"Loss": torch.zeros(1, device=self.opt.device)}
        torch.autograd.set_detect_anomaly(True)
        
        z = inputs
        # print(z.max())
        # print(z.min())
        if epoch < self.train_dec and not self.bp:
            h_pos, y, y_n, onehot_y, onehot_y_neg, enc_out, scalar_outputs = self._encoder_CVAE_mirror(z, labels, scalar_outputs, epoch)
            
            return scalar_outputs
        else: 
            
            with torch.no_grad():
                h_pos, y, y_n, onehot_y, onehot_y_neg, enc_out, scalar_outputs = self._encoder_CVAE_mirror(z, labels, scalar_outputs, epoch)

            h_pos, h_neg, mu, log_var, scalar_outputs = self._latent_CVAE_mirror(h_pos,onehot_y, scalar_outputs, y, y_n, onehot_y_neg)
            
            h_pos, scalar_outputs = self._decoder_CVAE_mirror(h_pos, y, y_n,  mu, log_var, enc_out, h_neg, scalar_outputs, epoch)
            
            return scalar_outputs

        
    

    def predict(self, inputs, labels, visualize=False, scalar_outputs=None, num_generate=25, grid_size=0.05):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }
            z= inputs
            y = labels.long().view(z.shape[0], 1)

            # Reconstruction task
            with torch.no_grad():
                z = inputs
                h_pos, y, y_n, onehot_y, onehot_y_neg, enc_out, scalar_outputs = self._encoder_CVAE_mirror(z, labels, scalar_outputs)

                h_pos, h_neg, mu, log_var, scalar_outputs = self._latent_CVAE_mirror(h_pos,onehot_y, scalar_outputs, y, y_n, onehot_y_neg)

                self.z_vis_list.append(h_pos[:, :self.latent_dim])
                self.label_vis_list.append(labels)

                h_pos, scalar_outputs = self._decoder_CVAE_mirror(h_pos, y, y_n,  mu, log_var, enc_out, h_neg, scalar_outputs)
            # print(h_pos.min(), h_pos.max())
            # print(self.opt.input.dataset== "GTSRB")
            # if (0 <= h_pos.min() and h_pos.max() <= 1) and (self.opt.input.dataset == 'cifar10' or self.opt.input.dataset == 'cifar100' or self.opt.input.dataset == 'GTSRB'):
            #     # print("Output is in the range [0, 1].")
            #     (IS, IS_std), FID = get_inception_score_and_fid(h_pos, self.path_fid_ref, use_torch=True)
            #     scalar_outputs["IS"] = IS
            #     scalar_outputs["IS_std"] = IS_std
            #     scalar_outputs["FID"] = FID
            

            
            if visualize:
                self.n_images = min(inputs.shape[0], 5)

                # Visualization of Reconstruction
                print("Visualizing VAE Results")
                utils.visualize_autoencoder_results(inputs, h_pos, num_images=self.n_images)

                # Visualization of Mean-Reconstructed Images
                print("Visualizing VAE Mean-Reconstructed Images")
                utils.display_and_save_batch(
                    title="CVAE Reconstruction",
                    batch=h_pos, # h_pos[:self.n_images]
                    save=True,
                    display=True
                )

                # Generate conditioned images for all labels
                
                utils.generate_and_visualize(
                        self._decoder_CVAE_mirror,
                        self.opt.device,
                        n_classes=self.n_classes,
                        num_images= 100,
                        latent_dim=self.latent_dim
                    )
                
                # Generate conditioned images for all labels in each row
                utils.generate_and_visualize_1D(
                        self._decoder_CVAE_mirror,
                        self.opt.device,
                        class_names= self.class_names,
                        n_classes=self.n_classes,
                        num_images= 100,
                        latent_dim=self.latent_dim
                )
                
                # Print latent space given images
            
            if len(self.z_vis_list) == self.n_batches_vis:
                z_vis = torch.cat(self.z_vis_list, dim=0)
                labels = torch.cat(self.label_vis_list, dim=0)
                utils.visualize_latent_space(
                        z_vis, labels, 
                        latent_dim=self.latent_dim,
                        class_names=self.class_names, 
                        device=self.opt.device
                    )
                
                
            return scalar_outputs


        
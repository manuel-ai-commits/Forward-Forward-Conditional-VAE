import math
import os
import sys
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torchvision.transforms as transforms

from src import utils, Layer_cnn, Layer_fc


class FFCCVAE(torch.nn.Module):
    """The model trained with Forward-Forward (FF)."""

    def __init__(self, opt):
        super(FFCCVAE, self).__init__()

        self.opt= opt

        if self.opt.device== "mps":
            torch.set_num_threads(8)

        self.training_mode= str.lower(self.opt.training.training_mode)
        if self.training_mode not in ("ff", "bp", "bp_ff"):
            raise ValueError("Wrong training algorithm selected. Available: ff, bp, or bp_ff")

        # Initial settings
        self.batch_size= self.opt.input.batch_size
        self.dataset= str.lower(self.opt.input.dataset)
        self.loss= self.opt.FFCCVAE.loss
        
        self.ilt= self.opt.FFCCVAE.ilt
        
        # Encoder/Decoder architecture settings
        self.enc_channel_list = self.opt.FFCCVAE.enc_channel_list
        self.dec_channel_list = self.opt.FFCCVAE.dec_channel_list
        self.enc_kernel = self.opt.FFCCVAE.enc_kernel
        self.dec_kernel = self.opt.FFCCVAE.dec_kernel
        self.batchnorm_enc = self.opt.FFCCVAE.batchnorm_enc
        self.batchnorm_dec = self.opt.FFCCVAE.batchnorm_dec
        self.relu_enc = self.opt.FFCCVAE.relu_enc
        self.relu_dec = self.opt.FFCCVAE.relu_dec
        self.train_enc = self.opt.FFCCVAE.train_enc
        self.maxpool = False

        # Latent/config settings
        self.latent_dim = self.opt.FFCCVAE.latent_dim
        self.latent_shape = self.opt.FFCCVAE.latent_shape
        self.elbo_kind = str.lower(self.opt.training.elbo_kind)
        self.beta = self.opt.FFCCVAE.beta

        # Grouping/classification settings
        self.n_groups = self.opt.FFCCVAE.n_groups
        self.ClassGroups = self.opt.FFCCVAE.classgroups
        self.cfse = self.opt.FFCCVAE.CFSE

        self.enc_model= nn.ModuleList()
        self.dec_model= nn.ModuleList()

        if self.dataset== 'mnist':
            self._mnist_setup()
        elif self.dataset== 'fmnist':
            self._fmnist_setup()
        elif self.dataset in ('cifar10', 'cifar100'):
            self._cifar_setup()
        elif self.dataset== 'gtsrb':
            self._gtsrb_setup()
        elif self.dataset== 'svhn':
            self._svhn_setup()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        # Dynamically add layers
        self.dims= [self.CNN_l1_dims]

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

    def _mnist_setup(self):
         ## MNIST ##
        self.n_classes= 10

        if self.ilt== 'Fast':
            self.start_end= [[0, 3], [1, 4], [2, 5], [3, 6], [4, 20], [5, 20]]
        elif self.ilt== "Acc":
            self.start_end= [[0, 2], [0, 3], [0, 4], [0, 5], [0, 20], [0, 20]]
        else:
            self.start_end= [[0, 50], [0, 100], [0, 150], [0, 200], [0, 250], [0, 300]]
        self.CNN_l1_dims= [1, 28, 28]  # Grayscale images, 28x28
        self.class_names= ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    def _fmnist_setup(self):
        ## Fashion-MNIST ##
        self.n_classes= 10
        if self.ilt== 'Fast':
            self.start_end= [[0, 7], [1, 10], [2, 13], [3, 16], [4, 30], [5, 40]]
        elif self.ilt== "Acc":
            self.start_end= [[0, 10], [0, 15], [0, 19], [0, 23], [0, 36], [0, 50]]
        else:
            self.start_end= [[0, 10], [0, 20], [0, 30], [0, 40], [0, 50], [0, 60]]
        self.CNN_l1_dims= [1, 28, 28]
        self.class_names= ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    def _cifar_setup(self):
        ## CIFAR-10 ##
        self.n_classes= 10 if self.opt.input.dataset== 'cifar10' else 100

        if self.ilt== 'Fast':
            self.start_end= [[0, 11], [2, 18], [4, 26], [6, 32], [8, 36], [10, 50]]
        elif self.ilt== "Acc":
            self.start_end= [[0, 11], [0, 16], [0, 21], [0, 25], [0, 36], [0, 50]]
        else:
            self.start_end= [[0, 100], [0, 150], [0, 200], [0, 250], [0, 300], [0, 350]]
        self.CNN_l1_dims= [3, 32, 32]  # RGB images, 32x32
        self.class_names= ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.path_fid_ref= os.path.join(os.getcwd(), 'fid_references', 'cifar10_train.npz')
    def _gtsrb_setup(self):
        ## GTSRB ##
        self.n_classes= len(self.opt.input.classes_allowed)

        if self.ilt== 'Fast':
            self.start_end= [[0, 11], [2, 18], [4, 26], [6, 32], [8, 36], [10, 50]]
        elif self.ilt== "Acc":
            self.start_end= [[0, 11], [0, 16], [0, 21], [0, 25], [0, 36], [0, 50]]
            # self.start_end= [[0, 30], [0, 30], [30, 60], [30, 60], [60, 85], [60, 85], [85, 100], [85, 100]]
        else:
            self.start_end= [[0, 100], [0, 150], [0, 200], [0, 250], [0, 300], [0, 350]]
        self.CNN_l1_dims= [3, 64, 64]  # RGB images, 30x30
        self.class_names= ["20", "30", "50", "60", "70", "80", "80 lifted", "100", "120", "no overtaking"]
        self.path_fid_ref= os.path.join(os.getcwd(), 'fid_references', 'GTSRB_train10.npz')
    def _svhn_setup(self):
        if self.ilt== 'Fast':
            self.start_end= [[0, 10], [2, 16], [4, 24], [6, 30], [8, 40], [10, 50]]
        elif self.ilt== "Acc":
            self.start_end= [[0, 10], [0, 15], [0, 20], [0, 25], [0, 35], [0, 50]]
        else:
            self.start_end= [[0, 10], [0, 20], [0, 30], [0, 40], [0, 50], [0, 60]]
        self.CNN_l1_dims= [3, 32, 32]  # SVHN images are RGB, 32x32
    def _enc_setup(self):
        for i, out_channels in enumerate(self.enc_channel_list):
            if self.ClassGroups:
                #ClassGroup case for CIFAR-100
                kernel= self.enc_kernel[i]
                if i % 2== 1 and self.cfse:
                    group= self.n_groups[i]
                else:
                    group= 1

                if self.n_classes== self.n_groups[i]:
                    class_groups= None
                else:
                    class_groups= int(self.n_classes/self.n_groups[i])

                in_channels= self.dims[-1][0] # [[1, 28, 28]]
                layer= Layer_cnn.Conv_Layer(
                    self.dims[-1], opt= self.opt, in_channels= in_channels, out_channels= out_channels, 
                    num_classes= self.n_groups[i], act_fn= self.relu_enc[i],
                    kernel_size= kernel["kernel_size"], stride= kernel["stride"],
                    padding= kernel["padding"], maxpool= self.maxpool, batchnorm= self.batchnorm_enc[i],
                    groups= group, droprate= 0, loss_criterion= self.loss, ClassGroups= class_groups
                ).to(self.opt.device)
                self.enc_model.append(layer)
                self.dims.append(layer.next_dims)
                
                    
            else:
                class_groups= None

                kernel= self.enc_kernel[i]
                # if CSFE is activated, the group is the number of classes
                if i % 2== 1 and self.cfse:
                    group= self.n_classes
                else:
                    group= 1

                in_channels= self.dims[-1][0] # [[1, 28, 28]]
                layer= Layer_cnn.Conv_Layer(
                                self.dims[-1], opt= self.opt, in_channels= in_channels,
                                out_channels= out_channels, num_classes= self.n_classes, act_fn= self.relu_enc[i],
                                kernel_size= kernel["kernel_size"], stride= kernel["stride"], 
                                padding= kernel["padding"], maxpool= self.maxpool, batchnorm= self.batchnorm_enc[i],
                                groups= group, droprate= 0, loss_criterion= self.loss, ClassGroups= class_groups
                                ).to(self.opt.device)
                self.enc_model.append(layer)
                self.dims.append(layer.next_dims)        
    def _latent_setup(self):
        # Layer for latent space
        
        self.fc= Layer_fc.FC_LayerCW(
            self.enc_channel_list[-1]*self.latent_shape[0]* self.latent_shape[1], 1024, 
            relu= True, dropout= False, normalize=False, batchnorm=True
        ).to(self.opt.device)
        # self.fc_mu= nn.Linear(self.enc_channel_list[-1]*4, self.latent_dim)
        self.fc_mu_var= Layer_fc.FC_LayerCW(
            1024, 2*self.latent_dim, 
            relu= False, dropout= False, normalize=False, batchnorm=False
        ).to(self.opt.device)

        self.decoder_input_0= Layer_fc.FC_LayerCW(
            self.latent_dim+self.n_classes, 1024, 
            relu= True, dropout= False, normalize=False, batchnorm=True
        ).to(self.opt.device)
        self.decoder_input_1= Layer_fc.FC_LayerCW(
            1024, self.enc_channel_list[-1] * self.latent_shape[0]* self.latent_shape[0],
            relu= True, dropout= False, normalize= False, batchnorm= True
        ).to(self.opt.device)
        # self.up_sample= nn.UpsamplingNearest2d(scale_factor=2)
        
        self.dims= [self.enc_model[-1].next_dims]
    def _dec_setup(self):
                
        

        for i, out_channels in enumerate(self.dec_channel_list):
            if self.ClassGroups:
                #ClassGroup case for CIFAR-100
                kernel= self.dec_kernel[i]
                if i % 2== 1 and self.cfse and i!=len(self.dec_channel_list)-1:
                    group= self.n_groups[i]
                else:
                    group= 1

                if self.n_classes== self.n_groups[i]:
                    class_groups= None
                else:
                    class_groups= int(self.n_classes/self.n_groups[i])

                in_channels= self.dims[-1][0]
                layer= Layer_cnn.Conv_Layer_transpose(
                    self.dims[-1], opt= self.opt, in_channels=in_channels, out_channels= out_channels, num_classes= self.n_groups[i], act_fn= self.relu_dec[i],
                    kernel_size= kernel["kernel_size"], stride= kernel["stride"], padding= kernel["padding"], output_padding= kernel["output_padding"] , maxpool= self.maxpool,
                    batchnorm= self.batchnorm_dec[i],  groups= group, droprate= 0, loss_criterion= self.loss, ClassGroups= class_groups
                ).to(self.opt.device)
                self.dec_model.append(layer)
                self.dims.append(layer.next_dims)
            else:
                class_groups= None
                kernel= self.dec_kernel[i]
                # if CSFE is activated, the group is the number of classes
                if i % 2== 1 and self.cfse and i!=len(self.dec_channel_list)-1:
                    group= self.n_classes
                else:
                    group= 1
                
                
            
                in_channels= self.dims[-1][0] # [[1, 28, 28]]
                layer= Layer_cnn.Conv_Layer_transpose(
                    self.dims[-1], opt= self.opt, in_channels= in_channels, out_channels= out_channels, num_classes= self.n_classes, act_fn= self.relu_dec[i],
                    kernel_size= kernel["kernel_size"], stride= kernel["stride"], padding= kernel["padding"], output_padding= kernel["output_padding"] ,
                    maxpool= self.maxpool, batchnorm= self.batchnorm_dec[i], groups= group, droprate=0, loss_criterion= self.loss, ClassGroups= class_groups
                ).to(self.opt.device)
                self.dec_model.append(layer)
                self.dims.append(layer.next_dims)

    def _N_classes(self):
        if self.opt.input.dataset== "cifar10" or self.opt.input.dataset== "mnist" or self.opt.input.dataset== "senti" or self.opt.input.dataset== "fmnist":
            return 10
        elif self.opt.input.dataset== "cifar100":
            return 100
        elif self.opt.input.dataset== "GTSRB":
            return len(self.opt.input.classes_allowed)

    def elbo_loss(self, recon_x, x, mu, log_var):
        """
        ELBO Optimization objective for gaussian posterior
        (reconstruction term + regularization term)
        """
        # MSE
        if self.elbo_kind == "mse":
            recon_loss= F.mse_loss(recon_x, x, reduction='sum')
        elif self.elbo_kind == "ce":
            x= (x * 255).long()
            x.data= x.data.view(-1)
            # print(recon_x.shape)
            recon_x= recon_x.permute(0, 2, 3, 4, 1)  # N * C * W * H
            # print(recon_x.shape)
            recon_x= recon_x.contiguous().view(-1, 256)
            recon_loss= F.binary_cross_entropy(recon_x, x, reduction='sum')
        else:
            raise ValueError(f"Invalid elbo_kind: {self.elbo_kind}. Valid options are: 'mse', 'ce'")


        # https://arxiv.org/abs/1312.6114 (Appendix B)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss= -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Return the combined loss (reconstruction + regularization)
        # print(f"MSE: {MSE}, KLD: {kld_loss}")
        return recon_loss/ x.size(0), kld_loss / x.size(0)
        # return -torch.mean(elbo)
    def _reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1)

        Params:
            mu (Tensor): Mean of Gaussian latent variables [B x D]
            logvar (Tensor): log-Variance of Gaussian latent variables [B x D]

        Returns: 
            z (Tensor) [B x D]
        """

        sigma= torch.exp(0.5 * logvar)
        eps= torch.randn_like(mu)
        z= eps.mul(sigma).add_(mu)

        return z
    def _one_hot(self, y):
        return torch.zeros(
            (y.shape[0], self.n_classes), 
            requires_grad=False, device= self.opt.device).scatter_(1, y, 1)

    ### CVAE MIRROR DECODER ###
    """
        All sizes are described in the comments
    """

    # ============================== TRAIN ============================== #
    def _encoder_CVAE_train(self, z, y, scalar_outputs, epoch):                
        # a0 -> [batch_size, channels, height, width]
        # y -> [batch_size, 1]
        # y_l -> [batch_size, 1]
        # y_n -> [batch_size, 1]
        y_l= y.clone()
        y_n= y[torch.randperm(y.size(0))]

        # Store the activations, that will be used in the decoding phase
        enc_out= []
        enc_out.append(z)


        y= y.view(y.shape[0], 1)
        y_n= y_n.view(y_n.shape[0], 1)
        # One hot encoding for positive data
        # onehot_y -> [batch_size, num_classes]
        onehot_y= self._one_hot(y)
        # One hot encoding for negative data
        # onehot_y_neg -> [batch_size, num_classes]
        onehot_y_neg= self._one_hot(y_n)

        # This step broadcasts the one-hot encoded labels (of shape [batch_size, num_classes])
        # so that they can be concatenated along the channel dimension with the feature maps 'z'.
        # It changes the shape of onehot_y from [batch_size, num_classes] to
        # [batch_size, z.shape[1], z.shape[2], num_classes], matching the spatial size of 'z'.
        # onehot_conv_y -> [batch_size, channels, height, num_classes]
        onehot_conv_y = onehot_y.view(z.shape[0], 1, 1, self.n_classes).expand(
            -1, z.shape[1], z.shape[2], -1
        )
        # The concation is neeeded for concatenating the data to its label
        # h_pos -> [batch_size, channels, height, num_classes + width]
        h_pos= torch.cat((z, onehot_conv_y), dim=3)

        for i, convlayer in enumerate(self.enc_model):
            s, e= self.start_end[i]
            if s <= epoch < e and self.training_mode == "ff":
                h_pos, ff_loss= convlayer.forward_forward(h_pos, y_l)
                scalar_outputs["Loss"] += ff_loss
                h_pos.detach()
                enc_out.append(h_pos[:,:,:,:h_pos.shape[2]].clone())
            elif s <= epoch < e and self.training_mode == "bp_ff":
                h_pos, ff_loss= convlayer.forward_forward(h_pos, y_l)
                scalar_outputs["Loss"] += ff_loss
                enc_out.append(h_pos[:,:,:,:h_pos.shape[2]].clone())
            else:
                h_pos= convlayer.forward(h_pos)
                enc_out.append(h_pos[:,:,:,:h_pos.shape[2]].clone().detach())
        enc_out.reverse()

        return h_pos, y, y_n, onehot_y, onehot_y_neg, enc_out, scalar_outputs
    def _latent_CVAE_train(self, h_pos, onehot_y, scalar_outputs, y, y_n, onehot_y_neg):
        # h_pos -> [batch_size, channels_enc, height_enc, width_enc]

        
        if self.training_mode == "ff":
            # h_pos -> [batch_size, channels_enc, height_enc, width_enc]
            # h_neg -> [batch_size, channels_enc, height_enc, width_enc]
            h_pos= utils.overlay_y_on_x3d(h_pos, y)  # overlay label on smoothed layer and first linear relu
            h_neg= utils.overlay_y_on_x3d(h_pos, y_n)

            # Flatten
            # h_pos -> [batch_size, size_fc]
            # h_neg -> [batch_size, size_fc]
            h_pos, h_neg, nn_loss= self.fc.forward_forward(h_pos, h_neg, y)
            scalar_outputs["Loss"] += nn_loss
            h_pos.detach()
            h_neg.detach()
        elif self.training_mode == "bp_ff":
            # h_pos -> [batch_size, channels_enc, height_enc, width_enc]
            # h_neg -> [batch_size, channels_enc, height_enc, width_enc]
            h_pos= utils.overlay_y_on_x3d(h_pos, y)  # overlay label on smoothed layer and first linear relu
            h_neg= utils.overlay_y_on_x3d(h_pos, y_n)

            # Flatten
            # h_pos -> [batch_size, size_fc]
            # h_neg -> [batch_size, size_fc]
            h_pos, h_neg, nn_loss= self.fc.forward_forward(h_pos, h_neg, y)
            scalar_outputs["Loss"] += nn_loss
        elif self.training_mode == "bp":
            # h_pos -> [batch_size, size_fc]
            h_pos= self.fc.forward(h_pos)

        # mu_var -> [batch_size, 2*latent_dim]
        mu_var= self.fc_mu_var.forward(h_pos)
        # mu -> [batch_size, latent_dim]
        mu= mu_var[:, :self.latent_dim]
        # log_var -> [batch_size, latent_dim]
        log_var= mu_var[:, self.latent_dim:]

        # h_pos -> [batch_size, latent_dim]
        h_pos= self._reparameterize(mu, log_var)
        # h_neg -> [batch_size, latent_dim + num_classes]
        h_neg= torch.cat((h_pos, onehot_y_neg), dim=1)
        # h_pos -> [batch_size, latent_dim + num_classes]
        h_pos= torch.cat((h_pos, onehot_y), dim=1)

        return h_pos, h_neg, mu, log_var, scalar_outputs
    def _decoder_CVAE_train(self, h_pos,  y, y_n, mu, log_var, enc_activations, h_neg, scalar_outputs, epoch):
        # h_pos -> [batch_size, latent_dim + num_classes]
        # h_neg -> [batch_size, latent_dim + num_classes]
        # y -> [batch_size, 1]
        # y_n -> [batch_size, 1]
        # mu -> [batch_size, latent_dim]
        # log_var -> [batch_size, latent_dim]
        # scalar_outputs -> dict
        # epoch -> int
        if self.training_mode == "ff":
            # h_pos -> [batch_size, latent_dim + num_classes]
            # h_neg -> [batch_size, latent_dim + num_classes]
            h_pos= utils.overlay_y_on_x(h_pos, y)  
            h_neg= utils.overlay_y_on_x(h_neg, y_n)
            # h_pos -> [batch_size, size_fc]
            # h_neg -> [batch_size, size_fc]
            h_pos, h_neg, nn_loss= self.decoder_input_0.forward_forward(h_pos, h_neg, y)
            scalar_outputs["Loss"] += nn_loss
            h_pos.detach()
            h_neg.detach()

            # h_pos -> [batch_size, channels_enc * height_enc * width_enc]
            h_pos= self.decoder_input_1.forward(h_pos)
            # h_pos -> [batch_size, channels_enc, height_enc, width_enc]
            h_pos= h_pos.view(-1, self.enc_channel_list[-1], self.latent_shape[0], self.latent_shape[0])
            # Loss over the decoder fcs\
            rec_loss, kld_loss= self.elbo_loss(h_pos, enc_activations[0], mu, log_var)
            scalar_outputs["Loss"] += rec_loss + kld_loss* self.beta
            h_pos.detach()
        
        if self.training_mode == "bp_ff":
            # h_pos -> [batch_size, latent_dim + num_classes]
            # h_neg -> [batch_size, latent_dim + num_classes]
            h_pos= utils.overlay_y_on_x(h_pos, y)  
            # h_pos -> [batch_size, size_fc]
            # h_neg -> [batch_size, size_fc]
            h_pos= utils.overlay_y_on_x(h_neg, y_n)
            h_pos, h_neg, nn_loss= self.decoder_input_0.forward_forward(h_pos, h_neg, y)
            scalar_outputs["Loss"] += nn_loss

            # h_pos -> [batch_size, channels_enc * height_enc * width_enc]
            h_pos= self.decoder_input_1.forward(h_pos)
            # h_pos -> [batch_size, channels_enc, height_enc, width_enc]
            h_pos= h_pos.view(-1, self.enc_channel_list[-1], self.latent_shape[0], self.latent_shape[0])
            # Loss over the decoder fcs\
            rec_loss, kld_loss= self.elbo_loss(h_pos, enc_activations[0], mu, log_var)
            scalar_outputs["Loss"] += rec_loss + kld_loss* self.beta
            
        if self.training_mode == "bp":
            # h_pos -> [batch_size, latent_dim + num_classes]
            h_pos= self.decoder_input_0.forward(h_pos)
            # h_pos -> [batch_size, channels_enc * height_enc * width_enc]
            h_pos= self.decoder_input_1.forward(h_pos)
            # h_pos -> [batch_size, channels_enc, height_enc, width_enc]
            h_pos= h_pos.view(-1, self.enc_channel_list[-1], self.latent_shape[0], self.latent_shape[0])

        # Remove the first activation from the list, because it was already used for the decoder input
        enc_activations.pop(0)

        for i, convlayer in enumerate(self.dec_model):
            # h_pos -> [batch_size, channels_dec, height_dec, width_dec]
            s, e= self.start_end[i]
            if (s <= epoch < e or epoch is None) and self.training_mode == "ff":
                h_pos= convlayer.forward(h_pos)
                rec_loss, kld_loss= self.elbo_loss(h_pos, enc_activations[i], mu, log_var)
                scalar_outputs["Loss"] += rec_loss + kld_loss * self.beta  # Includes in backprop
                h_pos.detach()
            elif (s <= epoch < e or epoch is None) and self.training_mode == "bp_ff":
                h_pos= convlayer.forward(h_pos)
                rec_loss, kld_loss= self.elbo_loss(h_pos, enc_activations[i], mu, log_var)
                scalar_outputs["Loss"] += rec_loss + kld_loss * self.beta  # Includes in backprop
            else:
                # Not propagating loss for this layer, computing for information purposes
                h_pos= convlayer.forward(h_pos)
                rec_loss, kld_loss= self.elbo_loss(h_pos, enc_activations[i], mu, log_var)
                if i== len(self.dec_model)-1:
                    scalar_outputs["Loss"] += rec_loss + kld_loss * self.beta
            scalar_outputs["MSE_loss"]= rec_loss
            scalar_outputs["KLD_loss"]= kld_loss
        
        return h_pos, scalar_outputs
    
    # ============================== GEN ============================== #
    def _encoder_CVAE_generate(self, z, y):                
        # a0 -> [batch_size, channels, height, width]
        # y -> [batch_size, 1]
        # y_l -> [batch_size, 1]
        # y_n -> [batch_size, 1]
        y_l= y.clone()
        y_n= y[torch.randperm(y.size(0))]

        y= y.view(y.shape[0], 1)
        y_n= y_n.view(y_n.shape[0], 1)
        # One hot encoding for positive data
        # onehot_y -> [batch_size, num_classes]
        onehot_y= self._one_hot(y)
        # One hot encoding for negative data
        # onehot_y_neg -> [batch_size, num_classes]
        onehot_y_neg= self._one_hot(y_n)

        # This step broadcasts the one-hot encoded labels (of shape [batch_size, num_classes])
        # so that they can be concatenated along the channel dimension with the feature maps 'z'.
        # It changes the shape of onehot_y from [batch_size, num_classes] to
        # [batch_size, z.shape[1], z.shape[2], num_classes], matching the spatial size of 'z'.
        # onehot_conv_y -> [batch_size, channels, height, num_classes]
        onehot_conv_y = onehot_y.view(z.shape[0], 1, 1, self.n_classes).expand(
            -1, z.shape[1], z.shape[2], -1
        )
        # The concation is neeeded for concatenating the data to its label
        # h_pos -> [batch_size, channels, height, num_classes + width]
        h_pos= torch.cat((z, onehot_conv_y), dim=3)

        for i, convlayer in enumerate(self.enc_model):
            if self.training_mode in ("ff", "bp_ff"):
                h_pos, _ = convlayer.forward_forward(h_pos, y_l)
            elif self.training_mode == "bp":
                h_pos= convlayer.forward(h_pos)

        return h_pos, y, y_n, onehot_y, onehot_y_neg
    def _latent_CVAE_generate(self, h_pos, onehot_y,  onehot_y_neg, y, y_n):
        # h_pos -> [batch_size, channels_enc, height_enc, width_enc]

        
        if self.training_mode in ("ff", "bp_ff"):
            # h_pos -> [batch_size, channels_enc, height_enc, width_enc]
            # h_neg -> [batch_size, channels_enc, height_enc, width_enc]
            h_pos= utils.overlay_y_on_x3d(h_pos, y)  # overlay label on smoothed layer and first linear relu
            h_neg= utils.overlay_y_on_x3d(h_pos, y_n)

            # Flatten
            # h_pos -> [batch_size, size_fc]
            # h_neg -> [batch_size, size_fc]
            h_pos, h_neg, _= self.fc.forward_forward(h_pos, h_neg, y)
        elif self.training_mode == "bp":
            # h_pos -> [batch_size, size_fc]
            h_pos= self.fc.forward(h_pos)

        # mu_var -> [batch_size, 2*latent_dim]
        mu_var= self.fc_mu_var.forward(h_pos)
        # mu -> [batch_size, latent_dim]
        mu= mu_var[:, :self.latent_dim]
        # log_var -> [batch_size, latent_dim]
        log_var= mu_var[:, self.latent_dim:]

        # h_pos -> [batch_size, latent_dim]
        h_pos= self._reparameterize(mu, log_var)
        # h_neg -> [batch_size, latent_dim + num_classes]
        h_neg= torch.cat((h_pos, onehot_y_neg), dim=1)
        # h_pos -> [batch_size, latent_dim + num_classes]
        h_pos= torch.cat((h_pos, onehot_y), dim=1)

        return h_pos, h_neg
    def _decoder_CVAE_generate(self, h_pos, h_neg, y, y_n):

        # h_pos -> [batch_size, latent_dim + num_classes]
        # h_neg -> [batch_size, latent_dim + num_classes]
        # y -> [batch_size, 1]
        # y_n -> [batch_size, 1]
        # mu -> [batch_size, latent_dim]
        # log_var -> [batch_size, latent_dim]
        # scalar_outputs -> dict
        # epoch -> int
        if self.training_mode in ("ff", "bp_ff"):
            # h_pos -> [batch_size, latent_dim + num_classes]
            # h_neg -> [batch_size, latent_dim + num_classes]
            h_pos= utils.overlay_y_on_x(h_pos, y)  
            h_neg= utils.overlay_y_on_x(h_neg, y_n)
            # h_pos -> [batch_size, size_fc]
            # h_neg -> [batch_size, size_fc]
            h_pos, h_neg, _= self.decoder_input_0.forward_forward(h_pos, h_neg, y)

            # h_pos -> [batch_size, channels_enc * height_enc * width_enc]
            h_pos= self.decoder_input_1.forward(h_pos)
            # h_pos -> [batch_size, channels_enc, height_enc, width_enc]
            h_pos= h_pos.view(-1, self.enc_channel_list[-1], self.latent_shape[0], self.latent_shape[0])
            # Loss over the decoder fcs\
        if self.training_mode == "bp":
            # h_pos -> [batch_size, latent_dim + num_classes]
            h_pos= self.decoder_input_0.forward(h_pos)
            # h_pos -> [batch_size, channels_enc * height_enc * width_enc]
            h_pos= self.decoder_input_1.forward(h_pos)
            # h_pos -> [batch_size, channels_enc, height_enc, width_enc]
            h_pos= h_pos.view(-1, self.enc_channel_list[-1], self.latent_shape[0], self.latent_shape[0])

        # Remove the first activation from the list, because it was already used for the decoder input

        for i, convlayer in enumerate(self.dec_model):
            # h_pos -> [batch_size, channels_dec, height_dec, width_dec]
            h_pos= convlayer.forward(h_pos)
        
        return h_pos
    
    # ============================== TEST ============================== #
    def _encoder_CVAE_test(self, z, label: int = None, scalar_outputs= None):                
        activations= []

        if label is not None:
            # Ensure the label is valid
            if not 0 <= label < self.n_classes:
                raise ValueError(f"Invalid label: {label}. Must be between 0 and {self.n_classes - 1}.")
            
            label= torch.full((z.shape[0], 1), label, dtype=torch.long,requires_grad=False, device=self.opt.device)
            label= label.long()
            label= label.view(z.shape[0], 1)
            
            # Generate one-hot for the selected label
            onehot_y= self._one_hot(label)
        else:
            # No label (all zeros)
            # onehot_y= torch.zeros((z.shape[0], self.n_classes),requires_grad=False, device=self.opt.device)
            # Neutral labels (all 0.1)
            onehot_y= torch.full((z.shape[0], self.n_classes), 0.1, requires_grad=False, device=self.opt.device)

        # Reshape one-hot encoding for concatenation
        onehot_conv_y = onehot_y.view(z.shape[0], 1, 1, self.n_classes).expand(
            -1, z.shape[1], z.shape[2], -1
        )

        # Concatenate one-hot encoding with latent representation
        h_pos= torch.cat((z, onehot_conv_y), dim=3)
        
        for i, convlayer in enumerate(self.enc_model):
            h_pos= convlayer.forward(h_pos)
            activations.append(h_pos.view(h_pos.size(0), -1) )

        return h_pos, onehot_y, activations, scalar_outputs
    def _latent_CVAE_test(self, h_pos, onehot_y, scalar_outputs, label= None):
        # h_pos -> [batch_size, channels_enc, height_enc, width_enc]
        if label is not None:
            y= torch.full((h_pos.size(0),), label, dtype=torch.long, device=h_pos.device)
            h_pos= utils.overlay_y_on_x3d(h_pos, y)
            # h_pos -> [batch_size, size_fc]
            h_pos= self.fc.forward(h_pos)
        
        else:
            h_pos= self.fc.forward(h_pos)
            
        # mu_var -> [batch_size, 2*latent_dim]
        mu_var= self.fc_mu_var.forward(h_pos)
        # mu -> [batch_size, latent_dim]
        mu= mu_var[:, :self.latent_dim]
        # log_var -> [batch_size, latent_dim]
        log_var= mu_var[:, self.latent_dim:]
        # h_pos -> [batch_size, latent_dim]
        h_pos= self._reparameterize(mu, log_var)
        # h_pos -> [batch_size, latent_dim + num_classes]
        h_pos= torch.cat((h_pos, onehot_y), dim=1)

        return h_pos
    def _decoder_CVAE_test(self, h_pos):
        # h_pos -> [batch_size, latent_dim + num_classes]
        # h_pos -> [batch_size, size_fc]
        h_pos= self.decoder_input_0.forward(h_pos)
        # h_pos -> [batch_size, channels_enc * height_enc * width_enc]
        h_pos= self.decoder_input_1.forward(h_pos)
        # h_pos -> [batch_size, channels_enc, height_enc, width_enc]
        h_pos= h_pos.view(-1, self.enc_channel_list[-1], self.latent_shape[0], self.latent_shape[0])
        for i, convlayer in enumerate(self.dec_model):
            h_pos= convlayer.forward(h_pos)
            
        return h_pos
        
    
    def forward(self, z, labels, epoch):
        scalar_outputs= {"Loss": torch.zeros(1, device=self.opt.device)}
        # torch.autograd.set_detect_anomaly(True)
        if epoch < self.train_enc and self.training_mode in ("ff", "bp_ff"):
            h_pos, y, y_n, onehot_y, onehot_y_neg, enc_activations, scalar_outputs= self._encoder_CVAE_train(z, labels, scalar_outputs, epoch)
            return scalar_outputs
        else: 
            if self.training_mode in ("ff", "bp_ff"):
                 with torch.no_grad():
                    h_pos, y, y_n, onehot_y, onehot_y_neg, enc_activations, scalar_outputs= self._encoder_CVAE_train(z, labels, scalar_outputs, epoch)
            else:
                h_pos, y, y_n, onehot_y, onehot_y_neg, enc_activations, scalar_outputs= self._encoder_CVAE_train(z, labels, scalar_outputs, epoch)

            h_pos, h_neg, mu, log_var, scalar_outputs= self._latent_CVAE_train(h_pos,onehot_y, scalar_outputs, y, y_n, onehot_y_neg)
            h_pos, scalar_outputs= self._decoder_CVAE_train(h_pos, y, y_n, mu, log_var, enc_activations, h_neg, scalar_outputs, epoch)
            
            return scalar_outputs

    def generation_reconstruction(self, inputs, labels, visualize=False):
        z= inputs
        n_batches_vis= 10
        z_vis_list= []
        label_vis_list = []
        # Reconstruction task
        with torch.no_grad():
            h_pos, y, y_n, onehot_y, onehot_y_neg= self._encoder_CVAE_generate(z, labels)
            h_pos, h_neg= self._latent_CVAE_generate(h_pos, onehot_y, onehot_y_neg, y, y_n)
            z_vis_list.append(h_pos[:, :self.latent_dim])
            label_vis_list.append(labels)
            
            h_pos= self._decoder_CVAE_generate(h_pos, h_neg, y, y_n)

        if visualize:
            n_images= min(z.shape[0], 5)

            # Visualization of Reconstruction
            print("Visualizing VAE Results")
            utils.visualize_autoencoder_results(inputs, h_pos, num_images=n_images)

            # Visualization of Mean-Reconstructed Images
            print("Visualizing VAE Mean-Reconstructed Images")
            utils.display_and_save_batch(
                title="CVAE Reconstruction",
                batch=h_pos, # h_pos[:n_images]
                save=True,
                display=True
            )

            # Generate conditioned images for all labels
            utils.generate_and_visualize(
                self._decoder_CVAE_test,
                self.opt.device,
                n_classes=self.n_classes,
                num_images= 100,
                latent_dim=self.latent_dim
            )
            
            # Generate conditioned images for all labels in each row
            utils.generate_and_visualize_1D(
                self._decoder_CVAE_test,
                self.opt.device,
                class_names= self.class_names,
                n_classes=self.n_classes,
                num_images= 100,
                latent_dim=self.latent_dim
            )
            
            # Print latent space given images
        
        if len(z_vis_list)== n_batches_vis:
            z_vis= torch.cat(z_vis_list, dim=0)
            labels= torch.cat(label_vis_list, dim=0)
            utils.visualize_latent_space(
                z_vis, labels, 
                latent_dim=self.latent_dim,
                class_names=self.class_names, 
                device=self.opt.device
            )

    def predict(self, inputs, labels):
        scalar_outputs= {"Loss": torch.zeros(1, device=self.opt.device)}
        epoch = 0
        
        with torch.no_grad():
            # give epoch as 
            h_pos, y, y_n, onehot_y, onehot_y_neg, enc_activations, scalar_outputs= self._encoder_CVAE_train(inputs, labels, scalar_outputs, epoch)
            h_pos, h_neg, mu, log_var, scalar_outputs= self._latent_CVAE_train(h_pos,onehot_y, scalar_outputs, y, y_n, onehot_y_neg)
            h_pos, scalar_outputs= self._decoder_CVAE_train(h_pos, y, y_n, mu, log_var, enc_activations, h_neg, scalar_outputs, epoch)
            
        return scalar_outputs
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch import nn
import torch

class Softmax_CF(nn.Module):
    def __init__(self, flattened_size, opt, out_classes=10,
                 bias=True, device=None, dtype=None, droprate=0, lr=0.01):
        super(Softmax_CF, self).__init__()

        self.opt = opt
        self.linear = nn.Linear(flattened_size, out_classes).to(self.opt.device)
        self.dropout = torch.nn.Dropout(p=droprate)
        self.relu = torch.nn.ReLU()
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.opt = SGD(self.parameters(), lr=self.lr)
        self.ep_losses = []

    def forward(self, x):

        if len(x.shape) > 2:
            x = x.reshape(x.size(0), -1)

        x = self.dropout(x)
        # normalize magnitude activations of previous layer to forward
        # only orientation of activity vector norm 2 sum of squared activations
        # x = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        x = self.linear(x)
        y = F.softmax(x, dim=1)

        return y

    def setdropout(self, drop_rate):
        self.dropout = torch.nn.Dropout(p=drop_rate)

    def epoch_loss(self):
        epl_mean = torch.tensor(self.ep_losses).mean().item()
        # if abs(epl_mean - self.ep_losses[-1]) < 0.00001:
        #     self.lr_decay()
        #     print('lr decay, new lr = ', self.lr)
        self.ep_losses = []
        # print('ep losses', self.ep_losses)
        return epl_mean

    def train_classifier(self, x_pos, gt, show):

        out = self.forward(x_pos)

        self.opt.zero_grad()
        loss = self.criterion(out, gt)
        self.ep_losses.append(loss)

        if show:
            print('Layer Loss: {}'.format(loss))

        loss.backward()
        self.opt.step()

        return out.detach()


class FC_LayerCW(nn.Linear):
    def __init__(self, in_features, out_features, layer_number = False, dropout = False, batchnorm = False,
                 relu =False, normalize = False, bias=True, device=None, dtype=None, droprate=0):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        self.isdropout = dropout
        self.isrelu = relu
        self.isnormalize = normalize
        if self.isrelu:
            self.relu = torch.nn.ReLU()
        if self.isdropout:
            self.dropout = torch.nn.Dropout(p=droprate)
        self.lr = 0.02  # 0,01
        self.opt = Adam(self.parameters(), lr=self.lr, betas=(0.95, 0.999))
        self.threshold = 2.0
        self.ep_losses = []
        self.layer_number = layer_number
        self.isbatchnorm = batchnorm
        if self.isbatchnorm:
            self.conv_bn = nn.BatchNorm1d(out_features)

    def forward(self, x):

        if len(x.shape) > 2:
            x = x.reshape(x.size(0), -1)

        if self.isdropout:
            x = self.dropout(x)
        # normalize magnitude activations of previous layer to forward
        # only orientation of activity vector norm 2 sum of squared activations
        if self.isnormalize:
            x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
            x = torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0)
        else:
            x = torch.mm(x, self.weight.T) + self.bias.unsqueeze(0)
        
        if self.isbatchnorm:
            x = self.conv_bn(x)
        # find activity vector y
        if self.isrelu:
            x = self.relu(x)

        return x

    def setdropout(self, drop_rate):
        self.dropout = torch.nn.Dropout(p=drop_rate)

    def goodness_factor(self, y_pos, y_neg):
        """
        compute the goodness score.
        Math: \sum_{y}^2
        Ref: ``Let us suppose that the goodness function for a layer
               is simply the sum of the squares of the activities of
               the rectified linear neurons input (y) to that layer.``
        """

        # print("G_pos and G_neg")
        g_pos = y_pos.pow(2).mean(1)
        # print(g_pos.shape)
        g_neg = y_neg.pow(2).mean(1)
        # print(g_neg)


        return g_pos, g_neg

    def vis_features(self, pos_features, H):
        # Plot features, Positive and Negative
        # Create a plot of the number of features positive and negative

        pos_features = pos_features.reshape((H, H))
        print(pos_features.shape)
        N_columns = 1  # pos_features.shape[0]
        N_rows = 2

        fig, axs = plt.subplots(N_rows, N_columns, figsize=(5, 5))
        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 4, figsize=(8, 8))
        axs[0].set_title('Positive Features')
        axs[1].set_title('Copy')

        axs[0].imshow(pos_features, cmap='gray')
        axs[1].imshow(pos_features, cmap='gray')

        plt.show()

    def loss(self, g_pos, g_neg, sigmoid=True):

        # error is if goodness for positive data is low -(g_pos - threshold)
        # or if goodness for negative data is high +(gpos - threshold)
        errors = torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold])

        if sigmoid:
            loss = torch.sigmoid(errors).mean()
        else:
            loss = torch.log(1 + torch.exp(errors)).mean()

        return loss

    def lr_decay(self):
        # decrease learning rate if loss becomes low
        decay = 0.95
        self.lr *= decay
        self.opt = Adam(self.parameters(), lr=self.lr, betas=(0.95, 0.999))

    def epoch_loss(self):
        epl_mean = torch.tensor(self.ep_losses).mean().item()
        # if abs(epl_mean - self.ep_losses[-1]) < 0.00001:
        #     self.lr_decay()
        #     print('lr decay, new lr = ', self.lr)
        self.ep_losses = []
        # print('ep losses', self.ep_losses)
        return epl_mean

    def forward_forward(self, x_pos, x_neg, y, show=False):

        # first forward pass
        y_pos = self.forward(x_pos)
        # second forward pass
        y_neg = self.forward(x_neg)

        # if len(self.ep_losses) == 500:
        #     k = y_pos.clone()
        #     nun = len(k[0])
        #     H = int(np.sqrt(nun))
        #     print('nu_nn: {}, Unflatten size: {}'.format(nun, H))
        #     unflatten = nn.Unflatten(1, torch.Size([1, H, H]))
        #     sample = unflatten(k)
        #     self.vis_features(sample[0].detach().cpu(), H)

        g_pos, g_neg = self.goodness_factor(y_pos, y_neg)

        # loss = mean sigmoid(error) for both positive and negative data
        loss = self.loss(g_pos, g_neg, sigmoid=False)

        if show:
            print('g_pos: {}, g_neg:{}'.format(g_pos.mean(), g_neg.mean()))
            # print('Layer Loss: {}'.format(loss))


        return y_pos, y_neg, loss
    




class FC_LayerVFFAE_enc(nn.Linear):
    def __init__(self, in_features, out_features, layer_number = False, dropout = False, batchnorm = False,
                 act_fn =None, normalize = False, layernorm = False, bias=True, device=None, dtype=None, droprate=0):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        self.isdropout = dropout
        self.isnormalize = normalize
        self.islayernorm = layernorm

        self.activation = act_fn
        if self.activation is None:
            self.act_fn = None
        elif self.activation == 'relu':
            self.act_fn = nn.ReLU(inplace = True)
        elif self.activation == "leakyrelu":
            self.act_fn = nn.LeakyReLU(inplace = True)
        elif self.activation == "tanh":
            self.act_fn = nn.Tanh()
        elif self.activation == "sigmoid":
            self.act_fn = nn.Sigmoid() 

        
        if self.isdropout:
            self.dropout = torch.nn.Dropout(p=droprate)

        self.lr = 0.02  # 0,01
        self.opt = Adam(self.parameters(), lr=self.lr, betas=(0.95, 0.999))
        self.threshold = 2.0
        self.ep_losses = []
        self.layer_number = layer_number
        self.isbatchnorm = batchnorm

        if self.isbatchnorm:
            self.conv_bn = nn.BatchNorm1d(out_features)
        if self.islayernorm:
            self.layernorm = nn.LayerNorm(out_features)

    def forward(self, x):

        if len(x.shape) > 2:
            x = x.reshape(x.size(0), -1)

        if self.isdropout:
            x = self.dropout(x)
        # normalize magnitude activations of previous layer to forward
        # only orientation of activity vector norm 2 sum of squared activations
        if self.isnormalize:
            x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
            x = torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0)
        else:
            x = torch.mm(x, self.weight.T) + self.bias.unsqueeze(0)
        
        if self.isbatchnorm:
            x = self.conv_bn(x)

        if self.islayernorm:
            x = self.layernorm(x)
            
        # find activity vector y
        if self.act_fn is not None:
            x = self.act_fn(x)

        return x

    def setdropout(self, drop_rate):
        self.dropout = torch.nn.Dropout(p=drop_rate)

    def loss(self, y, projected_x, sigmoid=False):
            # Optionally apply sigmoid activation to the predictions
            if sigmoid:
                projected_x = torch.sigmoid(projected_x)

            # Compute the MSE loss
            loss = F.mse_loss(projected_x, y)

            return loss

    def forward_forward(self, x):
        seed = 42
        torch.manual_seed(seed)
        
        y = self.forward(x)
        n_components = y.shape[1] # number of most important components equal to the output size of the layer
        
        projected_data, _ , _ = self._pca(x, n_components)

        loss = self.loss(y, projected_data, sigmoid=False)


        return y, loss
    
    def _pca(self, data, n_components):
        # Step 1: Flatten data if necessary
        if len(data.shape) > 2:
            data = data.reshape(data.size(0), -1)
        
        # Step 2: Standardize the data (optional)
        epsilon = 1e-8
        mean = data.mean(dim=0, keepdim=True) # shape: 
        std = data.std(dim=0, unbiased=False, keepdim=True) + epsilon
        standardized_data = (data - mean) / std

        # Step 3: Apply PCA with low-rank approximation
        U, S, V = torch.pca_lowrank(standardized_data, q=n_components)

        # Step 4: Project the data onto the top principal components
        projected_data = torch.matmul(data, V[:, :n_components])

        return projected_data, V, S
    


class FC_LayerVFFAE_dec(nn.Linear):
    def __init__(self, in_features, out_features, layer_number = False, dropout = False, batchnorm = False,
                 act_fn =None, normalize = False, layernorm = False, bias=True, device=None, dtype=None, droprate=0):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        self.isdropout = dropout
        self.isnormalize = normalize
        self.islayernorm = layernorm
        self.isbatchnorm = batchnorm

        self.activation = act_fn
        if self.activation is None:
            self.act_fn = None
        elif self.activation == 'relu':
            self.act_fn = nn.ReLU(inplace = True)
        elif self.activation == "leakyrelu":
            self.act_fn = nn.LeakyReLU(inplace = True)
        elif self.activation == "tanh":
            self.act_fn = nn.Tanh()
        elif self.activation == "sigmoid":
            self.act_fn = nn.Sigmoid() 
        if self.isdropout:
            self.dropout = torch.nn.Dropout(p=droprate)

        self.lr = 0.02  # 0,01
        self.opt = Adam(self.parameters(), lr=self.lr, betas=(0.95, 0.999))
        self.threshold = 2.0
        self.ep_losses = []
        self.layer_number = layer_number
        
        if self.isbatchnorm:
            self.conv_bn = nn.BatchNorm1d(out_features)

        if self.islayernorm:
            self.layernorm = nn.LayerNorm(out_features)


    def forward(self, x):

        if len(x.shape) > 2:
            x = x.reshape(x.size(0), -1)

        if self.isdropout:
            x = self.dropout(x)
        # normalize magnitude activations of previous layer to forward
        # only orientation of activity vector norm 2 sum of squared activations
        if self.isnormalize:
            x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
            x = torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0)
        else:
            x = torch.mm(x, self.weight.T) + self.bias.unsqueeze(0)
        
        if self.isbatchnorm:
            x = self.conv_bn(x)

        if self.islayernorm:
            x = self.layernorm(x)

        # find activity vector y
        if self.act_fn is not None:
            x = self.act_fn(x)

        return x

    def setdropout(self, drop_rate):
        self.dropout = torch.nn.Dropout(p=drop_rate)

    def loss(self, y, x_enc, sigmoid=False):
            # Optionally apply sigmoid activation to the predictions
            if sigmoid:
                x_enc = torch.sigmoid(x_enc)

            # Compute the MSE loss
            loss = F.mse_loss(x_enc, y)

            return loss

    def forward_forward(self, x, x_enc):

        y = self.forward(x)

        if len(x_enc.shape) > 2:
            x_enc = x_enc.reshape(x.size(0), -1)
        
        loss = self.loss(y, x_enc, sigmoid=False)



        return y, loss
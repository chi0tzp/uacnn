import torch
import torch.nn as nn
from torch.nn import Sigmoid
from torch.nn import functional as F
import numpy as np
from math import pi


class UAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        super(UAConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation,
                              groups=self.groups,
                              bias=self.bias,
                              padding_mode=self.padding_mode)

    def forward(self, x):
        """Propagate random input (in terms of means and variances) through a 2D convolution layer.
        More specifically ...

        Args:
            x (tuple): x = (x_mean, x_var), where
                x_mean (torch.Tensor): input means matrix
                x_var (torch.Tensor): input variances matrix

        Returns:
            y (list): y = [y_mean, y_var], where
                y_mean (torch.Tensor): output means matrix
                y_var (torch.Tensor): output variances matrix

        """
        # print("### self.conv.weight      = {}".format(self.conv.weight.norm()))
        # print("### self.conv.weight ** 2 = {}".format((self.conv.weight ** 2).norm()))

        return [self.conv(x[0]), F.conv2d(input=x[1],
                                          weight=self.conv.weight ** 2,
                                          bias=None,
                                          stride=self.conv.stride,
                                          padding=self.conv.padding,
                                          dilation=self.conv.dilation,
                                          groups=self.conv.groups)]


class UAAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(UAAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        """Spatially down-sample input tensor x using average pooling. Along with the mean maps, compute the
        corresponding variance maps.
        Args:
            x (torch.Tensor): input tensor of size (bs, ch, d, d), where bs is the batch size, ch is the
                              number of input channels, and d input dimension.
        Returns:
            y_mean (torch.Tensor): mean map of size (bs, ch, d, d).
            y_var (torch.Tensor): variance map of size (bs, ch, d, d).
        """
        x_mean = x[0]
        x_var = x[1]
        y_mean = self.avg_pool(x_mean)
        y_var = (1 / self.kernel_size**2) * self.avg_pool(x_var)
        return [y_mean, y_var]


class UALinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(UALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features=self.in_features, out_features=self.out_features)

    def forward(self, x):
        """Propagate random input (in terms of means and variances) through a given Linear layer (given as the set of
        its parameters; i.e., its weight and its bias).

        Args:
            x (tuple): x = (x_mean, x_var), where
                x_mean (torch.Tensor): input means matrix
                x_var (torch.Tensor): input variances matrix

        Returns:
            y (list): y = [y_mean, y_var], where
                y_mean (torch.Tensor): output means matrix
                y_var (torch.Tensor): output variances matrix


        """
        x_mean = x[0]
        x_var = x[1]
        y_mean = self.linear(x_mean)
        y_var = torch.mm(self.linear.weight ** 2, x_var.t()).t()
        return [y_mean, y_var]


class UADropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(UADropout, self).__init__()
        self.p = p
        self.inplace = inplace
        self.dropout = nn.Dropout(p=self.p, inplace=self.inplace)

    def forward(self, x):
        """Propagate random input (in terms of means and variances) through a Dropout layer. More specifically

        Args:
            x (tuple): x = (x_mean, x_var), where
                x_mean (torch.Tensor): input means matrix
                x_var (torch.Tensor): input variances matrix

        Returns:
            y (list): y = [y_mean, y_var], where
                y_mean (torch.Tensor): output means matrix
                y_var (torch.Tensor): output variances matrix

        """
        x_mean = x[0]
        x_var = x[1]
        y_mean = self.dropout(x_mean)
        dropout_mask = torch.div(y_mean, x_mean + 1e-12)
        y_var = (dropout_mask ** 2) * x_var
        return [y_mean, y_var]


class UAReLU(nn.Module):
    def __init__(self, inplace=False):
        super(UAReLU, self).__init__()
        self.inplace = inplace
        self.relu = nn.ReLU()

    def forward(self, x):
        """Propagate random input (in terms of means and variances) through a Rectified Linear Unit (ReLU).

        Args:
            x (tuple): x = (x_mean, x_var), where
                x_mean (torch.Tensor): input means matrix
                x_var (torch.Tensor): input variances matrix

        Returns:
            y (list): y = [y_mean, y_var], where
                y_mean (torch.Tensor): output means matrix
                y_var (torch.Tensor): output variances matrix

        """
        x_mean = x[0]
        x_var = x[1]
        eps = 1e-12
        r = torch.div(x_mean, torch.sqrt(2 * x_var) + eps)
        exp = torch.exp(-r ** 2)
        erf = torch.erf(r)
        y_mean = torch.sqrt(torch.div(x_var + eps, 2 * pi)) * exp + 0.5 * x_mean * (1 + erf)
        y_var = 0.5 * (x_var + x_mean ** 2) * (1 + erf) - \
            (1.0 / np.sqrt(2 * pi)) * x_mean * torch.sqrt(x_var + eps) * exp - y_mean ** 2

        return [y_mean, y_var]


class UASigmoid(nn.Module):
    def __init__(self):
        super(UASigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.alpha = 0.368

    def forward(self, x):
        """Propagate random input (in terms of means and variances) through sigmoid.

        Args:
            x (tuple): x = (x_mean, x_var), where
                x_mean (torch.Tensor): input means matrix
                x_var (torch.Tensor): input variances matrix

        Returns:
            y (list): y = [y_mean, y_var], where
                y_mean (torch.Tensor): output means matrix
                y_var (torch.Tensor): output variances matrix

        """
        return [self.sigmoid(torch.div(x[0], torch.sqrt(1 + self.alpha * x[1]))),
                self.sigmoid(torch.sqrt(1 + 3 * torch.div(x[1], pi ** 2)))
                * (1.0 - self.sigmoid(torch.sqrt(1 + 3 * torch.div(x[1], pi ** 2))))
                * (1.0 - 1.0 / torch.sqrt(1.0 + 3.0 * torch.div(x[1], pi ** 2)))]


class UABCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(UABCELoss, self).__init__()
        if reduction not in ('mean', 'sum'):
            raise ValueError("UABCELoss: invalid reduction method: {}. Choose 'mean' or 'sum'.".format(reduction))
        else:
            self.reduction = reduction
        self.sigmoid = Sigmoid()

    def forward(self, score_mean, score_var, target):
        loss = (1 - target) * score_mean - torch.log(self.sigmoid(score_mean)) + \
               0.5 * self.sigmoid(score_mean) * (1 - self.sigmoid(score_mean)) * score_var
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

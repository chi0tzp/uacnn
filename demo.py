import torch
from ualib import *


def main():
    # Define UAConv2d layer
    uaconv2d = UAConv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    # Define UAReLU
    uarelu = UAReLU()

    # Define UAAvgPool2d
    uaavgpool2d = UAAvgPool2d(kernel_size=2, stride=2)

    # TODO: add comment
    x_mean = torch.randn(16, 3, 224, 224)
    x_var = torch.rand(16, 3, 224, 224)


if __name__ == '__main__':
    main()

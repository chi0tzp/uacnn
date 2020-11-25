# UA-CNN: Uncertainty-aware CNN

Propagation of Gaussian uncertainty through typical CNN building blocks.

![](inkscape/filtering.svg)



- **UAConv2d**: Uncertainty-aware 2D convolution

  ![](inkscape/uaconv2d.svg)

- **UAAvgPool2d**: Uncertainty-aware 2D pooling

  ![](inkscape/uaavgp.svg)

- **UALinear**: Uncertainty-aware linear (fully-connected) layer

  ![](inkscape/uafc.svg)

- **UAReLU**: Uncertainty-aware rectified linear unit for various amounts of input uncertainty

  ![](inkscape/uarelu_mean_var_plot.svg)

- Expected BCE loss (**UABCELoss**) for various amounts of input uncertainty (dashed red lines) compared to standard BCE loss

  ![](inkscape/uabce_mean.svg)

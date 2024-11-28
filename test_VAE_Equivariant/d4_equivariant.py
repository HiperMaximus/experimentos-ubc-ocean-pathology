import torch
import torch.nn as nn
import torch.nn.functional as F

class D4EquivariantConv(nn.Module):
    """
    A convolutional layer that achieves equivariance to the dihedral group D4 (rotations and reflections).
    
    This layer creates 8 transformations for each kernel: 4 rotations (0°, 90°, 180°, 270°) and 4 reflections 
    of each one of these rotations. The output for each transformed kernel is concatenated along the channel dimension, 
    allowing the model to learn features equivariant to these transformations.
    
    Attributes:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel (default is 3).
    padding : int
        Padding applied to the input (default is 1).
    stride : int
        Stride of the convolution (default is 1).
    bias : bool
        If True, adds a learnable bias to the output (default is True).
    num_transformations : int
        Number of transformations in the D4 group, set to 8.
    
    Methods:
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Applies the D4 transformations to the convolutional kernel and performs convolution on the input tensor.
    
    Transformation Details:
    -----------------------
    For each original kernel:
        1. Rotate it by 90° counterclockwise three times to create 4 rotations (0°, 90°, 180°, 270°).
        2. Reflect the original kernel horizontally and then apply the 4 rotations to the reflected kernel.
        
    This process results in 8 transformations per kernel. The transformed kernels are used for convolution,
    allowing the model to learn features that are robust to rotations and reflections.
    
    Examples:
    --------
    >>> layer = D4EquivariantConv(in_channels=1, out_channels=16)
    >>> input_tensor = torch.randn(1, 1, 32, 32)
    >>> output = layer(input_tensor)
    """
    
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, padding:int=1, stride:int=1, bias:bool=True):
        super(D4EquivariantConv, self).__init__()
        # Order of Dihedral group D4
        self.num_transformations=8
        
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride

        self.weight = nn.Parameter(torch.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)))
        self.bias_weight = nn.Parameter(torch.zeros(self.out_channels)) if bias else None


        # Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x:torch.Tensor):
        # Create tensor to populate with all transformed kernels
        transformed_filters = torch.empty((self.num_transformations*self.out_channels,
                                           self.in_channels,
                                           self.kernel_size,
                                           self.kernel_size), device=x.device)
        # Generate all 8 transformations
        for i in range(4):
            # Pure rotation for each of the 4 rotations
            rotated_weight = torch.rot90(self.weight, i, [2, 3])
            transformed_filters[i * self.out_channels:(i + 1) * self.out_channels] = rotated_weight
            
            # Reflected versions of each rotation
            reflected_weight = torch.flip(rotated_weight, dims=[3])
            transformed_filters[(i + 4) * self.out_channels:(i + 5) * self.out_channels] = reflected_weight


        # Repeat bias for each transformed kernel if it exists
        expanded_bias = self.bias_weight.repeat(self.num_transformations) if self.bias_weight is not None else None

        # Perform convolution using the combined filters
        out = F.conv2d(x, transformed_filters, padding=self.padding, bias=expanded_bias, stride=self.stride)

        return out 


class D4EquivariantConvTranspose(nn.Module):
    """
    A layer that performs an equivariant "deconvolution" to the dihedral group D4 by:
    1. Upsampling the input by inserting zeros between rows and columns.
    2. Applying the D4-equivariant convolution layer to the upsampled input.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2, bias: bool = True):
        super(D4EquivariantConvTranspose, self).__init__()
                
        # Using an equivariant convolution layer
        self.equiv_conv = D4EquivariantConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same',
            stride=1,         # Stride of 1 since upsampling is done manually
            bias=bias
        )
        self.stride = stride

    def forward(self, x: torch.Tensor):
        # Step 1: Insert zeros between rows and columns to simulate ConvTranspose2d upsampling
        n, c, h, w = x.size()
        upsampled = torch.zeros(n, c, h * self.stride, w * self.stride, device=x.device)
        upsampled[:, :, ::self.stride, ::self.stride] = x
        
        # Step 2: Apply the D4 equivariant convolution to the upsampled input
        out = self.equiv_conv(upsampled)
        return out

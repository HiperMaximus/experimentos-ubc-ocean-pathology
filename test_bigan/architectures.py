"""Utility nn.layers, nn.models and architectures for BiGAN, taking into modern practices.

It takes into account Fixup initialization (also for activations), ConvNext type layers and
R1 and R2 gradient penalization for the Discriminator
This module provides:
- Conv2dZeroBias: Fixup aware conv layer;
- PReLUWithBias: Fixup modified PReLU;
- CustomConvNeXtBlock: Generic resnet like block;
- Generator: Generator for BiGAN;
- Encoder: Encoder for BiGAN;
- Discriminator: Discriminator for BiGAN with its loss
"""

from typing import Literal, override

import torch
from torch import nn


# Fixup layers
class Conv2dZeroBias(nn.Conv2d):
    """A conv2d layer that follows Fixup initialization guidelines."""

    @override
    def reset_parameters(self) -> None:
        super().reset_parameters()
        # Override the bias initialization.
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class PReLUWithBias(nn.Module):
    """Applies a PReLU activation function followed by an additive bias per channel following Fixup initialization.

    This module is designed for use after nn.Conv2d layers, where each output channel
    has its own learnable bias parameter. The PReLU activation has a single learnable
    parameter shared across all channels.

    Args:
        num_channels (int): Number of output channels from the preceding Conv2d layer.
        prelu_init (float, optional): Initial value for the PReLU parameter. Default is 0.25.

    """

    def __init__(self, num_channels: int, prelu_init: float = 0.25, prelu_num_parameters: int = 1) -> None:
        """Initialize a PReLU activation with an additional bias.

        Args:
            num_channels (int): Number of channels to set the size of the bias.
            prelu_init (float): Init value of PReLU.
            prelu_num_parameters (int): number of learnable parameters of PReLU, must be 1 or
            the number of channels of the input.

        """
        if prelu_num_parameters not in (1, prelu_num_parameters):
            msg = "'prelu_num_parameters' must be 1 or the number of channels of the input."
            raise ValueError(msg)
        super().__init__()
        # Create a PReLU activation module with the desired number of parameters.
        self.prelu = nn.PReLU(num_parameters=prelu_num_parameters, init=prelu_init)
        # Create a bias parameter. Its shape is (num_parameters,) so it can be broadcasted across spatial dimensions.
        self.bias = nn.Parameter(torch.full((1, num_channels, 1, 1), 0.0))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying PReLU activation followed by the addition of the bias.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, H, W).

        Returns:
            torch.Tensor: Activated tensor with bias added.

        """
        # Apply the PReLU activation.
        x = self.prelu(x)
        # Add the bias.
        return x + self.bias


class CustomConvNeXtBlock(nn.Module):
    """Residual conv2d block that follows Fixup initialization guidelines."""

    def __init__(self, num_channels: int, num_res_blocks: int, stride: int = 1) -> None:
        """Instantiate a ConvNeXt-style residual block incorporating Fixup initialization.

        The block consists of:
          1. Adding a bias initialized at 0.
          2. A pointwise convolution (expansion) with bias.
          3. An activation (PReLU) with an extra bias.
          4. A depthwise convolution with bias.
          5. A second activation (PReLU) with an extra bias.
          6. A learnable scaling parameter applied to the residual branch output.
          7. A residual connection adding the input to the processed output.

        Fixup initialization details:
          - Weight layers (except the last one) are scaled by L^(-1/(2*m-2)), where m is
          the number of linear-like layers (here m=3).
          - The last layer of the residual branch is initialized to zero.
          - Extra biases are added before each convolution and activation.

        Args:
            num_channels (int): Number of channels in the input feature map.
            num_res_blocks (int): Total number of residual blocks in the network (L).
            stride (int, optional): Set stride of depthwise layer, use stride = 2 for the usual downsample.

        """
        super().__init__()
        m = 3  # number of linear-like layers in the residual branch
        expanded_dim = 4 * num_channels  # expansion factor used in ConvNeXt

        # Compute the scaling factor for weight layers in the residual branch.
        # For m=3, the scale becomes L^(-1/(2*3-2)) = L^(-1/4)
        scale = num_res_blocks ** (-1 / (2 * m - 2))

        # Define the layers with explicit bias addition
        self.initial_bias = nn.Parameter(torch.full((1, num_channels, 1, 1), 0.0))
        # Fixup initialization:
        # Scale the weights of the first two layers in the residual branch.
        self.pwconv1 = Conv2dZeroBias(
            num_channels,
            expanded_dim,
            kernel_size=1,
            bias=True,
        )
        self.pwconv1.weight.data.mul_(scale)  # Scale weights

        self.dwconv = Conv2dZeroBias(
            expanded_dim,
            expanded_dim,
            kernel_size=7,
            padding="same",
            groups=expanded_dim,  # depthwise convolution: one group per channel
            stride=stride,
            bias=True,
        )
        self.dwconv.weight.data.mul_(scale)  # Scale weights

        self.pwconv2 = Conv2dZeroBias(
            expanded_dim,
            num_channels,
            kernel_size=1,
            bias=True,
        )
        # Initialize the weights of the last layer (pointwise projection) to zero,
        # ensuring that the residual branch initially outputs zero.
        nn.init.constant_(self.pwconv2.weight, 0)

        # Define the activation functions with explicit bias addition.
        self.prelu1 = PReLUWithBias(expanded_dim)
        self.prelu2 = PReLUWithBias(expanded_dim)

        # Learnable scaling parameter for the output of the residual branch
        self.layer_scale = nn.Parameter(torch.ones(1))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the CustomConvNeXtBlock.

        The input passes through the residual branch and is then added to the original
        input via a residual connection. This allows the block to learn perturbations
        to the identity mapping.

        Args:
            x (torch.Tensor): Input tensor of shape (N, dim, H, W).

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.

        """
        residual = x  # Store input for the residual connection

        # Pass input through the first pointwise convolution (with bias addition)
        x = self.pwconv1(x + self.initial_bias)
        # Apply the first activation (with bias addition)
        x = self.prelu1(x)
        # Pass through the depthwise convolution (with bias addition)
        x = self.dwconv(x)
        # Apply the second activation (with bias addition)
        x = self.prelu2(x)
        # Apply the final pointwise convolution (with bias addition)
        x = self.pwconv2(x)
        # Scale the output of the residual branch
        x = self.layer_scale * x
        # Combine with the input via a residual connection
        return x + residual


###############################################################################
# Generator
###############################################################################
class Generator(nn.Module):
    """Generator for GAN using the custom residual conv2d blocks that follows Fixup initialization guidelines."""

    def __init__(
        self,
        base_channels: int = 64,
        latent_dim: int = 128,
        num_upsamples: int = 4,
        num_blocks_per_scale: int = 2,
        img_channels: int = 3,
    ) -> None:
        """Initialize Generator network for BiGAN.

        It uses a normal sampled tensor of shape (latent_dim) and progressively upsamples to produce an output image.

        Args:
            base_channels (int): Number of channels at the starting 4x4 resolution.
            latent_dim (int): Size of the latent dimension of the Encoder output and Generator input.
            num_upsamples (int): How many times to upsample (each upsampling doubles spatial dims).
            num_blocks_per_scale (int): Number of ConvNeXt blocks per resolution scale.
            img_channels (int): Number of channels in the output image (e.g., 3 for RGB).

        """
        super().__init__()
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        num_res_blocks = num_upsamples * num_blocks_per_scale
        # Regularize initial training
        self.constant_bias = nn.Parameter(torch.zeros(self.latent_dim * 4 * 4))
        self.noise_scale = nn.Parameter(torch.ones(1) * 0.1)  # Small initial scale
        self.input_linear = nn.Linear(
            self.latent_dim,
            self.base_channels * 4 * 4,
            bias=True,
        )

        # Create a list of modules for each resolution scale.
        self.upsample_stages = nn.ModuleList()

        for _ in range(num_upsamples):
            # For each scale, stack a few ConvNeXt blocks.
            blocks = [
                CustomConvNeXtBlock(num_channels=base_channels, num_res_blocks=num_res_blocks)
                for _ in range(num_blocks_per_scale)
            ]
            self.upsample_stages.append(nn.Sequential(*blocks))
            # Optionally, one could also vary the number of channels here (e.g., doubling channels
            # at lower resolutions and halving them later). Here we keep it constant for simplicity.

        # Final RGB conversion convolution.
        self.to_rgb = nn.Conv2d(base_channels, img_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.to_rgb.weight)
        if self.to_rgb.bias is not None:
            nn.init.zeros_(self.to_rgb.bias)  # Initialize bias to 0

        # A final activation.
        self.final_activation = PReLUWithBias(num_channels=3, prelu_init=0.9, prelu_num_parameters=3)

    def forward(self, input_noise: torch.Tensor) -> torch.Tensor:
        """Forward pass for the generator.

        Args:
            input_noise (torch.Tensor): Noise tensor from where the images are generated shape (batch_size, latent_dim)

        Returns:
            torch.Tensor: Generated image tensor of shape (batch_size, img_channels, H, W).

        """
        # Start with a random vector for the seed generation.
        x: torch.Tensor = (input_noise + self.constant_bias) * self.noise_scale  # Add bias and scale initial noise
        x = self.input_linear(x)
        x = x.reshape((-1, self.base_channels, 4, 4))

        # Progressive upsampling: each stage applies ConvNeXt blocks then upsamples.
        for stage in self.upsample_stages:
            # Apply the sequence of ConvNeXt blocks.
            x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = stage(x)

        # Apply the final layer
        x = self.to_rgb(x)
        # The output does not use tanh; this allows the network to learn proper pixel intensities.
        return self.final_activation(x)


###############################################################################
# Encoder
###############################################################################
class Encoder(nn.Module):
    """Encoder for BiGAN that inverts the Generator.

    This architecture mirrors the Generator: it maps an image down to a latent vector
    of dimension `latent_dim` (e.g. 128), using a progressive downsampling strategy.
    """

    def __init__(
        self,
        base_channels: int = 64,
        latent_dim: int = 128,
        num_downsamples: int = 4,
        num_blocks_per_scale: int = 2,
        img_channels: int = 3,
    ) -> None:
        """Initialize Encoder network for BiGAN.

        It uses a normal sampled tensor of shape (latent_dim) and progressively upsamples to produce an output image.

        Args:
            base_channels (int): Number of channels after the initial mapping.
            latent_dim (int): Size of the latent vector (must match Generator).
            num_downsamples (int): Number of times to downsample the input image.
            num_blocks_per_scale (int): Number of ConvNeXt blocks per resolution scale.
            img_channels (int): Number of channels in the input image (e.g., 3 for RGB).

        """
        super().__init__()
        num_res_blocks = num_downsamples * num_blocks_per_scale

        # Map input image to the base feature space with a 1x1 convolution.
        self.from_rgb = nn.Conv2d(
            img_channels,
            base_channels,
            kernel_size=1,
            bias=True,
        )

        # Create downsampling stages.
        self.downsample_stages = nn.ModuleList()
        for _ in range(num_downsamples):
            blocks = []
            for i in range(num_blocks_per_scale):
                if i + 1 == num_blocks_per_scale:
                    # Last block in each stage: downsample using stride=2.
                    blocks.append(
                        CustomConvNeXtBlock(
                            num_channels=base_channels,
                            num_res_blocks=num_res_blocks,
                            stride=2,
                        ),
                    )
                else:
                    blocks.append(
                        CustomConvNeXtBlock(
                            num_channels=base_channels,
                            num_res_blocks=num_res_blocks,
                        ),
                    )
            self.downsample_stages.append(nn.Sequential(*blocks))
            # For simplicity, we keep the channel count constant.

        # After downsampling, the spatial dimensions should be 4x4.
        # Flatten feature map: dimension = base_channels * 4 * 4.
        self.linear_1 = nn.Linear(base_channels * 4 * 4, base_channels * 4 * 4)
        self.linear_activation = PReLUWithBias(num_channels=base_channels * 4 * 4)
        # Project down to the desired latent dimension.
        self.linear_2 = nn.Linear(base_channels * 4 * 4, latent_dim)
        self.final_activation = PReLUWithBias(num_channels=latent_dim, prelu_init=0.9, prelu_num_parameters=latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Encoder.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, img_channels, H, W).
                For best symmetry, H and W should match the Generator's output resolution (e.g., 64).

        Returns:
            torch.Tensor: Encoded latent vector of shape (batch_size, latent_dim).

        """
        # Map image to feature space.
        x = self.from_rgb(x)
        # Apply downsampling stages.
        for stage in self.downsample_stages:
            x = stage(x)
        # Flatten the feature map (expected shape: (batch, base_channels, 4, 4)).
        x = x.view(x.size(0), -1)
        x = self.linear_1(x)
        x = self.linear_activation(x)
        x = self.linear_2(x)
        return self.final_activation(x)


###############################################################################
# Discriminator
###############################################################################
class Discriminator(nn.Module):
    """BiGAN Discriminator with relativistic LS loss and R1/R2 gradient penalties.

    This network takes a joint pair (x, z) as input. The image x is processed via
    a ResNet-like architecture and the latent vector z is projected to the same dimension;
    the two are fused (via concatenation) to produce joint features.
    """

    def __init__(
        self,
        base_channels: int = 64,
        num_downsamples: int = 4,
        num_blocks_per_scale: int = 2,
        img_channels: int = 3,
        latent_dim: int = 128,
    ) -> None:
        """Initialize Discriminator network that progressively downsamples the image using strided convolution.

        It uses a ResNet-like design, where residual blocks (our ConvNeXt blocks) are used, and
        skip connections are naturally integrated via the block design.

        Args:
            base_channels (int): Number of channels after the initial mapping.
            num_downsamples (int): Number of times to downsample the input image.
            num_blocks_per_scale (int): Number of ConvNeXt blocks per resolution scale.
            img_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            latent_dim (int): Size of the latent dimension of the Encoder output and Generator input

        """
        super().__init__()
        num_res_blocks = num_downsamples * num_blocks_per_scale

        # Map the input image to the base feature space with a 1x1 convolution.
        self.from_rgb = nn.Conv2d(
            img_channels,
            base_channels,
            kernel_size=1,
            bias=True,
        )

        # Create downsampling stages.
        self.downsample_stages = nn.ModuleList()
        blocks: list[CustomConvNeXtBlock]
        for _ in range(num_downsamples):
            blocks = []
            for i in range(num_blocks_per_scale):
                if i + 1 == num_blocks_per_scale:
                    blocks.append(
                        CustomConvNeXtBlock(
                            num_channels=base_channels,
                            num_res_blocks=num_res_blocks,
                            stride=2,
                        ),
                    )
                else:
                    blocks.append(
                        CustomConvNeXtBlock(
                            num_channels=base_channels,
                            num_res_blocks=num_res_blocks,
                        ),
                    )
            self.downsample_stages.append(nn.Sequential(*blocks))
            # For simplicity, we keep channel count constant.

        # A final classifier head.
        # After downsampling, we expect the spatial dimensions to be 4x4 (if starting, e.g., from 64x64).
        # Flatten the feature map and apply a linear layer to produce a scalar output.
        feature_dim = base_channels * 4 * 4
        # Allow concatenating the projected latent vector
        feature_dim *= 2

        # Linear layers for final classification on the joint feature.
        self.linear_1 = nn.Linear(feature_dim, feature_dim)
        self.linear_activation = PReLUWithBias(num_channels=feature_dim, prelu_num_parameters=feature_dim)
        self.linear_2 = nn.Linear(feature_dim, 1)
        self.final_activation = PReLUWithBias(num_channels=1, prelu_init=0.9)

        # Latent projection: project z to the same dimension as the flattened image feature.
        self.latent_projection = nn.Linear(latent_dim, feature_dim)
        # Latent activation to apply after the projection
        self.latent_activation = PReLUWithBias(num_channels=1, prelu_init=0.9)

    @override
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Forward pass for the BiGAN discriminator.

        If a latent vector z is provided, it is projected and added to the flattened image features.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch, img_channels, H, W).
            z (torch.Tensor): Latent vector corresponding to x via the Encoder
            (or seed z of Generator), of shape (batch,latent_dim).

        Returns:
            torch.Tensor: A scalar output for each image indicating realness.

        """
        # Process image x.
        x = self.from_rgb(x)
        for stage in self.downsample_stages:
            x = stage(x)
        # Flatten image features.
        f_img = x.view(x.size(0), -1)

        # Project latent z.
        f_z = self.latent_projection(z)
        f_z = self.latent_activation(f_z)

        # Concatenate along the feature dimension.
        # This yields a feature vector of dimension: feature_dim*2.
        f_joint = torch.cat((f_img, f_z), dim=1)

        # Process the joint feature through the linear layers.
        out = self.linear_1(f_joint)
        out = self.linear_activation(out)
        out = self.linear_2(out)
        return self.final_activation(out)

    def f_least_squares(self, v: torch.Tensor) -> torch.Tensor:
        """Least squares classification loss: (1 - v)**2."""
        return torch.square(1 - v).mean()

    def calculate_relativistic_loss(
        self,
        real_images: torch.Tensor,
        real_latents: torch.Tensor,
        fake_images: torch.Tensor,
        fake_latents: torch.Tensor,
        gamma: float,
        gradient_penalty: Literal["R1", "R2"],
    ) -> torch.Tensor:
        """Compute the relativistic LS loss with an alternating gradient penalty R1 and R2.

        Args:
            real_images (torch.Tensor): Real images.
            real_latents (torch.Tensor): Latent codes from the encoder E(x).
            fake_images (torch.Tensor): Generated images G(z).
            fake_latents (torch.Tensor): Original latent codes z.
            gamma (float): Regularization weight.
            gradient_penalty (Literal["R1","R2"]): If "R1", apply R1 penalty (real data); if "R2", apply R2 penalty (fake data).

        Returns:
            torch.Tensor: Total loss (relativistic adversarial loss + chosen gradient penalty).

        """
        # Compute discriminator outputs for real and fake pairs.
        pred_real: torch.Tensor = self(real_images, real_latents)  # D(x, E(x))
        pred_fake: torch.Tensor = self(fake_images, fake_latents)  # D(G(z), z)

        # Compute relativistic difference.
        v = pred_real - pred_fake
        adv_loss = self.f_least_squares(v)

        # ----- Gradient Penalties -----
        # Compute gradient penalty based on flag.
        if gradient_penalty == "R1":
            # R1: Zero-centered gradient penalty on real data

            # Create a copy of real_images and real_latents with gradient tracking enabled.
            real_images_gp = real_images.clone().detach().requires_grad_(mode=True)
            # Compute discriminator outputs for these real inputs.
            pred_real_gp = self(real_images_gp, real_latents)

            # Prepare gradient outputs (ones tensor of the same shape as pred_real_gp).
            grad_outputs_real = torch.ones_like(pred_real_gp)
            # Compute gradients of pred_real_gp with respect to real_images_gp.
            gradients_real = torch.autograd.grad(
                outputs=pred_real_gp,
                inputs=real_images_gp,
                grad_outputs=grad_outputs_real,
                create_graph=True,
                retain_graph=False,
                only_inputs=True,
            )[0]

            # Reshape gradients per sample.
            gradients_real = gradients_real.view(real_images.size(0), -1)

            # Compute the R1 penalty (gamma * mean squared norm of gradients)
            grad_penalty = gamma * torch.mean(torch.sum(gradients_real**2, dim=1))
        else:
            # R2: Zero-centered gradient penalty on fake data
            # Create a copy of fake_images with gradient tracking enabled.
            fake_images_gp = fake_images.clone().detach().requires_grad_(mode=True)

            # Compute discriminator outputs for fake images.
            pred_fake_gp = self(fake_images_gp, fake_latents)

            # Prepare gradient outputs (ones tensor of the same shape as pred_fake_gp).
            grad_outputs_fake = torch.ones_like(pred_fake_gp)

            # Compute gradients of D_fake_gp with respect to fake_images_gp.
            gradients_fake = torch.autograd.grad(
                outputs=pred_fake_gp,
                inputs=fake_images_gp,
                grad_outputs=grad_outputs_fake,
                create_graph=True,
                retain_graph=False,
                only_inputs=True,
            )[0]

            # Reshape gradients per sample.
            gradients_fake = gradients_fake.view(fake_images.size(0), -1)

            # Compute the R2 penalty (gamma * mean squared norm of gradients)
            grad_penalty = gamma * torch.mean(torch.sum(gradients_fake**2, dim=1))

        # Total Discriminator Loss includes gradient penalties
        return adv_loss + grad_penalty

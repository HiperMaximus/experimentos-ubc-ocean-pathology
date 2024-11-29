import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from d4_equivariant import D4EquivariantConv, D4EquivariantConvTranspose

class D4_Equivariant_VAE(nn.Module):
    def __init__(self, latent_dim:int):
        super(D4_Equivariant_VAE, self).__init__()
        self.latent_dim=latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            D4EquivariantConv(3, 32//8, 3, stride=1, padding=1),  # Output: (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            D4EquivariantConv(32, 32//8, 3, stride=1, padding=1),  # Output: (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            D4EquivariantConv(32, 32//8, 4, stride=2, padding=1),  # Output: (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            D4EquivariantConv(32, 64//8, 4, stride=2, padding=1), # Output: (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            D4EquivariantConv(64, 128//8, 4, stride=2, padding=1), # Output: (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            D4EquivariantConv(128, 256//8, 4, stride=2, padding=1), # Output: (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            D4EquivariantConv(256, 512//8, 4, stride=2, padding=1), # Output: (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.middle_linear = nn.Linear(4 * 4 * 512, 4 * 4 * 512)

        # Embedding mean and variance
        self.fc_mu = nn.Linear(4 * 4 * 512, self.latent_dim)
        self.fc_logvar = nn.Linear(4 * 4 * 512, self.latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(self.latent_dim, 4 * 4 * 512)

        # Decoder using ConvTranspose2d for upsampling
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            
            # Reshape from latent vector to feature map
            D4EquivariantConvTranspose(512, 256//8, kernel_size=4, stride=2),  # Output: (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            D4EquivariantConvTranspose(256, 128//8, kernel_size=4, stride=2),  # Output: (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            D4EquivariantConvTranspose(128, 64//8, kernel_size=4, stride=2),   # Output: (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            D4EquivariantConvTranspose(64, 32//8, kernel_size=4, stride=2),    # Output: (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            D4EquivariantConvTranspose(32, 32//8, kernel_size=4, stride=2),    # Output: (32, 128, 128)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            D4EquivariantConv(32, 32//8, 3, stride=1, padding=1),  # Output: (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            D4EquivariantConv(32, 16//8, 3, stride=1, padding=1),  # Output: (16, 64, 64)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            
            # Final layer: project to 3 channels (RGB) with 3x3 convolution
            D4EquivariantConv(16, 3, kernel_size=3),  # Output: (3*8, 128, 128)

        )
        self.tanh=nn.Tanh()  # Or Sigmoid depending on normalization
        # Final layer outputs RGB values between -1 and 1 so Tanh
    
    
    def encode(self, x):
        # Apply the encoder
        x = self.encoder(x)
        #x = x.view(x.size(0), 512, 8, 4, 4)  # Reshaping to separate group dimension
        
        # Perform max pooling over the group dimension (dim=2)
        #x = torch.amax(x, dim=2)
        
        
        # Flatten the tensor to feed it into fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.middle_linear(x)
        # Apply the fully connected layers to obtain mu (mean) and logvar (log of variance)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

    
    def decode(self,z):
        # Decode
        x = self.decoder_input(z)
        x = x.view(-1, 512, 4, 4)
        x = self.decoder(x)
        x=x.view(x.size(0), 3, 8, 128, 128)  # Reshaping to separate group dimension
        # Perform max pooling over the group dimension (dim=2)
        x = torch.amax(x, dim=2)
        x=self.tanh(x)
        return x
    
    def vae_loss(self, recon_x, x, mu, logvar, beta=4):
        # Reconstruction loss MSE
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        # Calculate M and N
        M = self.latent_dim  # latent_dim
        N = x.size(1) * x.size(2) * x.size(3)  # C * H * W
        
        # Normalize the beta value
        beta_norm = beta * (M/N)  # This ensures that the KL term remains in a balanced scale

        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)


        # KL Divergence loss
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # SSIM loss
        ssim = StructuralSimilarityIndexMeasure(data_range=2).to(x.device)
        ssim_loss = 1 - ssim(recon_x, x)  # 1 - SSIM because higher SSIM means better quality

        
        return recon_loss + (beta_norm*kld_loss) + ssim_loss
    
    def forward(self, x):
        # Encode
        mu, logvar =self.encode(x)

        # Reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Decode
        x=self.decode(z)
        return x, mu, logvar

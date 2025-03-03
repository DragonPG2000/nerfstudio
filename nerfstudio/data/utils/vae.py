import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======= 1D Squeeze-Excitation (SE) Block =======
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock1D, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Pool across sequence length
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, l = x.shape
        y = self.global_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1)
        return x * y

class Encoder1D(nn.Module):
    def __init__(self, in_channels=141, latent_dim=64):
        super(Encoder1D, self).__init__()
        # Gradually reduce channels from 141 -> 128 -> 64
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=3, padding=1)
        self.se1 = SEBlock1D(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.se2 = SEBlock1D(64)
        self.conv3 = nn.Conv1d(64, latent_dim, kernel_size=3, padding=1)  # Final compression
        self.latent_dim = latent_dim

    def forward(self, x):
        # Input shape: [B, C, H, W] -> Reshape to [B, C, H*W]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        
        x = F.relu(self.conv1(x))
        x = self.se1(x)
        x = F.relu(self.conv2(x))
        x = self.se2(x)
        x = self.conv3(x)  # No activation for latent space
        x = x.view(B,self.latent_dim,H,W)
        return x

# ======= 1D Decoder =======
class Decoder1D(nn.Module):
    def __init__(self, latent_dim=64, out_channels=141):
        super(Decoder1D, self).__init__()
        # Gradually expand channels from 64 -> 128 -> 141
        self.deconv1 = nn.ConvTranspose1d(latent_dim, 64, kernel_size=3, padding=1)
        self.se1 = SEBlock1D(64)
        self.deconv2 = nn.ConvTranspose1d(64, 128, kernel_size=3, padding=1)
        self.se2 = SEBlock1D(128)
        self.deconv3 = nn.ConvTranspose1d(128, out_channels, kernel_size=3, padding=1)
        self.out_channels = out_channels

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        x = F.relu(self.deconv1(x))
        x = self.se1(x)
        x = F.relu(self.deconv2(x))
        x = self.se2(x)
        x = self.deconv3(x)  # Output channels match input
        x = x.view(B, self.out_channels, H, W) # Reshape to original shape
        return x

# ======= Complete 1D Autoencoder =======
class SpectralAutoencoder1D(nn.Module):
    def __init__(self, in_channels=141, latent_dim=64):
        super(SpectralAutoencoder1D, self).__init__()
        self.encoder = Encoder1D(in_channels, latent_dim)
        self.decoder = Decoder1D(latent_dim, in_channels)

    def forward(self, x):
        # Remember original spatial dimensions
        B, C, H, W = x.shape
        
        # Encode to latent space
        latent = self.encoder(x)
        
        # Decode and reshape back
        reconstructed = self.decoder(latent)
        reconstructed = reconstructed.view(B, C, H, W)  # Restore spatial dims
        
        return reconstructed
    

# ======= Inference Function =======
def load_model(checkpoint_path, device):
    model = SpectralAutoencoder1D(in_channels=141, latent_dim=64).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def infer(model, input_tensor, device):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        latent = model.encoder(input_tensor)
    return latent.cpu()
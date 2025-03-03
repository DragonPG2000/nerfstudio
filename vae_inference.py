import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm

# ======= Define the SEBlock2D Class =======
class SEBlock2D(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock2D, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.global_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

# ======= Define the Autoencoder Model =======
class Encoder2D(nn.Module):
    def __init__(self, in_channels=141, latent_dim=64):
        super(Encoder2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.se1 = SEBlock2D(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.se2 = SEBlock2D(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.se3 = SEBlock2D(128)
        self.conv4 = nn.Conv2d(128, latent_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.se1(F.relu(self.conv1(x)))
        x = self.se2(F.relu(self.conv2(x)))
        x = self.se3(F.relu(self.conv3(x)))
        x = self.conv4(x)
        return x

class Decoder2D(nn.Module):
    def __init__(self, latent_dim=64, out_channels=141):
        super(Decoder2D, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x

class SpectralAutoencoder2D(nn.Module):
    def __init__(self, in_channels=141, latent_dim=64):
        super(SpectralAutoencoder2D, self).__init__()
        self.encoder = Encoder2D(in_channels, latent_dim)
        self.decoder = Decoder2D(latent_dim, in_channels)

    def forward(self, x):
        latent = self.encoder(x)
        return latent

# ======= Inference Function =======
def load_model(checkpoint_path, device):
    model = SpectralAutoencoder2D(in_channels=141, latent_dim=64).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def infer(model, input_tensor, device):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        latent = model.encoder(input_tensor)
    return latent.cpu()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperspectral Encoder Inference")
    parser.add_argument("--input_dir", type=str, required=False, help="Path to input folder containing .pt tensors", default="/nethome/skumar704/flash/FAKEPLANT1_2/torch_masked/")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to model checkpoint", default="/nethome/skumar704/flash/hyperspectral_3d/PyTorch-VAE/vae_checkpoints/model_checkpoint_FAKEPLANT_12_epoch_100.pt")
    parser.add_argument("--latent_dir", type=str, required=False, help="Path to save latent codes", default="/nethome/skumar704/flash/FAKEPLANT1_2/latent/")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    
    os.makedirs(args.latent_dir, exist_ok=True)
    
    for file_name in tqdm(os.listdir(args.input_dir), desc="Processing files"):
        if file_name.endswith(".pt"):
            input_path = os.path.join(args.input_dir, file_name)
            latent_path = os.path.join(args.latent_dir, file_name)
            
            input_tensor = torch.load(input_path).permute(2, 0, 1).unsqueeze(0).float()
            latent_tensor = infer(model, input_tensor, device)
            
            torch.save(latent_tensor, latent_path)
            print(f"Processed {file_name}: Saved latent code to {latent_path}")

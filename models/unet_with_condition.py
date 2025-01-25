import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from models.cvae_v0 import *
from models.BioAligner import *
from data.Diffusion_dataset import *
import wandb  # Import WandB
import torch
import torch.nn as nn



class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width)
        value = self.value_conv(x).view(batch_size, -1, width)

        attention = torch.bmm(query, key)
        attention = self.softmax(attention)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width)

        out = self.gamma * out + x
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width)
        value = self.value_conv(x).view(batch_size, -1, width)

        attention = torch.bmm(query, key)
        attention = self.softmax(attention)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width)

        out = self.gamma * out + x
        return out


class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim,latent_dim):
        super(UNet1D, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.encoder5 = self.conv_block(512, 1024)

        self.attn1 = SelfAttention(64)
        self.attn2 = SelfAttention(128)
        self.attn3 = SelfAttention(256)
        self.attn4 = SelfAttention(512)
        self.attn5 = SelfAttention(1024)

        self.pool = nn.MaxPool1d(2)
        self.upconv4 = self.upconv_block(1024, 512)
        self.upconv3 = self.upconv_block(1024, 256)
        self.upconv2 = self.upconv_block(512, 128)
        self.upconv1 = self.upconv_block(256, 64)

        self.out_conv = nn.Conv1d(128, out_channels, kernel_size=1)

        # Conditional processing to handle `t` (batch_size, 1) and `c` (batch_size, embedding_dim)
        self.cond_processing = nn.Sequential(
            nn.Linear(1 + embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, in_channels),
            nn.ReLU()
        )

        self.condition_attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4)
        self.dropout = nn.Dropout(p=0.5)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t, c):
        # Concatenate `t` (batch_size, 1) and `c` (batch_size, embedding_dim)
        t_c_combined = torch.cat([t.view(-1, 1), c], dim=1).float()
        t_processed = self.cond_processing(t_c_combined)

        # Expand `t_processed` to match spatial dimensions of `x`
        t_processed = t_processed.view(-1, x.size(1), 1).repeat(1, 1, x.size(-1))

        # Integrate condition with attention
        x_attn, _ = self.condition_attention(t_processed, x, x)
        x = x + x_attn

        enc1 = self.encoder1(x)
        enc1 = self.attn1(enc1)
        enc2 = self.encoder2(self.pool(enc1))
        enc2 = self.attn2(enc2)
        enc3 = self.encoder3(self.pool(enc2))
        enc3 = self.attn3(enc3)
        enc4 = self.encoder4(self.pool(enc3))
        enc4 = self.attn4(enc4)
        enc5 = self.encoder5(self.pool(enc4))
        enc5 = self.attn5(enc5)

        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dropout(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dropout(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dropout(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dropout(dec1)
        out = self.out_conv(dec1)

        return out





class DDPMModel(pl.LightningModule):
    def __init__(self, unet_model, noise_scheduler, cvae_model,clip_model, learning_rate=1e-5):
        super(DDPMModel, self).__init__()
        self.unet_model = unet_model
        self.noise_scheduler = noise_scheduler
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.cvae_model = cvae_model
        self.clip_model = clip_model
        self.best_loss = float('inf')

    def forward(self, x, t, c):
        return self.unet_model(x, t, c)

    def training_step(self, batch, batch_idx):
        data, features, mask,text = batch
        data = data.float().to(device)
        features = features.to(device)
        mask = mask.to(device)
        mu, log_var = self.cvae_model.encode(data, features, mask)
        encoded_data = self.cvae_model.reparameterize(mu, log_var)
        encoded_data = encoded_data.unsqueeze(1)

        noise = torch.randn_like(encoded_data).to(device)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (encoded_data.size(0),),
                          device=device).long()
        condition = self.clip_model.text_encoder.encode(text,convert_to_tensor=True)
        noisy_data = self.noise_scheduler.add_noise(encoded_data, noise,t)
        # print(f"noisy_data:{noisy_data.shape}")
        # print(f"t:{t.shape}")
        # print(f"condition:{condition.shape}")
        predicted_noise = self.unet_model(noisy_data, t,condition)
        loss = self.criterion(predicted_noise, noise)

        # Log the loss with WandB
        wandb.log({"train_loss": loss.item(), "epoch": self.current_epoch, "batch_idx": batch_idx})

        print(f"Epoch: {self.current_epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
        if loss < self.best_loss:
            self.best_loss = loss
            self.save_model('condition_unet_cvae.pth')
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.unet_model.parameters(), lr=self.learning_rate)
        return optimizer

    def save_model(self, path):
        # Only save the UNet model's state dictionary
        torch.save(self.unet_model.state_dict(), path)




import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


# ²Ð²î¿é
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class CVAE(nn.Module):
    def __init__(self, latent_dim, onehot_dim, seq_length, feature_dim, hidden_dim, batch_size):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.onehot_dim = onehot_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=self.hidden_dim, max_len=self.seq_length)

        # Convolutional layers with residual blocks
        self.conv1_onehot = ResidualBlock(self.onehot_dim, 64)
        self.conv1_feature = ResidualBlock(self.feature_dim, 64)
        self.conv2 = ResidualBlock(64, 128)
        self.conv3 = ResidualBlock(128, hidden_dim)

        # Transformer encoder with dropout
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=2, dim_feedforward=512,
                                                   batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Cross Attention Encoder with increased attention heads
        self.cross_attention_encoder = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4)

        # Fully connected layers for encoding with dropout
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.seq_length * self.hidden_dim + self.seq_length * self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2 * latent_dim)
        )

        # Fully connected layers for decoding with dropout
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + self.seq_length * self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.seq_length * self.hidden_dim)  # Output shape: [seq_length * hidden_dim]
        )

        # Positional encoding for decoder
        self.pos_decoder = PositionalEncoding(d_model=self.hidden_dim, max_len=self.seq_length)

        # Convolutional layers to decode from hidden_dim back to onehot_dim
        self.deconv1 = nn.ConvTranspose1d(hidden_dim, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose1d(64, self.onehot_dim, kernel_size=3, stride=1,
                                          padding=1)  # Changed to onehot_dim

        # Transformer decoder with dropout
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=4, dim_feedforward=512,
                                                   batch_first=True, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    def encode(self, x, c, src_key_padding_mask=None):
        x = x.permute(0, 2, 1)  # [batch_size, onehot_dim, seq_length]
        x = self.conv1_onehot(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)  # [batch_size, seq_length, hidden_dim]

        c = c.permute(0, 2, 1)  # [batch_size, feature_dim, seq_length]
        c = self.conv1_feature(c)
        c = self.conv2(c)
        c = self.conv3(c)
        c = c.permute(0, 2, 1)

        h = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # Apply Transformer
        h = h.view(h.size(0), -1)  # Flatten [batch_size, seq_length * hidden_dim]

        if c is not None:
            # Cross attention between x and c
            x_attended, _ = self.cross_attention_encoder(c, x, x)  # Cross attention layer
            x_attended = x_attended.view(x_attended.size(0), -1)
            h = h + x_attended
            c_flat = c.reshape(c.size(0), -1)  # Flatten c to [batch_size, seq_length * feature_dim]
            x_cat = torch.cat((h, c_flat), dim=1)  # Concatenate h and flattened c
        else:
            x_cat = h

        h = self.encoder_fc(x_cat)  # Fully connected encoding
        mu, log_var = torch.chunk(h, 2, dim=1)  # Split into mu and log_var
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std  # Removed batch normalization for stability

    def decode(self, z, c, tgt_key_padding_mask=None):
        if c is not None:
            c_flat = c.view(c.size(0), -1)  # Flatten c to [batch_size, seq_length * feature_dim]
            z = torch.cat((z, c_flat), dim=1)  # Concatenate latent vector z and c

        h = self.decoder_fc(z)  # Decode through fully connected layers
        h = h.view(h.size(0), self.seq_length, self.hidden_dim)

        c = c.permute(0, 2, 1)  # [batch_size, feature_dim, seq_length]
        c = self.conv1_feature(c)
        c = self.conv2(c)
        c = self.conv3(c)
        c = c.permute(0, 2, 1)

        h_attention = self.transformer_decoder(h, c, tgt_key_padding_mask=tgt_key_padding_mask)  # Transformer decoder
        h = h_attention.permute(1, 2, 0)  # [batch_size, hidden_dim, seq_len]

        # Deconvolutions to return to [batch_size, seq_length, onehot_dim]
        h = self.deconv1(h)
        h = self.deconv2(h)
        h = self.deconv3(h)
        #print(h.shape)

        return h.permute(2, 0, 1)  # Final output [batch_size, seq_length, onehot_dim]


    def forward(self, x, c, src_key_padding_mask=None, tgt_key_padding_mask=None):
        mu, log_var = self.encode(x, c, src_key_padding_mask)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, c, tgt_key_padding_mask), mu, log_var


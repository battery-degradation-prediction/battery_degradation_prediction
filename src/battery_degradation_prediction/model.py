"""model module"""
import numpy as np
import torch
from torch import nn, Tensor
from torch import optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import math


from torch.utils.data import dataset


class Net(nn.Module):
    """TODO"""

    def __init__(self, input_shape):
        super().__init__()
        # an affine operation: y = Wx + b
        # window_size = input_shape[0]
        self.fc1 = nn.Linear(20, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        """TODO"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc4(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, input_size, d_model, nhead, num_layers, output_size, latent_size, dropout=0.1
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_size = output_size
        self.latent_size = latent_size

        # Define the positional encoding layer
        self.pos_encoder = nn.Sequential(nn.Linear(input_size[1], d_model), nn.Tanh())

        # Define the transformer encoder layer
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        # Define the output layer
        self.fc = nn.Linear(d_model, output_size)
        self.fc_features = nn.Linear(d_model*input_size[0], input_size[1])
        self.fc_to_latent = nn.Linear(d_model*input_size[0], latent_size)
        self.fc_latent_to_original = nn.Linear(latent_size, input_size[0]*input_size[1])

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]

        # Add positional encoding to the input
        # x = self.fc1(x)
        x = self.pos_encoder(x)

        # Transpose the input to match the expected shape for the transformer
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, input_size]
        # Pass the input sequence through the transformer encoder layer
        out = self.transformer_encoder(x)  # [seq_len, batch_size, d_model]
        x_out = out.permute(1, 0, 2) # [batch_size, seq_len, d_model]
        x_out = self.fc_features(torch.flatten(x_out, 1)) # [batch_size, input_size]
        x_out = x_out.view(-1, self.input_size[1])
        # Pass the output of the transformer through the output layer
        out = self.fc(out[-1, :, :])  # [batch_size, output_size]

        return out, x_out
    
class TransformerReduction(nn.Module):
    def __init__(
        self, input_size, d_model, nhead, num_layers, output_size, latent_size, dropout=0.1
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_size = output_size
        self.latent_size = latent_size

        # Define the positional encoding layer
        self.pos_encoder = nn.Sequential(nn.Linear(input_size[1], d_model), nn.Tanh())

        # Define the transformer encoder layer
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        # Define the output layer
        self.fc = nn.Linear(d_model, output_size)
        self.fc_to_latent = nn.Linear(d_model*input_size[0], latent_size)
        self.fc_latent_to_original = nn.Linear(latent_size, input_size[0]*input_size[1])

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]

        # Add positional encoding to the input
        x = self.pos_encoder(x)

        # Transpose the input to match the expected shape for the transformer
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, input_size]
        # Pass the input sequence through the transformer encoder layer
        out = self.transformer_encoder(x)  # [seq_len, batch_size, d_model]
        x_out = out.permute(1, 0, 2) # [batch_size, seq_len, d_model]
        latent = self.fc_to_latent(torch.flatten(x_out, 1)) # [batch_size, latent_size]
        x_out = self.fc_latent_to_original(latent)
        x_out = x_out.view(-1, self.input_size[0], self.input_size[1])

        return x_out, latent


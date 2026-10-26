
import torch
import torch.nn as nn
from dataclasses import dataclass
from utils import logger
from torch.utils.data import TensorDataset, DataLoader


# --- Global configuration ---

@dataclass
class Config:

    # Model architecture
    input_size = 2
    hidden_size = 32
    kernel_size = 5
    pool_size = 16

    # Training
    learning_rate = 0.001
    batch_size = 32
    n_epochs = 50
    rnd_seed = 10


# -----------------------------
# 1. --- Temporal CNN ---
# -----------------------------

class TrajectoryCNN(nn.Module):

    def __init__(self, config, n_points_future):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(config.input_size, config.hidden_size,
                      kernel_size=config.kernel_size, padding=2, dilation=1),
            nn.GELU(),

            nn.Conv1d(config.hidden_size, config.hidden_size,
                      kernel_size=config.kernel_size, padding=4, dilation=2),
            nn.GELU(),

            nn.Conv1d(config.hidden_size, config.hidden_size,
                      kernel_size=config.kernel_size, padding=8, dilation=4),
            nn.GELU(),

            # Compress the temporal representation to fewer positions
            nn.AdaptiveAvgPool1d(config.pool_size))

        self.predict_future = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_size*config.pool_size, 128),
            nn.GELU(),
            nn.Linear(128, n_points_future*2))
        
    def forward(self, x):

        ntraj, ntime, nfeat = x.shape # (batch, time, features - xy)
        last_position = x[:,-1:,:]  # last observed position (batch, 1, 2)

        # Conv1d inside the CNN expects (batch, channels, time)
        x = x.transpose(1,2)  # (batch, features - xy, time)
        features = self.features(x)

        # Predict displacement at each future step
        displacement = self.predict_future(features)
        displacement = displacement.reshape(ntraj, ntime, nfeat)

        # Accumulate displacements from the last observed position
        future = last_position + torch.cumsum(displacement, dim=1)

        return future
    

# Training function

def train_CNN(X_train, Y_train, future_steps=None, config=None):

    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = TrajectoryCNN(config, future_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    tr_losses = []

    def loss_fn(prediction, target, smoothness_weight=0):

        # Ordinary positional prediction error (mse or L1)
        mse = nn.functional.mse_loss(prediction, target)
        l1 = nn.functional.smooth_l1_loss(prediction, target, beta=1)

        # Smoothing
        acceleration = torch.diff(prediction, n=2, dim=1)
        smoothness = torch.mean(acceleration ** 2)

        return l1 + smoothness_weight * smoothness

    for n in range(config.n_epochs):

        logger.info(f'Epoch {n+1} ...')
        model.train()
        train_loss = 0

        for X_batch, Y_batch in loader:

            optimizer.zero_grad()
            prediction = model(X_batch)
            loss = loss_fn(prediction, Y_batch, smoothness_weight=1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        tr_losses.append(train_loss / len(loader))

    return model, tr_losses


# --------------------------------
# 2. --- Autoregressive GRU ---
# --------------------------------

class AutoregressiveGRU(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = nn.GRU(input_size=config.input_size, 
                          hidden_size=config.hidden_size,
                          batch_first=True)

        self.decoder = nn.GRU(input_size=config.input_size, 
                          hidden_size=config.hidden_size,
                          batch_first=True)

        self.output_layer = nn.Linear(config.hidden_size, config.input_size)

    def forward(self, past, future=None, future_steps=None, teacher_forcing=False):

        _, hidden = self.encoder(past)
        last_observed = past[:, -1:, :] # last observed position
        decoder_input = last_observed
        predictions = [last_observed]

        for t in range(future_steps-1):

            # Predict one future step
            output, hidden = self.decoder(decoder_input, hidden)
            prediction = self.output_layer(output)
            predictions.append(prediction)

            # Training: use true previous coordinate.
            # Evaluation: use the model's previous prediction.
            if teacher_forcing and future is not None:
                decoder_input = future[:, t:t+1, :]
            else:
                decoder_input = prediction

        return torch.cat(predictions, dim=1)


# Training function

def train_GRU(X_train, Y_train, config=None, teacher_forcing=False):

    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = AutoregressiveGRU(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    tr_losses = []

    def loss_fn(prediction, target, smoothness_weight=0):

        # Ordinary positional prediction error (mse or L1)
        mse = nn.functional.mse_loss(prediction, target)
        l1 = nn.functional.smooth_l1_loss(prediction, target, beta=1)

        # Smoothing
        acceleration = torch.diff(prediction, n=2, dim=1)
        smoothness = torch.mean(acceleration ** 2)

        return mse + smoothness_weight * smoothness

    for _ in range(config.n_epochs):

        logger.info(f'Epoch {_+1} ...')
        model.train()
        train_loss = 0

        for X_batch, Y_batch in loader:

            fs = Y_batch.shape[1]
            optimizer.zero_grad()

            if teacher_forcing:
                # Next prediction based on true last position
                prediction = model(X_batch, future=Y_batch, 
                                future_steps=fs, teacher_forcing=teacher_forcing)
            else:
                # Next prediction based on model's last prediction
                prediction = model(X_batch, future_steps=fs)

            loss = loss_fn(prediction[:, 1:, :], Y_batch[:, 1:, :], 
                           smoothness_weight=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.shape[0]

        tr_losses.append(train_loss / len(loader))

    return model, tr_losses

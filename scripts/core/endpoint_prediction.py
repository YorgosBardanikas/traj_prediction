
import os
import mne
import torch
import pickle
import numpy as np
import torch.nn as nn
from scripts import utils
from scripts.utils import logger, path_to_save
from scripts.helper_funcs import reveal_trajectory
from torch.utils.data import TensorDataset, DataLoader

# --- RNN class ---

class EndpointGRU(nn.Module):

    def __init__(self, input_size=2, hidden_size=64, output_size=2):
        super().__init__()

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True)

        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, time, features - xy)
        _, hidden = self.gru(x) # (1, batch, hidden_size)
        output = self.output(hidden[0]) # (batch, 2)
        return output
    

# --- Training function ---

def train_GRU(X_train, Y_train, n_epochs=50):

    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = EndpointGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    tr_losses = []

    for _ in range(n_epochs):

        model.train()
        train_loss = 0

        for X_batch, Y_batch in loader:
            # Remove old gradients
            optimizer.zero_grad()
            # Forward pass
            prediction = model(X_batch)
            # Calculate error
            loss = loss_fn(prediction, Y_batch)
            # Backpropagation
            loss.backward()
            # Update model parameters
            optimizer.step()
            # Accumulate loss
            train_loss += loss.item()

        tr_losses.append(train_loss/len(loader))

    return model, tr_losses


# --- Prediction and error ---

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X)
    
def denormalize(predictions, endpoints, std, mean):
    predictions, endpoints = predictions.numpy(), endpoints.numpy()
    std, mean = std.reshape(1,2), mean.reshape(1,2)
    predictions_norm = predictions * std + mean
    endpoints_norm = endpoints * std + mean
    return predictions_norm, endpoints_norm

def calculate_error(predictions, endpoints):
    return np.linalg.norm(predictions - endpoints, axis=1)
    

if __name__ == '__main__':
    
    # Constants
    utils.set_seed(10)
    session_names = utils.get_session_group('j')[:2]
    onset = 'hand' # alignment event: targ or hand onset
    t_start = -0.1 # starting time point (in sec)
    dns = 5 # downsample factor
    train_percent = 0.7 # train on 70% of trials
    n_shuffles = 10 # number of shuffles for null models
    conditions = [2,3,4] # target in the sequence
    reveal_percentages = np.arange(10,91,10) # percentage of trajectory revealed to model

    # Load the data
    bhv_list = utils.load_mne_epochs(session_names, onset)
    bhv = mne.concatenate_epochs(bhv_list, on_mismatch='ignore')

    # Extract x,y coordinates of hand movement
    target_rank = bhv.metadata['Target Rank'].to_numpy()
    hand_x = bhv.get_data(picks=['Hand X-Position'], tmin=t_start).squeeze()
    hand_y = bhv.get_data(picks=['Hand Y-Position'], tmin=t_start).squeeze()
    hand_xy = np.array([hand_x, hand_y]).transpose(1,2,0) # ntrials, ntime, xy
    hand_xy = hand_xy[:,::dns,:] # downsample time 

    # Split trials to train and test
    ntrials = hand_xy.shape[0]
    rng = np.random.default_rng(seed=10)
    inds = rng.permutation(ntrials)
    split = int(train_percent * ntrials)
    train_inds = inds[:split]
    test_inds = inds[split:]

    train_data = hand_xy[train_inds]
    test_data = hand_xy[test_inds]
    rank_train = target_rank[train_inds]
    rank_test = target_rank[test_inds]

    # Normalize based on train set
    mean = train_data.mean(axis=(0,1), keepdims=True)
    std = train_data.std(axis=(0,1), keepdims=True)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    # Initialize dictionaries to save results
    real_errors_all = {c: {} for c in conditions}
    null_errors_all = {c: {} for c in conditions}
    losses = {c: {} for c in conditions}
    real_states = {c: {} for c in conditions}
    real_predictions = {c: {} for c in conditions}
    null_predictions = {c: {} for c in conditions}

    for condition in conditions:

        logger.info(f' Condition {condition}')

        # Select trials belonging to this condition
        train_mask = rank_train == condition
        test_mask = rank_test == condition

        Xtr_condition = train_data[train_mask]
        Ytr_condition = train_data[train_mask,-1,:]
        Ytr_condition = torch.tensor(Ytr_condition, dtype=torch.float32)

        Xte_condition = test_data[test_mask]
        Yte_condition = test_data[test_mask,-1,:]
        Yte_condition = torch.tensor(Yte_condition, dtype=torch.float32)

        # Reveal progressively more of the trajectory
        for percentage in reveal_percentages:

            logger.info(f'{percentage}% spatial reveal ...')

            Xtr = reveal_trajectory(Xtr_condition, percentage)
            Xte = reveal_trajectory(Xte_condition, percentage)
            Xtr = torch.tensor(Xtr, dtype=torch.float32)
            Xte = torch.tensor(Xte, dtype=torch.float32)

            # ----------------------
            # Real model
            # ----------------------

            model_real, loss = train_GRU(Xtr, Ytr_condition, n_epochs=50)
            prediction = predict(model_real, Xte)
            prediction, endpoint = denormalize(prediction, Yte_condition, std, mean)

            # Euclidean endpoint error for each test trial
            real_errors = calculate_error(prediction, endpoint)

            # Save errors, losses, predictions, and weights
            real_errors_all[condition][percentage] = real_errors
            losses[condition][percentage] = np.array(loss)
            real_predictions[condition][percentage] = prediction
            real_states[condition][percentage] = {k: v.detach().clone()
                                                for k, v in model_real.state_dict().items()}

            # ----------------------
            # Shuffled-label nulls
            # ----------------------

            null_errors = []
            null_preds = []

            for shuffle in range(n_shuffles):

                logger.info(f'Shuffle {shuffle + 1}/{n_shuffles}')

                # Shuffle endpoints only within this condition's training trials
                permutation = torch.randperm(len(Ytr_condition))
                Y_shuffled = Ytr_condition[permutation]

                model_null, _ = train_GRU(Xtr, Y_shuffled, n_epochs=50)
                predictions = predict(model_null, Xte)
                predictions, endpoints = denormalize(predictions, Yte_condition, std, mean)
                err = calculate_error(predictions, endpoints)

                null_errors.append(err)
                null_preds.append(predictions)

            null_errors_all[condition][percentage] = np.array(null_errors)
            null_predictions[condition][percentage] = np.array(null_preds)

    # Save results
    results = {
        'real_errors': real_errors_all,
        'null_errors': null_errors_all,
        'train_losses': losses,
        'real_predictions': real_predictions,
        'null_predictions': null_predictions,
        'real_states': real_states,
        'train_indices': train_inds,
        'test_indices': test_inds,
        'rank_train': rank_train,
        'rank_test': rank_test,
        'settings': {
            'conditions': conditions,
            'reveal_percentages': reveal_percentages}
        }

    with open(os.path.join(path_to_save, 'endpoint_prediction.pkl'), 'wb') as f:
        pickle.dump(results, f)
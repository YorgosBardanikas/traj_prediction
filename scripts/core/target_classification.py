
import os
import mne
import utils
import torch
import pickle
import numpy as np
import torch.nn as nn
from utils import logger, path_to_save
from helper_funcs import reveal_trajectory
from torch.utils.data import TensorDataset, DataLoader

# --- RNN class ---

class TargetGRU(nn.Module):

    def __init__(self, input_size=2, hidden_size=64, n_targets=6):
        super().__init__()

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True)

        self.output = nn.Linear(hidden_size, n_targets)

    def forward(self, x):
        # x: (batch, time, features - hand_xy)
        _, hidden = self.gru(x) # (1, batch, hidden_size)
        output = self.output(hidden[0]) # (batch, 6), raw logits
        return output


# --- Training function ---

def train_GRU_classifier(X_train, Y_train, n_epochs=50):

    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TargetGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    tr_losses = []

    for _ in range(n_epochs):

        model.train()
        train_loss = 0

        for X_batch, Y_batch in loader:

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        tr_losses.append(train_loss/len(loader))

    return model, tr_losses


def predict(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        return predictions.numpy(), probabilities.numpy()


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
    target_id = bhv.metadata['currentTargetID'].to_numpy()-2 # 0 to 5
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
    target_train = target_id[train_inds]
    target_test = target_id[test_inds]
    Y_train = torch.tensor(target_train, dtype=torch.long)

    # Normalize based on train set
    mean = train_data.mean(axis=(0,1), keepdims=True)
    std = train_data.std(axis=(0,1), keepdims=True)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    # Initialize dictionaries to save results
    real_results = {c: {} for c in conditions}
    null_results = {c: {} for c in conditions}
    real_states = {c: {} for c in conditions}
    losses = {c: {} for c in conditions}

    for condition in conditions:

        logger.info(f' Condition {condition}')

        # Select trials belonging to this condition
        train_mask = rank_train == condition
        test_mask = rank_test == condition

        Xtr_condition = train_data[train_mask]
        Ytr_condition = Y_train[train_mask]

        Xte_condition = test_data[test_mask]
        Yte_condition = target_test[test_mask]

        # Reveal progressively more of the trajectory
        for percentage in reveal_percentages:

            logger.info(f'{percentage}% spatial reveal ...')

            Xtr = reveal_trajectory(Xtr_condition, percentage)
            Xte = reveal_trajectory(Xte_condition, percentage)
            Xtr = torch.tensor(Xtr, dtype=torch.float32)
            Xte = torch.tensor(Xte, dtype=torch.float32)

            # ----------------------
            # Real classifier
            # ----------------------

            model, loss = train_GRU_classifier(Xtr, Ytr_condition, n_epochs=50)
            predictions, probabilities = predict(model, Xte)
            accuracy = (predictions == Yte_condition).mean()

            real_results[condition][percentage] = {'accuracy': accuracy,
                                                'predictions': predictions,
                                                'probabilities': probabilities,
                                                'true_labels': Yte_condition}
            real_states[condition][percentage] = {k: v.detach().clone()
                                                    for k, v in model.state_dict().items()}
            losses[condition][percentage] = np.array(loss)

            # ----------------------
            # Shuffled-label nulls
            # ----------------------

            null_accuracies = []
            null_probabilities = []
            null_model_states = []

            for shuffle in range(n_shuffles):

                permutation = torch.randperm(len(Ytr_condition))
                Y_shuffled = Ytr_condition[permutation]

                model_null, _ = train_GRU_classifier(Xtr, Y_shuffled, n_epochs=50)
                preds_null, probs_null = predict(model_null, Xte)

                null_acc = (preds_null == Yte_condition).mean()
                null_accuracies.append(null_acc)
                null_probabilities.append(probs_null)
                
            null_results[condition][percentage] = {'accuracies': np.array(null_accuracies),
                                                'probabilities': np.array(null_probabilities)}

    # Save results
    results = {
        'real_results': real_results,
        'null_results': null_results,
        'train_losses': losses,
        'real_states': real_states,
        'train_indices': train_inds,
        'test_indices': test_inds,
        'rank_train': rank_train,
        'rank_test': rank_test,
        'settings': {
            'conditions': conditions,
            'reveal_percentages': reveal_percentages}
        }

    with open(os.path.join(path_to_save, 'target_classification.pkl'), 'wb') as f:
        pickle.dump(results, f)
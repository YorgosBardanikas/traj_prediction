
import os
import mne
import torch
import pickle
import numpy as np
from scripts import utils
import scripts.helper_funcs as funcs
from trajectory_models import Config, train_CNN, train_GRU


def predict(model_name, model, X_test, X_train, n_points):
    
    model.eval()
    with torch.no_grad():

        if model_name == 'CNN':
            predicted_future = model(X_test)
            predicted_future_train = model(X_train)

        elif model_name == 'GRU':
            predicted_future = model(X_test, future_steps=n_points)
            predicted_future_train = model(X_train, future_steps=n_points)
        else:
            raise ValueError(f'Unknown model: {model_name}')
        
    return predicted_future.numpy(), predicted_future_train.numpy()


def linear_extrapolation(X, n_points_future, n_velocity_points=5):

    # Last observed position
    last_position = X[:,-1:,:]
    recent = X[:,-n_velocity_points:,:]

    time = torch.arange(n_velocity_points, dtype=X.dtype, device=X.device)
    time_ = time - time.mean()
    future_time = torch.arange(1, n_points_future+1, dtype=X.dtype, device=X.device)
    
    # Estimate velocity with a linear fit over the last recent points
    # Slope of position against time, separately for x and y
    velocity = (time_[None, :, None] * recent).sum(dim=1) / (time_ ** 2).sum()

    # Linear extrapolation (n_trials, n_points_future, xy)
    prediction = (last_position + future_time[None, :, None] * velocity[:, None, :])

    return prediction.numpy()


if __name__ == '__main__':

    # Select which model to use: multi-step CNN or autoregressive GRU
    model_name = 'GRU'

    # Constants
    utils.set_seed(10)
    config = Config()
    session_names = utils.get_session_group('j')[:2]
    onset = 'hand' # alignment event: targ or hand onset
    t_start = -0.1 # starting time point (in sec)
    dns = 5 # downsample factor
    train_percent = 0.7 # train on 70% of trials

    # Model dictionaries
    suffix = {'CNN': 'multi-step',
              'GRU': 'autoregressive'}
    train_function = {'CNN': train_CNN,
                      'GRU': train_GRU}
    model_params = {'CNN': {'config':config},
                    'GRU': {'config':config,
                            'teacher_forcing':False}}
    
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
    ntrials, ntimes, _ = hand_xy.shape
    rng = np.random.default_rng(seed=config.rnd_seed)
    inds = rng.permutation(ntrials)
    split = int(train_percent * ntrials)
    train_inds = inds[:split]
    test_inds = inds[split:]

    train_data = hand_xy[train_inds]
    test_data = hand_xy[test_inds]
    target_id_test = target_id[test_inds]

    # Normalize based on train set
    mean = train_data.mean(axis=(0,1), keepdims=True)
    std = train_data.std(axis=(0,1), keepdims=True)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    # The model sees X (initial part of hand x,y trajectory)
    # and predicts Y (final part of hand x,y trajectory)
    n_points = ntimes//2 # points to resample time after splitting trajectory
    x_train, y_train = funcs.split_trajectory(train_data, n_points)
    x_test, y_test = funcs.split_trajectory(test_data, n_points)

    X_train = torch.tensor(x_train, dtype=torch.float32)
    X_test = torch.tensor(x_test, dtype=torch.float32)
    Y_train = torch.tensor(y_train, dtype=torch.float32)

    straightness, path_length = funcs.calculate_straightness(X_test)
    model_params['CNN'].update({'future_steps': n_points})

    train = train_function[model_name]
    model, losses = train(X_train, Y_train, **model_params[model_name])
    
    predicted_future, predicted_future_train = predict(
        model_name, model, X_test, X_train, n_points)

    # Baseline errors: linear extrapolation of the recent past trajectory
    linear_prediction = linear_extrapolation(X_test, n_points, 
                                             n_velocity_points=5)
    
    # Denormalize to real scale
    true_past = X_test.numpy() * std + mean
    true_future = y_test * std + mean
    true_future_train = Y_train.numpy() * std + mean
    pred_future = predicted_future * std + mean
    pred_future_train = predicted_future_train * std + mean
    linear_prediction = linear_prediction * std + mean

    # Calculate Euclidean errors over xy
    errors = np.linalg.norm(pred_future - true_future, axis=2)
    errors_train = np.linalg.norm(pred_future_train - true_future_train, axis=2)
    linear_errors = np.linalg.norm(linear_prediction - true_future, axis=2)

    # Save results
    results = {
        'errors': errors,
        'errors_train': errors_train,
        'linear_errors': linear_errors,
        'predictions': pred_future,
        'true_future': true_future,
        'true_past': true_past,
        'losses': np.array(losses),
        'trajectory_vars': {
            'straightness': straightness,
            'path_length': path_length,
            'target_id': target_id_test}
    }

    filename = f'trajectory_prediction_{suffix[model_name]}.pkl'
    with open(os.path.join(utils.path_to_save, filename), 'wb') as f:
        pickle.dump(results, f)
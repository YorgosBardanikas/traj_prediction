
import torch
import numpy as np

def reveal_trajectory(trajectories, percentage, n_points=50):

    n_trials, _ , n_feats = trajectories.shape
    revealed_trajectory = np.zeros((n_trials, n_points, n_feats))
    fraction = percentage/100

    def resample(segment, n_points):
        old_t = np.linspace(0, 1, segment.shape[0])
        new_t = np.linspace(0, 1, n_points)
        resamp = [np.interp(new_t, old_t, segment[:, axis])
                  for axis in range(n_feats)]

        return np.column_stack(resamp)

    for i, trajectory in enumerate(trajectories):

        step_displacements = np.diff(trajectory, axis=0)
        step_distances = np.linalg.norm(step_displacements, axis=1)
        # Append 0 to match trajectory indices before np.diff
        cumulative_distance = np.concatenate([[0], np.cumsum(step_distances)])
        total_distance = cumulative_distance[-1]

        # Find revealed distance
        revealed_distance = fraction * total_distance

        # Find the last index where cumulative sum is smaller 
        # or equal to distance to be revealed
        idx = np.where(cumulative_distance <= revealed_distance)[0][-1]

        # Resample each segment to a fixed length
        revealed_trajectory[i] = resample(trajectory[:idx], n_points)

    return revealed_trajectory


def split_trajectory(trajectories, n_points):

    n_trials, _ , n_feats = trajectories.shape
    past = np.zeros((n_trials, n_points, n_feats))
    future = np.zeros((n_trials, n_points, n_feats))

    def resample(segment, n_points):
        old_t = np.linspace(0, 1, segment.shape[0])
        new_t = np.linspace(0, 1, n_points)
        resamp = [np.interp(new_t, old_t, segment[:, axis])
                  for axis in range(n_feats)]

        return np.column_stack(resamp)

    for i, trajectory in enumerate(trajectories):

        step_displacements = np.diff(trajectory, axis=0)
        step_distances = np.linalg.norm(step_displacements, axis=1)
        # Append 0 to match trajectory indices before np.diff
        cumulative_distance = np.concatenate([[0], np.cumsum(step_distances)])
        total_distance = cumulative_distance[-1]
        half_distance = total_distance/2
        # Find the last index where cumulative sum is smaller or equal to half
        idx = np.where(cumulative_distance <= half_distance)[0][-1]

        # Resample each segment to a fixed length
        past[i] = resample(trajectory[:idx], n_points)
        future[i] = resample(trajectory[idx:], n_points)

    return past, future


def calculate_straightness(x):
    # Calculate trajectory straightness (0-1: 1 most straight)
    step_displacements = torch.diff(x, dim=1)
    step_distances = torch.linalg.vector_norm(step_displacements, dim=2)
    path_length = step_distances.sum(dim=1)

    # Straight line distance
    net_displacement = torch.linalg.vector_norm(
        x[:,-1,:] - x[:,0,:], dim=1)
    
    # Straightness (batch,)
    straightness = net_displacement / path_length

    return straightness.numpy(), path_length.numpy()


# --- Plotting functions ---

def plot_workspace(xTarg, yTarg, ax):

    from matplotlib.patches import Circle, Ellipse

    for x, y in zip(xTarg, yTarg):

        center = (x, y)
        width, height = 2, 2
        target = Circle(center, radius=0.325, color='k')
        valid_target_area = Ellipse(center, width, height, linestyle='dashed',
                                    linewidth=1, color='k', fill=False)
        ax.add_patch(target)
        ax.add_patch(valid_target_area)
        ax.set_aspect('equal', adjustable='box')


def plot_trajectory(ax, i, true_past, true_future, pred_future, 
                    targ_x, targ_y, errors):

    true_past, true_future, pred_future, error = true_past[i], true_future[i],\
                                                pred_future[i], errors[i]
    
    plot_workspace(targ_x, targ_y, ax)
    ax.plot(true_past[:,0], true_past[:,1], label='True past', color='grey')
    ax.scatter(true_past[-1,0], true_past[-1,1], marker='o', color='grey')
    ax.plot(true_future[:,0], true_future[:,1], label='True future', color='b')
    ax.plot(pred_future[:,0], pred_future[:,1], label='Predicted future', color='g')
    ax.set_title(f'Mean error = {error:.3f}')
    ax.legend()
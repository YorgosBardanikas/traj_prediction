
import os
import sys
import mne
import random
import logging
import torch
import numpy as np

# --- Handle paths ---
CWD = os.getcwd()
if 'envau' in CWD:
    path_to_load = '/envau/work/comco/bardanikas.g/Vision4Action/V4A_data'
    path_to_save = '/envau/work/comco/bardanikas.g/Vision4Action/MLtorch/data'
elif 'C:' in CWD or 'c:' in CWD:
    # path_to_load = 'E:/G.Bardanikas/Aix-Marseille University/V4A_data'
    path_to_load = 'C:/Users/georg/OneDrive/Έγγραφα/PyTorch/data'
    path_to_save = 'C:/Users/georg/OneDrive/Έγγραφα/PyTorch/data/results'

# --- Handle logger ---
logging.basicConfig(level=logging.INFO, format="%(message)s",
                    stream=sys.stdout, force=True)

logger = logging.getLogger(__name__)

# --- Utility functions ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_session_group(sub):
    if sub == 'j':
        session_names = ['j210204-land-001','j210204-land-002',
                        'j210208-land-001','j210211-land-001',
                        'j210211-land-002','j210212-land-001','j210215-land-002',
                        'j210215-land-003','j210215-land-004','j210216-land-001',
                        'j210216-land-003','j210217-land-001','j210218-land-003',
                        'j210219-land-001','j210225-land-001']
    elif sub == 'y': 
        session_names = ['y190829-land-001','y190902-land-001','y190903-land-001', 
                        'y190903-land-002','y190905-land-001','y190906-land-001', 
                        'y190906-land-002','y190909-land-001','y190909-land-002',
                        'y190909-land-003','y190911-land-001','y190911-land-002', 
                        'y190913-land-001','y190913-land-002']
    return session_names

def load_mne_epochs(session_names, onset, targetids=[2,3,4]):
    epochs_list = []
    for session_name in session_names:
        for targetid in targetids:
            epoch_file = f'{session_name}_bhv_{onset}{targetid}-epo.fif'
            epoch = mne.read_epochs(
                os.path.join(path_to_load, session_name, epoch_file), verbose='error')
            epochs_list.append(epoch) 
    return epochs_list

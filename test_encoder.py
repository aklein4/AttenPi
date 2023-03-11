


import torch
import torch.nn as nn
import torch.nn.functional as F

from variational_trajectory import VariationalTrajectory
from train_utils import DEVICE

import matplotlib.pyplot as plt
import numpy as np


TAUS_FILE = "data/val_taus.pt"

MODEL_FILE = "data/latent_policy.pt"


def main():

    model = VariationalTrajectory()
    model.load_state_dict(torch.load(MODEL_FILE))
    model = model.to(DEVICE)
    
    states, actions, _ = torch.load(TAUS_FILE, map_location=DEVICE)
    actions = actions[:,:,:4] # messed up shape
    
    torch.no_grad()
    model.eval()
    
    ind = -1
    while True:
        ind += 1
        
        s, a = states[ind], actions[ind]
        a_hat, _, _ = model((s, a))
        
        print('\n')
        for i in range(a.shape[0]):
            print("{} -> {}".format(torch.argmax(a[i].int()).item(), np.round(a_hat[i].detach().cpu().numpy(), 3)))

        input('\n... ')


if __name__ == '__main__':
    main()
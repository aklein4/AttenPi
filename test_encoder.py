


import torch
import torch.nn as nn
import torch.nn.functional as F

from variational_trajectory import VariationalTrajectory
from train_utils import DEVICE

import matplotlib.pyplot as plt


TAUS_FILE = "data/val_taus.pt"

MODEL_FILE = "data/reckoner.pt"


def main():

    model = VariationalTrajectory()
    model.load_state_dict(torch.load(MODEL_FILE))
    model = model.to(DEVICE)
    
    trajectories, _, _ = torch.load(TAUS_FILE)
    trajectories = trajectories.to(DEVICE)
    
    torch.no_grad()
    model.eval()
    
    ind = -1
    while True:
        ind += 1
        
        traj_1 = trajectories[ind]
        mu_1, sig_1 = model.encode(traj_1)
        
        traj_2 = trajectories[ind+1]
        mu_2, sig_2 = model.encode(traj_2)
        
        KL_div = torch.sum(torch.log(sig_2 / sig_1) + (sig_1**2 + (mu_1 - mu_2)**2)/(2*sig_2**2) - 0.5)
        
        pred_x = []
        pred_y = []
        targ_x = []
        targ_y = []
        
        for i in range(traj_2.shape[0]):
            pred_x.append(traj_2[i, 0].item())
            pred_y.append(traj_2[i, 1].item())
            targ_x.append(traj_1[i, 0].item())
            targ_y.append(traj_1[i, 1].item())
        
        plt.plot(targ_x, targ_y)
        plt.plot(pred_x, pred_y)
        plt.savefig("logs/exemple_decoding.png")
        plt.close()
        
        input("\nKL Divergence: {} ...".format(round(KL_div.item(), 3)))


if __name__ == '__main__':
    main()
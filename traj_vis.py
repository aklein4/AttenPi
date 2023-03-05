
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

LEN = 100
SEQ_LEN = 10
R_EPS = 0.05

GAMMA = 0.8

def main():
    rewards = np.zeros((100,))

    for i in range(100):
        if random.random() < R_EPS:
            rewards[i] = -1
        if random.random() > 1-R_EPS:
            rewards[i] = 1
    rewards[-1] = 10

    v_x = list(range(0, LEN, SEQ_LEN))
    vals = []
    for i in v_x:
        vals.append(
            np.sum(rewards[i:i+SEQ_LEN]) + np.sum(rewards[i+SEQ_LEN:] * GAMMA ** np.arange(rewards.shape[0]-i-SEQ_LEN))
        )
        vals[-1] -= vals[0]

    r_x = np.arange(rewards.shape[0])[rewards != 0]
    r_y = rewards[rewards != 0]

    plt.scatter(v_x, vals, c='b')
    plt.scatter(r_x, r_y, c='r')
    plt.show()


if __name__ == '__main__':
    main()
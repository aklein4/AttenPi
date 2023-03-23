
import csv
import numpy as np
import matplotlib.pyplot as plt

VERSIONS = [
    'baseline',
    'no_reg',
    'enc_reg',
    'semi_reg',
    'all_reg'
]

TARGET = 'avg_reward'

def getData(version):
    header = None
    data = {}
    with open('logs/{}.csv'.format(version), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if data == {}:
                header = row
                for k in row:
                    data[k] = []
            elif len(row) < len(header):
                continue
            else:
                try:
                    for i in range(len(row)):
                        data[header[i]].append(float(row[i])*100)
                except:
                    continue
    
    for k in data.keys():
        data[k] = np.array(data[k])

    return data


def main():

    data = []
    for v in VERSIONS:
        data.append(getData(v)[TARGET])
    
    data = np.stack(data)

    plt.plot(data.T)
    plt.legend([
        'RL Baseline',
        'No Regularization',
        'L_mu Regularization',
        'L_s Regularization',
        'All Regularization'
    ])
    plt.title("Average Greedy Action Reward")
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward (1000 steps)")
    plt.savefig("reward.png")


if __name__ == '__main__':
    main()
import sys
import torch
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

if len(sys.argv) is not 2:
    print("Two arguments required.")
    sys.exit(1)

model_path = sys.argv[1]
fig_path = "./"
if len(sys.argv) == 3:
    fig_path = sys.argv[3]

print_batchnorm = False

if __name__ == '__main__':
    model = torch.load(model_path)
    for name, param in model.items():
        # Skip if we want to ignore batch norm layers
        if name[:2] == 'bn' and not print_batchnorm:
            continue

        data = param.cpu().numpy()
        print(name, data.shape)
        data = np.reshape(data, -1)

        (mu, sigma) = norm.fit(data)

        # the histogram of the data
        # bin_weights = np.ones_like(data)/float(len(data))
        n, bins, patches = plt.hist(data, 'auto', facecolor='green', alpha=0.75)

        # plot
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title(r'$\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
        plt.grid(True)

        plt.show()

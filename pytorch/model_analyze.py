import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) is not 2:
    print("Two arguments required.")
    sys.exit(1)

model_path = sys.argv[1]
fig_path = "./"
if len(sys.argv) == 3:
    fig_path = sys.argv[3]

if __name__ == '__main__':
    model = torch.load(model_path)
    for name, param in model.items():
        data = param.cpu().numpy()
        print(name, data.shape)
        data = np.reshape(data, -1)
        n, bins, patches = plt.hist(data, 100)
        plt.xlabel(name)
        plt.ylabel('Count')
        plt.title('Histogram of ' + name)
        plt.grid(True)
        # plt.savefig(fig_path + 'hist-' + name.replace('.', '-'), dpi=300)
        plt.show()

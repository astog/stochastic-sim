from progressbar import printProgressBar
import matplotlib.pyplot as plt
import stoch_test as st
import numpy as np
import argparse
import os
parser = argparse.ArgumentParser(description='Numpy Stochastic Grapher')
parser.add_argument('--samples', type=int, default=1000, metavar='N',
                    help='number of samples to take (default: 1000)')
parser.add_argument('--length-max', type=int, default=1024, metavar='N',
                    help='maximum length (default: 1024)')
parser.add_argument('--bipolar', action='store_true', default=False,
                    help='Whether to test using [-1, 1] (default: unipolar [0, 1])')
parser.add_argument('--deterministic', action='store_true', default=False,
                    help='Use deterministic way to generate stochastic stream (default: false)')
parser.add_argument('--save', action='store_true', default=False,
                    help='Enables saving of data to file and loading from it if it exists')
parser.add_argument('--csv-file', type=str, default="stoch_graph_data.csv",
                    help='Log file to save/load info from')
args = parser.parse_args()


lengths = np.arange(1, args.length_max + 1, 1)
file_loaded = False
stats = None


# if using save, the load data if file exists, else compute
if args.save and os.path.isfile(args.csv_file):
    print("LOADING FILE: " + args.csv_file)
    stats = np.loadtxt(args.csv_file, delimiter=',')
    file_loaded = True


def compute(length):
    return np.array(st.get_approx_stats(args.samples, length, args.bipolar, args.deterministic))


if not file_loaded:
    stats = np.zeros((args.length_max, 2))

    for i in range(lengths.size):
        printProgressBar(i, args.length_max - 1,
                         prefix="Computing: ", fill='#')
        stats[i] = compute(i + 1)

    # update file if using save
    if args.save:
        np.savetxt(args.csv_file, stats, delimiter=',')

plt.scatter(lengths, stats[:, 1])
plt.xlabel('Stream Length')
plt.ylabel('Delta (std dev)')
plt.title('Deterministic Stochastic Stream')
plt.grid(True)
# plt.savefig("stoch-avg.png")
plt.show()

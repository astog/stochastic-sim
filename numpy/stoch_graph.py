from progressbar import printProgressBar
import matplotlib.pyplot as plt
import stoch_util as su
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Numpy Stochastic Grapher')
parser.add_argument('--samples', type=int, default=1000, metavar='N',
                    help='number of samples to take (default: 1000)')
parser.add_argument('--length-max', type=int, default=1024, metavar='N',
                    help='maximum length (default: 1024)')
parser.add_argument('--dotp-length', type=int, default=1024, metavar='N',
                    help='size of vector during dot productor (default: 1024)')
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
dotp_lengths = np.arange(1, args.dotp_length + 1, 1)
file_loaded = False
stats = None


# if using save, the load data if file exists, else compute
if args.save and os.path.isfile(args.csv_file):
    print("LOADING FILE: " + args.csv_file)
    stats = np.loadtxt(args.csv_file, delimiter=',')
    file_loaded = True


if not file_loaded:
    stats = np.zeros((args.length_max, 2))
    for i in xrange(args.length_max):
        vals = np.zeros(args.samples)
        dotp_length = args.dotp_length
        length = i + 1
        for j in xrange(args.samples):
            printProgressBar((i * args.samples) + j, args.length_max * args.samples - 1, prefix='Sampling:', suffix='', decimals=10, length=100, fill='#')

            xv = np.random.rand(dotp_length)
            yv = np.random.rand(dotp_length)
            if args.bipolar:
                xv = 2 * xv - 1
                yv = 2 * yv - 1

            xyd = np.dot(xv, yv)
            if args.bipolar:
                xyd = np.clip(xyd, -1, 1)
            else:
                xyd = np.clip(xyd, 0, 1)

            xm = su.stochify_vector(xv, length, args.bipolar, args.deterministic)
            ym = su.stochify_vector(yv, length, args.bipolar, args.deterministic)
            ap_xyd = su.toRealN(su.stoch_dot(xm, ym, args.bipolar, args.deterministic, 1), args.deterministic)
            vals[j] = xyd - ap_xyd

        stats[i] = (np.average(vals), np.std(vals))

    # update file if using save
    if args.save:
        np.savetxt(args.csv_file, stats, delimiter=',')

plt.scatter(lengths, stats[:, 0])
plt.xlabel('Stream Length')
plt.ylabel('Delta (avg)')
plt.grid(True)
# plt.savefig("stoch-avg.png")
plt.show()

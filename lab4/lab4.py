import numpy as np
import sys


def shuffle1(real_data, synth_data, p):
    indices = np.arange(len(real_data))
    np.random.shuffle(indices)
    mixed_data = np.where(indices < p * len(indices), real_data, synth_data)
    return mixed_data


def shuffle2(real_data, synth_data, p):
    mixed_data = np.where(np.random.choice([True, False], size=len(real_data), p=[1 - p, p]), real_data, synth_data)
    return mixed_data


def shuffle3(real_data, synth_data, p):
    mixed_data = np.where(np.random.rand(len(real_data)) < p, synth_data, real_data)
    return mixed_data


def shuffle4(real_data, synth_data, p):
    synth_indices = np.random.choice(np.arange(len(real_data)), size=int(len(real_data) * p), replace=False)
    mixed_data = np.copy(real_data)
    mixed_data[synth_indices] = synth_data[:len(synth_indices)]
    return mixed_data


def main():
    real_file = sys.argv[1]
    synth_file = sys.argv[2]
    p = float(sys.argv[3])
    real_data = np.loadtxt(real_file)
    synth_data = np.loadtxt(synth_file)
    print(shuffle1(real_data, synth_data, p))
    print(shuffle2(real_data, synth_data, p))
    print(shuffle3(real_data, synth_data, p))
    print(shuffle4(real_data, synth_data, p))


main()

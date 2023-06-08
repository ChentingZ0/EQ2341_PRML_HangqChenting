import numpy as np
import matplotlib.pyplot as plt
import os


data_path = '../data'

files = ['iphone_data_3.txt', 'iphone_test.txt']

data_file = [os.path.join(data_path, f) for f in files]


def read_file(txt_file):
    file = open(txt_file, 'r')
    # Read the file line by line
    lines = file.readlines()
    txt_list = []
    t = 0
    for line in lines:
        if t == 0:
            t += 1
        else:
            line = line[:-1].split(";")
            line = [float(f) for f in line]
            txt_list.append(line)
    file.close()

    return np.array(txt_list)[:, 1:]


data0 = read_file(data_file[0])
data1 = read_file(data_file[1])  # [data_length, 3]

np.save(os.path.join(data_path, "train_data.npy"), data0)
np.save(os.path.join(data_path, "test_data.npy"), data1)


def plot_data(data_array):
    x = range(data_array.shape[0])
    for i in range(data_array.shape[1]):
        plt.plot(x, data_array[:, i])
    plt.legend(["x", "y", "z"])
    plt.xlabel("t")
    plt.ylabel("accelerate")
    plt.title("Variation of the accelerate data")
    plt.show()


plot_data(data0)
plot_data(data1)

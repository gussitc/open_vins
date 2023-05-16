import numpy as np
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import os

lk_types = ["none", "translation", "euclidean", "affine"]#, "perspective"]
titles = ["fps", "lost_feats/frame", "track_length/lost_feat", "marg_tracks/frame"]

def average_rows(data, n):
    if n == 1:
        return data
    (rows, cols, page) = data.shape
    r = rows % n
    if r != 0:
        mat = data[:-r,:].reshape(rows//n, n, cols, page)
    else:
        mat = data.reshape(rows//n, n, cols, page)
    return np.mean(mat, axis=1).reshape(rows//n, cols, page)

def window_average_rows(data, n):
    if n == 1:
        return data
    elif n % 2 == 0:
        raise ValueError("n must be odd")
    out_list = []
    r = (n-1)//2
    for i in range(r, data.shape[0] - r):
        out = np.mean(data[(i-r):(i+r),:,:], axis=0)
        out_list.append(out)
    data_avg = np.stack(out_list, axis=0)
    return data_avg

def main():
    root_folder = os.path.dirname(os.path.realpath(__file__))
    data_list = []
    for type in lk_types:
        data = np.genfromtxt(root_folder + "/results_" + type + ".txt")
        data_list.append(data)

    # data = np.stack(data_list, axis=2)
    data = window_average_rows(np.stack(data_list, axis=2), 21)

    for i in range(1, data.shape[1]):
        plt.figure()
        plt.title(titles[i-1])
        plt.plot(data[:,0,0], data[:,i,:])
        print(titles[i-1])
        for j in range(data.shape[2]):
            print(f'{lk_types[j]:<12}: {np.mean(data[:,i,j]):.2f}')
        print()
        plt.legend(lk_types)
        plt.grid()
        plt.xlabel("Time [s]")
    plt.show()

if __name__ == "__main__":
    main()

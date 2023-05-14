import numpy as np
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import os

names = ["nogyro", "cv", "euclid"]
titles = ["fps", "lost_feats/frame", "track_length/lost_feat", "marg_tracks/frame"]

def average_rows(data, n):
    if n == 1:
        return data
    (rows, cols, page) = data.shape
    mat = data[:-(rows%n),:].reshape(rows//n, n, cols, page)
    return np.mean(mat, axis=1).reshape(rows//n, cols, page)

def main():
    root_folder = os.path.dirname(os.path.realpath(__file__))
    data_list = []
    for name in names:
        data = np.genfromtxt(root_folder + "/results_" + name + ".txt")
        data_list.append(data)

    data = average_rows(np.stack(data_list, axis=2), 10)

    for i in range(1, data.shape[1]):
        plt.figure()
        plt.title(titles[i-1])
        plt.plot(data[:,0,0], data[:,i,:])
        print(titles[i-1])
        for j in range(data.shape[2]):
            print(f'{names[j]:<8}: {np.mean(data[:,i,j]):.2f}')
        print()
        plt.legend(names)
        plt.grid()
        plt.xlabel("Time [s]")
    plt.show()

if __name__ == "__main__":
    main()

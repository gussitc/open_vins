import numpy as np
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import os

titles = ["fps", "lost_feats/frame", "track_length/lost_feat", "marg_tracks/frame"]
def main():
    root_folder = os.path.dirname(os.path.realpath(__file__))
    data = np.genfromtxt(root_folder + "/results0.txt")
    for i in range(1, data.shape[1]):
        plt.figure()
        plt.title(titles[i-1])
        plt.plot(data[:,0], data[:,i])
    plt.show()

if __name__ == "__main__":
    main()

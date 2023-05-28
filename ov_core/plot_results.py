import numpy as np
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import os


root_folder = os.path.dirname(os.path.realpath(__file__))
figure_folder = '/home/gustav/repos/masters-thesis/Images/openvins'
dataset_titles = ["Drone racing", "Rotation mix"]
smoothing_windows = [31, 21]

lk_types = ["none", "translation", "perspective", "adaptive"]
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

def get_yticks(data):
    data_max = np.max(np.max(data, axis=0))
    data_min = np.min(np.min(data, axis=0))
    precision = 0.1
    diff = data_max - data_min
    if (diff > 2):
        precision = 1
    data_max = np.ceil(data_max/precision)
    data_min = np.floor(data_min/precision)

    ticks = np.arange(data_min, data_max+1)*precision
    while (len(ticks) > 10):
        if (ticks[0] % 2 == 1 and ticks[0] * ticks[-1] < 0):
            ticks = ticks[1::2]
        else:
            ticks = ticks[0::2]
    if (len(ticks) <= 5):
        ticks = np.arange(data_min, data_max+1, 0.5)*precision

    return ticks

def plot_dataset(dataset, smoothing_window):
    filename = dataset.replace(' ', '_').lower()

    data_list = []
    for type in lk_types:
        data = np.genfromtxt(root_folder + f"/results/{filename}/results_" + type + ".txt")
        data_list.append(data)

    # data = np.stack(data_list, axis=2)
    data = window_average_rows(np.stack(data_list, axis=2), smoothing_window)

    plt.rcParams.update({'font.size': 16})
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(4,1,sharex=True, figsize=(16,10))
    fig.suptitle(f"OpenVINS performance for \"{dataset}\" dataset")
    time = data[:,0,0]
    xticks = np.arange(0, np.ceil(time[-1]), 2)
    for i in range(1, data.shape[1]):
        idx = i-1
        ax[idx].plot(data[:,0,0], data[:,i,:])
        print(titles[idx])
        for j in range(data.shape[2]):
            print(f'{lk_types[j]:<12}: {np.mean(data[:,i,j]):.2f}')
        print()
        ax[idx].grid()
        ax[idx].set_ylabel(titles[idx])
        ax[idx].set_xticks(xticks)
        ax[idx].set_yticks(get_yticks(data[:,i,:]))
    ax[-1].set_xlabel("Time [s]")
    ax[0].legend(lk_types, loc='upper left')
    fig.tight_layout()
    fig.savefig(figure_folder + f"/{filename}.pdf")

def main():
    for dataset, window in zip(dataset_titles, smoothing_windows):
        plot_dataset(dataset, window)
    plt.show()

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import os


root_folder = os.path.dirname(os.path.realpath(__file__))
ang_vel_folder = root_folder + "/results/ang_vel"
figure_folder = '/home/gustav/repos/masters-thesis/Images/openvins'
dataset_titles = ["Drone racing", "Rotation mix", "Foreground mix"]
smoothing_windows = [11, 11, 11]
# dataset_titles = ["Primary Z"]
# dataset_titles = ["Rotation mix"]
# dataset_titles = ["Drone racing"]
# dataset_titles = ["Foreground mix"]
# smoothing_windows = [11]

lk_types = ["no gyro", "translation", "perspective", "adaptive"]
# titles = ["fps", "lost_feats/frame", "track_length/lost_feat", "marg_tracks/frame"]
# titles = ["FPS", "Lost feats", "Length per lost", "Marg tracks"]
titles = ["FPS [Hz]", "Lost features", "Marginalized tracks"]

FPS = 30
start_time = 4
end_time = 11

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
    if (diff > 50):
        precision = 10
    elif (diff > 20):
        precision = 5
    elif (diff > 10):
        precision = 2
    elif (diff > 5):
        precision = 1

    data_max = np.ceil(data_max/precision)
    data_min = np.floor(data_min/precision)

    yticks = list(np.arange(data_min, data_max+1)*precision)
    while (len(yticks) >= 10):
        if ((yticks[0] / precision) % 2 == 1):
            yticks = [yticks[0] - precision] + yticks[1::2]
        elif ((yticks[-1] / precision) % 2 == 1):
            yticks = yticks[0::2] + [yticks[-1] + precision]
        else:
            yticks = yticks[0::2]

    if (len(yticks) < 5):
        yticks = np.arange(data_min, data_max+1, 0.5)*precision

    if (len(yticks) >= 10):
        print("Too many yticks")

    return yticks

def plot_dataset(dataset, smoothing_window):
    filename = dataset.replace(' ', '_').lower()
    width = 10
    hist_scale = 0.8
    perf_scale = 1.1

    data_list = []
    for type in lk_types:
        if ('no' in type):
            type = 'none'
        data = np.genfromtxt(root_folder + f"/results/{filename}/results_" + type + ".txt")
        data_list.append(data)

    data = np.stack(data_list, axis=2)
    data_avg = window_average_rows(data, smoothing_window)
    data_avg = np.delete(data_avg, 3, axis=1)
    ang_vel = np.genfromtxt(ang_vel_folder + f"/{filename}.txt")

    plt.rcParams.update({
        "text.usetex": True,
        'font.size': 16
    })
    fig, ax = plt.subplots(4,1,sharex=True, figsize=(width,width*perf_scale))
    fig.suptitle(f"OpenVINS performance for \"{dataset}\" dataset")
    time = data_avg[:,0,0]
    # xticks = np.arange(0, np.ceil(time[-1]), 2)
    ax[0].plot(ang_vel[:,0], ang_vel[:,1:])
    ax[0].grid()
    ax[0].set_ylabel("Delta angles [rad]")
    ax[0].legend(['x', 'y', 'z'], loc='upper left')
    ax[0].set_yticks(get_yticks(ang_vel[:,1:]))
    means = []
    for i in range(1, data_avg.shape[1]):
        means.append([])
        idx = i-1
        ax_idx = idx+1
        ax[ax_idx].plot(data_avg[:,0,0], data_avg[:,i,:])
        print(titles[idx])
        for j in range(data_avg.shape[2]):
            mean = np.mean(data_avg[:,i,j])
            std = np.std(data_avg[:,i,j])
            print(f'{lk_types[j]:<12}: {mean:.2f} +- {std:.2f}')
            means[-1].append(mean)
        print()
        ax[ax_idx].grid()
        ax[ax_idx].set_ylabel(titles[idx])
        # ax[idx].set_xticks(xticks)
        ax[ax_idx].set_yticks(get_yticks(data_avg[:,i,:]))
    ax[-1].set_xlabel("Time [s]")
    ax[1].legend(lk_types, loc='upper left')
    fig.tight_layout()
    fig.savefig(figure_folder + f"/{filename}.pdf")

    ##### PRINT TABLE #####
    for i, type in enumerate(lk_types):
        print(f'{type:<12} & ', end='')
        for j in range(len(means)):
            print(f'${means[j][i]:6.2f}$', end=' & '*(j < len(means)-1))
        print('\\\\')
    print()

    ##### HISTOGRAM #####
    fig, ax = plt.subplots(3,1,sharex=True, figsize=(width,width*hist_scale))
    fig.suptitle(f"Marginalized tracks for \"{dataset}\" dataset")
    bin_size = 10
    bins = np.arange(0, 80, bin_size)
    if ('drone' in filename):
        yticks = np.arange(0, 250, 40)
    else:
        yticks = np.arange(0, 160, 20)
    #     bin_size = 5
    #     bins = np.arange(0, 55, bin_size)

    def plot_hist(ax, dt, color, ylabel):
        ax.hist(dt, bins=bins, rwidth=0.9, color=color, zorder=10)
        ax.grid()
        ax.set_xticks(bins)
        ax.set_yticks(yticks)
        ax.set_ylabel(ylabel)

    d = data[:,-1,0]
    none_hist, bin_edges = np.histogram(d, bins=bins)
    plot_hist(ax[0], d, 'C0', 'No gyro')

    d = data[:,-1,3]
    plot_hist(ax[1], d, 'C3', 'Adaptive')
    adap_hist, bin_edges = np.histogram(d, bins=bins)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax[2].bar(bin_centers, (adap_hist-none_hist)/1,align='center', width=0.9*bin_size, color='C4', zorder=10)
    ax[2].axhline(y=0, color='black', linewidth=1)
    ax[2].set_xticks(bins)
    ax[2].grid()
    ax[2].set_xlabel("Marginalized tracks per frame")
    ax[2].set_ylabel("Difference")
    # if ('drone' in dataset):
    ax[2].set_yticks(np.arange(-40, 30, 10))
    # else:
        # ax[2].set_yticks(np.arange(-30, 31, 10))
    fig.tight_layout()
    fig.savefig(figure_folder + f"/{filename}_hist.pdf")

def main():
    for dataset, window in zip(dataset_titles, smoothing_windows):
        print(dataset)
        plot_dataset(dataset, window)
    plt.show()

if __name__ == "__main__":
    main()

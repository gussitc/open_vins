import re
import numpy as np
import pathlib

import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import addcopyfighandler

axis = 'mix'

folder_name = f'/output_rot_{axis}/'
# folder_name = '/output/'
folder_path = pathlib.Path(__file__).parent.resolve().as_posix() + '/my-klt/' + folder_name
angvel_file = 'angVel.txt'
trackrate_file = 'trackRate.txt'
featsmarg_file = 'featsMarg.txt'
runtime_file = 'runtime.txt'
# angvel_file = 'angVel_V203_full.txt'

def get_float(pattern, string):
    match = re.search(pattern, string)
    return float(match.group(1))

def get_vector(pattern, string):
    match = re.search(pattern, string)
    return float(match.group(1)), float(match.group(2)), float(match.group(3))

def get_vectors(pattern, string):
    vectors = []
    matches = re.findall(pattern, string)
    for match in matches:
        vectors.append([float(match[0]), float(match[1])])
    return vectors

def get_vecs_from_file(file):
    timestamps = []
    vecs_list = []
    mean_list = []
    with open(folder_path + file) as f:
        for line in f.readlines():
            timestamps.append(get_float(r'(.*):', line))
            vecs = get_vectors(r'\[(.*?), (.*?)\]', line)
            vecs_list.append(vecs)
            mean_list.append(np.mean(np.abs(vecs), axis=0))
    return np.array(timestamps), vecs_list, np.array(mean_list)

def main():
    angvel_timestamps = []
    angvel_x = []
    angvel_y = []
    angvel_z = []
    angvel_mean = []

    trackrate_timestamps = []
    trackrate = []
    gyro_trackrate = []
    affine_trackrate = []
    featsmarg_timestamps = []
    featsmarg = []
    gyro_featsmarg = []
    affine_featsmarg = []
    featsmarg_timestamps_avg = []
    featsmarg_avg = []
    gyro_featsmarg_avg = []
    affine_featsmarg_avg = []
    runtime_timestamps = []
    runtime = []
    gyro_runtime = []
    affine_runtime = []

    # timestamps, gyro_flows, gyro_mean  = get_vecs_from_file('gyroFlow.txt')
    # _, match_flows, match_mean = get_vecs_from_file('matchFlow.txt')
    # _, error_flows, error_mean = get_vecs_from_file('errorFlow.txt')

    # L=1
    # param_range = np.array(range(2, 40, 2))
    # num_passes = []
    # num_keypoints = []
    # for w in param_range:
    #     max_flow = w * 2**L
    #     pass_count = 0
    #     points = 0
    #     for frame in match_flows:
    #         for (dx, dy) in frame:
    #             if dx <= max_flow and dy <= max_flow:
    #                 pass_count += 1
    #             points += 1
    #     num_passes.append(pass_count)
    #     num_keypoints.append(points)

    # fig, ax = plt.subplots()
    # ax.plot(2*param_range + 1, np.array(num_passes)/np.array(num_keypoints), label='num_passes')
    # ax.grid()
    # ax.legend()


    with open(folder_path + angvel_file) as f:
        for line in f.readlines():
            angvel_timestamps.append(get_float(r'(.*):', line))
            x,y,z = get_vector(r'\[(.*), (.*), (.*)\]', line)
            angvel_mean.append((x+y+z)/3)
            angvel_x.append(x)
            angvel_y.append(y)
            angvel_z.append(z)
    
    with open(folder_path + trackrate_file) as f:
        for line in f.readlines():
            trackrate_timestamps.append(get_float(r'(.*):', line))
            trackrate.append(get_float(r': (.*)', line))

    with open(folder_path + "affine_" + trackrate_file) as f:
        for line in f.readlines():
            affine_trackrate.append(get_float(r': (.*)', line))

    with open(folder_path + "gyro_" + trackrate_file) as f:
        for line in f.readlines():
            gyro_trackrate.append(get_float(r': (.*)', line))

    with open(folder_path + featsmarg_file) as f:
        for line in f.readlines():
            featsmarg_timestamps.append(get_float(r'(.*):', line))
            featsmarg.append(get_float(r': (.*)', line))

    with open(folder_path + "affine_" + featsmarg_file) as f:
        for line in f.readlines():
            affine_featsmarg.append(get_float(r': (.*)', line))

    with open(folder_path + "gyro_" + featsmarg_file) as f:
        for line in f.readlines():
            gyro_featsmarg.append(get_float(r': (.*)', line))

    with open(folder_path + runtime_file) as f:
        for line in f.readlines():
            runtime_timestamps.append(get_float(r'(.*):', line))
            runtime.append(1/get_float(r': (.*)', line))

    with open(folder_path + "affine_" + runtime_file) as f:
        for line in f.readlines():
            affine_runtime.append(1/get_float(r': (.*)', line))

    with open(folder_path + "gyro_" + runtime_file) as f:
        for line in f.readlines():
            gyro_runtime.append(1/get_float(r': (.*)', line))

    for i in range(len(featsmarg_timestamps)//10):
        if (i+1)*10 > len(featsmarg_timestamps):
            break
        featsmarg_timestamps_avg.append(np.mean(featsmarg_timestamps[10*i:(i+1)*10]))
        featsmarg_avg.append(np.mean(featsmarg[10*i:(i+1)*10]))
        affine_featsmarg_avg.append(np.mean(affine_featsmarg[10*i:(i+1)*10]))
        gyro_featsmarg_avg.append(np.mean(gyro_featsmarg[10*i:(i+1)*10]))

    timestamps_lists = [angvel_timestamps, trackrate_timestamps, featsmarg_timestamps_avg, runtime_timestamps]
    min_timestamp = min([min(l) for l in timestamps_lists])
    max_timestamp = max([max(l) for l in timestamps_lists])
    # subtract min timestamp from all timestamps
    angvel_timestamps = [t - min_timestamp for t in angvel_timestamps]
    trackrate_timestamps = [t - min_timestamp for t in trackrate_timestamps]
    featsmarg_timestamps_avg = [t - min_timestamp for t in featsmarg_timestamps_avg]
    runtime_timestamps = [t - min_timestamp for t in runtime_timestamps]

    # pad the marginal feature lists with the first and last values
    featsmarg_avg = [featsmarg_avg[0]] + featsmarg_avg + [featsmarg_avg[-1]]
    affine_featsmarg_avg = [affine_featsmarg_avg[0]] + affine_featsmarg_avg + [affine_featsmarg_avg[-1]]
    gyro_featsmarg_avg = [gyro_featsmarg_avg[0]] + gyro_featsmarg_avg + [gyro_featsmarg_avg[-1]]
    # pad the timestamps with min and max timestamps
    featsmarg_timestamps_avg = [0] + featsmarg_timestamps_avg + [max_timestamp - min_timestamp]

    # print('Dataset length: ', timestamps[-1] - timestamps[0])
    # print('Mean error flow: ', np.mean(error_flows))

    # Create the plot and add lines for x, y, and z
    title = f'Increasing angular velocity around {axis}-axis'
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(4, 1, sharex=True)
    fig.set_figwidth(14)
    fig.set_figheight(10)
    ax[0].plot(angvel_timestamps, angvel_x, label='x')
    ax[0].plot(angvel_timestamps, angvel_y, label='y')
    ax[0].plot(angvel_timestamps, angvel_z, label='z')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_ylabel('Angular velocity [rad/s]')
    ax[0].set_title(title)

    ax[1].plot(trackrate_timestamps, trackrate, label='no gyro')
    ax[1].plot(trackrate_timestamps, gyro_trackrate, label='gyro')
    ax[1].plot(trackrate_timestamps, affine_trackrate, label='affine')
    # add averages as dashed horizontal lines
    ax[1].axhline(y=np.mean(trackrate), color='C0', linestyle='--')
    ax[1].axhline(y=np.mean(gyro_trackrate), color='C1', linestyle='--')
    ax[1].axhline(y=np.mean(affine_trackrate), color='C2', linestyle='--')


    ax[1].set_ylabel('Track rate [%]')
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(featsmarg_timestamps_avg, featsmarg_avg, label='no gyro')
    ax[2].plot(featsmarg_timestamps_avg, gyro_featsmarg_avg, label='gyro')
    ax[2].plot(featsmarg_timestamps_avg, affine_featsmarg_avg, label='affine')
    # add averages as dashed horizontal lines as above
    ax[2].axhline(y=np.mean(featsmarg_avg), color='C0', linestyle='--')
    ax[2].axhline(y=np.mean(gyro_featsmarg_avg), color='C1', linestyle='--')
    ax[2].axhline(y=np.mean(affine_featsmarg_avg), color='C2', linestyle='--')
    ax[2].set_ylabel('Marginal features')
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(runtime_timestamps, runtime, label='no gyro')
    ax[3].plot(runtime_timestamps, gyro_runtime, label='gyro')
    ax[3].plot(runtime_timestamps, affine_runtime, label='affine')
    # add averages as dashed horizontal lines as above
    ax[3].axhline(y=np.mean(runtime), color='C0', linestyle='--')
    ax[3].axhline(y=np.mean(gyro_runtime), color='C1', linestyle='--')
    ax[3].axhline(y=np.mean(affine_runtime), color='C2', linestyle='--')
    ax[3].set_xlabel('Time [s]')
    ax[3].set_ylabel('FPS [Hz]')
    ax[3].legend()
    ax[3].grid()

    fig.savefig(f"{folder_path}/rot_{axis}.png",bbox_inches='tight')

    # fig, ax = plt.subplots()
    # ax.plot(timestamps, gyro_flows[:,0], label='gyro flow')
    # ax.plot(timestamps, match_flows[:,0], label='match flow')
    # ax.plot(timestamps, error_flows[:,0], label='error flow')
    # # def plot_line(ax, x):
    # gyro_percentile = np.percentile(gyro_flows, 90)
    # # ax.axhline(y = gyro_percentile, color = 'C0', linestyle='dashed', label = f'90% = {gyro_percentile:.2f}')

    # match_percentile = np.percentile(match_flows, 90)
    # # ax.axhline(y = match_percentile, color = 'C1', linestyle='dashed', label = f'90% = {match_percentile:.2f}')
    # ax.legend()


    # fig, ax = plt.subplots()
    # ax.plot(timestamps, gyro_mean[:,1], label='gyro flow')
    # ax.plot(timestamps, match_mean[:,1], label='match flow')
    # ax.plot(timestamps, error_mean[:,1], label='error flow')
    # # def plot_line(ax, x):
    # gyro_percentile = np.percentile(gyro_mean[:,1], 90)
    # ax.axhline(y = gyro_percentile, color = 'C0', linestyle='dashed', label = f'90% = {gyro_percentile:.2f}')

    # match_percentile = np.percentile(match_mean[:,1], 90)
    # ax.axhline(y = match_percentile, color = 'C1', linestyle='dashed', label = f'90% = {match_percentile:.2f}')
    # ax.legend()
    # ax.grid()

    # ax.set_xlabel('Time')
    # ax.set_ylabel('Pixels')
    # ax.set_title('Average absolute flows in vertical direction')

    # fig, ax = plt.subplots()
    # plt.hist(error_flows, density=True,
    #             bins='auto',
    #             # bins=25,
    #             # histtype = 'step',
    #             color='#0504aa',
    #             rwidth=0.85,
    #             alpha=0.7)

    # Display the plot
    plt.show()
    
if __name__ == '__main__':
    main()
import re
import numpy as np
import pathlib

import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import addcopyfighandler

folder_path = pathlib.Path(__file__).parent.resolve().as_posix() + '/output/'
angvel_file = 'angVel.txt'
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

    timestamps, gyro_flows, gyro_mean  = get_vecs_from_file('gyroFlow.txt')
    _, match_flows, match_mean = get_vecs_from_file('matchFlow.txt')
    _, error_flows, error_mean = get_vecs_from_file('errorFlow.txt')

    L=1
    param_range = np.array(range(2, 40, 2))
    num_passes = []
    num_keypoints = []
    for w in param_range:
        max_flow = w * 2**L
        pass_count = 0
        points = 0
        for frame in match_flows:
            for (dx, dy) in frame:
                if dx <= max_flow and dy <= max_flow:
                    pass_count += 1
                points += 1
        num_passes.append(pass_count)
        num_keypoints.append(points)

    fig, ax = plt.subplots()
    ax.plot(2*param_range + 1, np.array(num_passes)/np.array(num_keypoints), label='num_passes')
    ax.grid()
    ax.legend()


    with open(folder_path + angvel_file) as f:
        for line in f.readlines():
            angvel_timestamps.append(get_float(r'(.*):', line))
            x,y,z = get_vector(r'\[(.*), (.*), (.*)\]', line)
            angvel_mean.append((x+y+z)/3)
            angvel_x.append(x)
            angvel_y.append(y)
            angvel_z.append(z)

    # print('Dataset length: ', timestamps[-1] - timestamps[0])
    # print('Mean error flow: ', np.mean(error_flows))

    # Create the plot and add lines for x, y, and z
    fig, ax = plt.subplots()
    ax.plot(angvel_timestamps, angvel_x, label='x')
    ax.plot(angvel_timestamps, angvel_y, label='y')
    ax.plot(angvel_timestamps, angvel_z, label='z')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('AngVel')
    ax.set_title('Angular velocities as measured by gyroscope')

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


    fig, ax = plt.subplots()
    ax.plot(timestamps, gyro_mean[:,1], label='gyro flow')
    ax.plot(timestamps, match_mean[:,1], label='match flow')
    ax.plot(timestamps, error_mean[:,1], label='error flow')
    # def plot_line(ax, x):
    gyro_percentile = np.percentile(gyro_mean[:,1], 90)
    ax.axhline(y = gyro_percentile, color = 'C0', linestyle='dashed', label = f'90% = {gyro_percentile:.2f}')

    match_percentile = np.percentile(match_mean[:,1], 90)
    ax.axhline(y = match_percentile, color = 'C1', linestyle='dashed', label = f'90% = {match_percentile:.2f}')
    ax.legend()
    ax.grid()

    ax.set_xlabel('Time')
    ax.set_ylabel('Pixels')
    ax.set_title('Average absolute flows in vertical direction')

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
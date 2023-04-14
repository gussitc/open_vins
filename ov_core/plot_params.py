import re
import numpy as np
import pathlib

import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
from scan_params import pyr_levels, results_file_path
# from scan_params import gyro_results_file, no_gyro_results_file

no_gyro_results_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/no_gyro_results.txt'
gyro_results_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/gyro_results.txt'

# no_gyro_results_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/pyr3_no_gyro_results.txt'
# gyro_results_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/pyr3_gyro_results.txt'

def get_data_from_file(file):
    gyro_results = np.genfromtxt(file)
    gyro_win_size_arr = 2*gyro_results[:, 0]+1
    gyro_fps_arr = gyro_results[:, 1]
    gyro_track_rate_arr = gyro_results[:, 2]
    marg_tracks = gyro_results[:, 3]
    return [gyro_win_size_arr, gyro_fps_arr, gyro_track_rate_arr, marg_tracks]

def plot_line(ax, x):
    ax.axvline(x = x, color = 'r', label = f'x = {x}')

def main():
    for pyr in pyr_levels:
        gyro_data = get_data_from_file(results_file_path + f'pyr{pyr}_gyro_results.txt')
        affine_data = get_data_from_file(results_file_path + f'pyr{pyr}_affine_results.txt')
        no_gyro_data = get_data_from_file(results_file_path + f'pyr{pyr}_no_gyro_results.txt')
        names = ['win_size', 'fps', 'track_rate', 'marg_tracks']
        for gyro, no_gyro, affine, name in zip(gyro_data[1:], no_gyro_data[1:], affine_data[1:], names[1:]):
            fig, ax = plt.subplots()
            ax.plot(gyro_data[0], gyro, label='gyro')
            ax.plot(no_gyro_data[0], no_gyro, label='no gyro')
            ax.plot(affine_data[0], affine, label='affine')
            # plot_line(ax, x)
            ax.set_xlabel('Patch size')
            ax.set_ylabel(name)
            ax.set_title(f'Pyr level: {pyr}')
            ax.grid()
            ax.legend()
 

    # ax.set_title('Average track rate (good tracks / num keypoints)')

    # ax.set_xlabel('Time')
    # ax.set_ylabel('AngVel')
    # ax.set_title('Angular velocities as measured by gyroscope')

    # fig, ax = plt.subplots()
    # ax.plot(timestamps, gyro_flows, label='gyro flow')
    # ax.plot(timestamps, match_flows, label='match flow')
    # ax.plot(timestamps, error_flows, label='error flow')
    # ax.legend()

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
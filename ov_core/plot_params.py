import re
import numpy as np
import pathlib

import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
from scan_params import gyro_results_file, no_gyro_results_file


def main():
    gyro_results = np.genfromtxt(gyro_results_file)
    gyro_win_size_arr = 2*gyro_results[:, 0]+1
    gyro_fps_arr = gyro_results[:, 1]
    gyro_track_rate_arr = gyro_results[:, 2]

    no_gyro_results = np.genfromtxt(no_gyro_results_file)
    no_gyro_win_size_arr = no_gyro_results[:, 0]
    no_gyro_fps_arr = no_gyro_results[:, 1]
    no_gyro_track_rate_arr = no_gyro_results[:, 2]

    # Create the plot and add lines for x, y, and z
    fig, ax = plt.subplots()
    ax.plot(gyro_win_size_arr, gyro_fps_arr, label='gyro')
    ax.plot(no_gyro_win_size_arr, no_gyro_fps_arr, label='no gyro')
    # ax.axvline(x = 40, color = 'r', label = 'x = 40')
    ax.set_xlabel('Window size')
    ax.set_ylabel('FPS')
    ax.set_title('Average FPS')
    ax.grid()
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(gyro_win_size_arr, gyro_track_rate_arr, label='gyro')
    ax.plot(no_gyro_win_size_arr, no_gyro_track_rate_arr, label='no gyro')
    # ax.axvline(x = 40, color = 'r', label = 'x = 40')
    ax.set_xlabel('Window size')
    ax.set_ylabel('Track rate')
    ax.set_title('Average track rate (good tracks / num keypoints)')
    ax.grid()
    ax.legend()

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
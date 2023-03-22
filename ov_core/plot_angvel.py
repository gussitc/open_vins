import re
import numpy as np
import pathlib
import matplotlib.pyplot as plt

folder_path = pathlib.Path(__file__).parent.resolve().as_posix() + '/output/'
angvel_file = 'angVel.txt'

def get_float(pattern, string):
    match = re.search(pattern, string)
    return float(match.group(1))

def get_vector(pattern, string):
    match = re.search(pattern, string)
    return float(match.group(1)), float(match.group(2)), float(match.group(3))

def get_vector_lengths(pattern, string):
    vector_len = []
    matches = re.findall(pattern, string)
    for match in matches:
        vector_len.append(np.sqrt(float(match[0])**2 + float(match[1])**2))
    return vector_len

def get_mean_vecs_from_file(file):
    timestamps = []
    mean_vecs = []
    with open(folder_path + file) as f:
        for line in f.readlines():
            timestamps.append(get_float(r'(.*):', line))
            vec_len = get_vector_lengths(r'\[(.*?), (.*?)\]', line)
            mean_vecs.append(np.mean(vec_len))
    return timestamps, mean_vecs

def main():
    timestamps = []
    angvel_x = []
    angvel_y = []
    angvel_z = []
    angvel_mean = []

    timestamps, gyro_flows = get_mean_vecs_from_file('gyroFlow.txt')
    timestamps, match_flows = get_mean_vecs_from_file('matchFlow.txt')
    timestamps, error_flows = get_mean_vecs_from_file('errorFlow.txt')

    with open(folder_path + angvel_file) as f:
        for line in f.readlines():
            # angvel_timestamps.append(get_float(r'(.*):', line))
            x,y,z = get_vector(r'\[(.*), (.*), (.*)\]', line)
            angvel_mean.append((x+y+z)/3)
            angvel_x.append(x)
            angvel_y.append(y)
            angvel_z.append(z)

    print('Dataset length: ', timestamps[-1] - timestamps[0])
    # print('Mean gyro flow: ', np.mean(gyro_flows))

    # Create the plot and add lines for x, y, and z
    fig, ax = plt.subplots()
    ax.plot(timestamps, angvel_x, label='x')
    ax.plot(timestamps, angvel_y, label='y')
    ax.plot(timestamps, angvel_z, label='z')
    # ax.plot(timestamps, angvel_mean, label='mean')

    ax.legend()

    ax.set_xlabel('Time')
    ax.set_ylabel('AngVel')
    ax.set_title('Angular velocities as measured by gyroscope')

    fig, ax = plt.subplots()
    ax.plot(timestamps, gyro_flows, label='gyro flow')
    ax.plot(timestamps, match_flows, label='match flow')
    ax.plot(timestamps, error_flows, label='error flow')
    ax.legend()

    fig, ax = plt.subplots()
    plt.hist(match_flows, density=True,
                bins='auto',
            #  bins=17,
            #  histtype = 'step',
                color='#0504aa',
                rwidth=0.85,
                alpha=0.7)

    # Display the plot
    plt.show()
    
if __name__ == '__main__':
    main()
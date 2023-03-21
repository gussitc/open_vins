import re
import numpy as np
import pathlib
import matplotlib.pyplot as plt

folder_path = pathlib.Path(__file__).parent.resolve().as_posix() + '/output/'
file_name = 'angVel.txt'

def get_float(pattern, string):
    match = re.search(pattern, string)
    return float(match.group(1))

def get_vector(pattern, string):
    match = re.search(pattern, string)
    group = match.group
    return float(match.group(1)), float(match.group(2)), float(match.group(3))

def main():
    timestamps = []
    angvel_x = []
    angvel_y = []
    angvel_z = []

    with open(folder_path + file_name) as f:
        for line in f.readlines():
            timestamps.append(get_float(r'(.*):', line))
            x,y,z = get_vector(r'\[(.*), (.*), (.*)\]', line)
            angvel_x.append(x)
            angvel_y.append(y)
            angvel_z.append(z)

    print('Dataset length: ', timestamps[-1] - timestamps[0])

    # Create the plot and add lines for x, y, and z
    fig, ax = plt.subplots()
    ax.plot(timestamps, angvel_x, label='x')
    ax.plot(timestamps, angvel_y, label='y')
    ax.plot(timestamps, angvel_z, label='z')

    ax.legend()

    ax.set_xlabel('Time')
    ax.set_ylabel('AngVel')
    ax.set_title('Angular velocities as measured by gyroscope')

    # Display the plot
    plt.show()
    
if __name__ == '__main__':
    main()
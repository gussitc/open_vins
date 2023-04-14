import re
import subprocess
import numpy as np
import os

config_file = '/home/gustav/catkin_ws_ov/src/open_vins/config/euroc_mav/estimator_config.yaml'
# config_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/test_config.yaml'
program_file = '/home/gustav/catkin_ws_ov/src/open_vins/build/Release/devel/lib/ov_core/test_tracking'
no_gyro_results_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/no_gyro_results.txt'
gyro_results_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/gyro_results.txt'
results_file_path = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/data/'
# results_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/results.txt'

# USE_GYRO = True
USE_GYRO = False

def change_yaml_params(file, param_value_tuples):
    for param, val in param_value_tuples:
        with open(file, 'r') as f:
            content = f.read()
            ret = re.sub(param + r': .*', f'{param}: {val}', content)
            # print(ret)
        if ret != '':
            with open(file, 'w') as f:
                f.write(ret)


# pyr_levels = range(0, 2, 1)
pyr_levels = [0, 2, 3]
half_patch_sizes = range(2, 20, 1)
param_arr = np.array(half_patch_sizes)
param_name = 'half_patch_size'

def perform_scan(results_file):
    for pyr in pyr_levels:
        print(f'pyr_levels: {pyr}')
        change_yaml_params(config_file, [('pyr_levels', pyr)])
        for param_val in param_arr:
            print(f'{param_name}: {param_val}')
            change_yaml_params(config_file, [(param_name, param_val)])

            output = subprocess.check_output([program_file])
            # print(output)
            lines = output.decode('ascii').split('\n')
            marg_tracks = float(lines[-4])
            track_rate = float(lines[-3])
            fps = float(lines[-2])
            print(marg_tracks)
            print(track_rate)
            print(fps)
            with open(f'{results_file_path}pyr{pyr}_{results_file}', 'a') as f:
                f.write(f'{param_val} {fps} {track_rate} {marg_tracks}\n')

def main():
    # os.mkdir(results_file_path)

    change_yaml_params(config_file, [('use_gyro_aided_tracker', 'false')])
    perform_scan('no_gyro_results.txt')

    # results_file = gyro_results_file
    change_yaml_params(config_file, [('use_gyro_aided_tracker', 'true'), ('predict_transform', 'false')])
    perform_scan('gyro_results.txt')

    # results_file = no_gyro_results_file
    change_yaml_params(config_file, [('use_gyro_aided_tracker', 'true'), ('predict_transform', 'true')])
    perform_scan('affine_results.txt')

        


if __name__ == '__main__':
    main()
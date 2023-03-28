import re
import subprocess
import numpy as np

config_file = '/home/gustav/catkin_ws_ov/src/open_vins/config/euroc_mav/estimator_config.yaml'
# config_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/test_config.yaml'
program_file = '/home/gustav/catkin_ws_ov/src/open_vins/build/Release/devel/lib/ov_core/test_tracking'
no_gyro_results_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/no_gyro_results.txt'
gyro_results_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/gyro_results.txt'
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

def main():
    param_range = range(2, 40, 2)
    param_arr = np.array(param_range)
    if USE_GYRO:
        param_name = 'half_patch_size'
        results_file = gyro_results_file
        change_yaml_params(config_file, [('use_gyro_aided_tracker', 'true')])
    else:
        param_arr = 2 * param_arr + 1
        param_name = 'win_size'
        results_file = no_gyro_results_file
        change_yaml_params(config_file, [('use_gyro_aided_tracker', 'false')])

    for param_val in param_arr:
        print(f'{param_name}: {param_val}')
        change_yaml_params(config_file, [(param_name, param_val)])

        output = subprocess.check_output([program_file])
        print(output)
        lines = output.decode('ascii').split('\n')
        track_rate = float(lines[-3])
        fps = float(lines[-2])
        print(track_rate)
        print(fps)
        with open(results_file, 'a') as f:
            f.write(f'{param_val} {fps} {track_rate}\n')
        


if __name__ == '__main__':
    main()
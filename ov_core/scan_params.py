import re
import subprocess
import numpy as np
import os

config_file = '/home/gustav/catkin_ws_ov/src/open_vins/config/euroc_mav/estimator_config.yaml'
# config_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/test_config.yaml'
program_file = '/home/gustav/catkin_ws_ov/src/open_vins/build/Release/devel/lib/ov_core/test_tracking'
# cmd = ["perf", "stat", "-e", "cpu-clock", program_file]
cmd = [program_file]
no_gyro_results_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/no_gyro_results.txt'
gyro_results_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/gyro_results.txt'
results_file_path = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/data/'
runtime_file_path = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/my-klt/runtime.txt'
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
pyr_levels = [0, 1, 2, 3]
half_patch_sizes = range(2, 21, 1)
param_arr = np.array(half_patch_sizes)
param_name = 'half_patch_size'

num_repetitions = 5

def perform_scan(results_file):
    for pyr in pyr_levels:
        print(f'pyr_levels: {pyr}')
        change_yaml_params(config_file, [('pyr_levels', pyr)])
        for param_val in param_arr:
            print(f'{param_name}: {param_val}')
            change_yaml_params(config_file, [(param_name, param_val)])

            # output = subprocess.check_output([program_file])
            os.remove(runtime_file_path)
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # print(output)
            # user_time_search = re.search(r'(\d+),(\d+) seconds user', result.stderr)
            # sys_time_search = re.search(r'(\d+),(\d+) seconds sys', result.stderr)
            # user_time = float(user_time_search[1]) + float('0.' + user_time_search[2])
            # sys_time = float(sys_time_search[1]) + float('0.' + sys_time_search[2])
            # total_time = user_time + sys_time

            runtimes = np.loadtxt(runtime_file_path)

            lines = result.stdout.split('\n')
            # lines_err = result.stderr.split('\n')
            marg_tracks = float(lines[-4])
            track_rate = float(lines[-3])
            frames = int(lines[-2])
            assert frames == len(runtimes)
            fps = runtimes.size/np.sum(runtimes)
            print(marg_tracks)
            print(track_rate)
            print(fps)
            with open(f'{results_file_path}pyr{pyr}_{results_file}', 'a') as f:
                f.write(f'{param_val} {fps} {track_rate} {marg_tracks}\n')

def main():
    os.mkdir(results_file_path)

    for i in range(num_repetitions):
        change_yaml_params(config_file, [('lk_method', '1')])
        perform_scan(f'no_gyro_results{i}.txt')

        # results_file = gyro_results_file
        change_yaml_params(config_file, [('lk_method', '2')])
        perform_scan(f'gyro_results{i}.txt')

    # results_file = no_gyro_results_file
    # change_yaml_params(config_file, [('lk_method', '3')])
    # perform_scan('affine_results.txt')

        


if __name__ == '__main__':
    main()
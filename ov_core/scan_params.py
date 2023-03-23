import re
import subprocess

config_file = '/home/gustav/catkin_ws_ov/src/open_vins/config/euroc_mav/estimator_config.yaml'
# config_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/test_config.yaml'
program_file = '/home/gustav/catkin_ws_ov/src/open_vins/build/Release/devel/lib/ov_core/test_tracking'
no_gyro_results_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/no_gyro_results.txt'
gyro_results_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/gyro_results.txt'
results_file = '/home/gustav/catkin_ws_ov/src/open_vins/ov_core/results.txt'

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
    # for win_size in range(105, 155, 5):
    for half_patch_size in range(2, 55, 2):
        # print(f'win_size: {win_size}')
        print(f'half_patch_size: {half_patch_size}')
        change_yaml_params(config_file, [('half_patch_size', half_patch_size)])

        output = subprocess.check_output([program_file])
        print(output)
        lines = output.decode('ascii').split('\n')
        track_rate = float(lines[-3])
        fps = float(lines[-2])
        print(track_rate)
        print(fps)
        with open(results_file, 'a') as f:
            f.write(f'{half_patch_size} {fps} {track_rate}\n')
        


if __name__ == '__main__':
    main()
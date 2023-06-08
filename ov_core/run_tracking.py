import os
import subprocess
import numpy as np
from ruamel.yaml import YAML
from plot_results import main as plot_results
from plot_results import lk_types

root_folder = os.path.dirname(os.path.realpath(__file__))
program_file = root_folder + "/../build/Release/devel/lib/ov_core/test_tracking"
config_file = root_folder + "/my-klt/config.yaml"
# results_folder = root_folder + "/results/primary_z"
# results_folder = root_folder + "/results/drone_racing"
# results_folder = root_folder + "/results/rotation_mix"
results_folder = root_folder + "/results/foreground_mix"

yaml = YAML()
yaml.preserve_quotes = True

def edit_config(parameter, value):
    with open(config_file, 'r') as file:
        data = yaml.load(file)
    data[parameter] = value
    with open(config_file, 'w') as file:
        file.write("%YAML:1.0\n---\n")
        yaml.dump(data, file)

def main():
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    for type in lk_types:
        print("Running " + type + "...")
        edit_config("lk_type", type.upper())
        subprocess.run(program_file)
        os.rename(root_folder + "/results.txt", results_folder + "/results_" + type + ".txt")
        print('\n')

if __name__ == "__main__":
    main()
    # plot_results()
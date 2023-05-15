import os
import subprocess
import numpy as np
from ruamel.yaml import YAML
from plot_results import main as plot_results
from plot_results import lk_types

root_folder = os.path.dirname(os.path.realpath(__file__))
program_file = root_folder + "/../build/Release/devel/lib/ov_core/test_tracking"
config_file = root_folder + "/my-klt/config.yaml"

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
    for type in lk_types:
        edit_config("lk_type", type.upper())
        subprocess.run(program_file)
        os.rename(root_folder + "/results.txt", root_folder + "/results_" + type + ".txt")

if __name__ == "__main__":
    main()
    plot_results()
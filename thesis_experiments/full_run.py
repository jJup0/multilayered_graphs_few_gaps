import os
import subprocess

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

thesis_experiments_dirname = "thesis_experiments"

generate_cmd_args = [
    "python",
    "-m",
    f"{thesis_experiments_dirname}.generate_2_layer_test_instances",
    "20",
    "15",
    "10",
    "0.2",
    "thesis_experiments/local_tests/in",
]
subprocess.run(generate_cmd_args, cwd=cwd, shell=True)

minimize_cmd_args = [
    "python",
    "-m",
    f"{thesis_experiments_dirname}.minimize_crossings",
    "--sidegaps",
    "--in_dir",
    f"{thesis_experiments_dirname}/local_tests/in",
    "median",
    f"{thesis_experiments_dirname}/local_tests/out/out2.csv",
]
subprocess.run(minimize_cmd_args, cwd=cwd, shell=True)

minimize_cmd_args[-2] = "ilp"
subprocess.run(minimize_cmd_args, cwd=cwd, shell=True)

print("done")

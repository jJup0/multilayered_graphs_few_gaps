import os
import subprocess

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

thesis_experiments_dirname = "thesis_experiments"

def run_regular():
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

def create_batch_jobs(in_directory_path: str, out_csv_path: str):
    files = os.listdir(in_directory_path)
    standard run cmds
    for alg_name in ["median", "barycenter", "ilp"]:
        for file in files:
            subprocess.run(["qsub", ""])
        

print("done")

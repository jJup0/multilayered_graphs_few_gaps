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

    standard_run_cmds = [
        "qsub",
        # "-N",
        # "JobName",
        # memory
        "-l",
        "s_vmem=1.8G",
        "-l",
        "h_vmem=1.9G",
        "-l",
        "mem_free=1.9G",
        # restart on fail
        "-r",
        "y",
        # output
        "-e",
        "$TEMPDIR/stderr.txt",
        "-o",
        "$TEMPDIR/stdout.txt",
        "python",
        "-m",
        f"{thesis_experiments_dirname}.minimize_crossings",
        "--sidegaps",
        "--in_file",
        f"{in_directory_path}",
        "<<algorithm_name>>",
        f"out_csv_path",
    ]
    for alg_name in ["median", "barycenter", "ilp"]:
        for filename in files:
            standard_run_cmds[-2] = alg_name
            standard_run_cmds[-4] = filename


if __name__ == "__main__":
    create_batch_jobs(
        os.path.realpath("./performance_tests/in"),
        os.path.realpath("./performance_tests/out/out2.csv"),
    )

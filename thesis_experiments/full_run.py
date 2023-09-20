import csv
import os
import subprocess

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

thesis_experiments_dirname = "thesis_experiments"


def run_regular(test_case_name: str):
    in_dir_name = f"{thesis_experiments_dirname}/local_tests/{test_case_name}/in"
    graph_gen_count = 20
    for real_node_count in (10, 20, 30, 40, 50):
        virtual_node_count = real_node_count // 2
        generate_cmd_args = [
            "python",
            "-m",
            f"{thesis_experiments_dirname}.generate_2_layer_test_instances",
            f"{graph_gen_count}",
            f"{real_node_count}",
            f"{virtual_node_count}",
            "0.2",
            in_dir_name,
        ]
        print(f"generating {graph_gen_count} graphs with {real_node_count=}")
        subprocess.run(generate_cmd_args, cwd=cwd, shell=True)

    out_csv_file = f"{thesis_experiments_dirname}/local_tests/{test_case_name}/out.csv"
    os.makedirs(os.path.dirname(out_csv_file), exist_ok=True)

    with open(out_csv_file, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            (
                "alg_name",
                "real_nodes_per_layer_count",
                "virtual_nodes_l2_count",
                "real_edge_density",
                "instance_name",
                "crossings",
                "time_s",
            )
        )

    minimize_cmd_args = [
        "python",
        "-m",
        f"{thesis_experiments_dirname}.minimize_crossings",
        "--sidegaps",
        "--in_dir",
        in_dir_name,
        "median",
        out_csv_file,
    ]
    subprocess.run(minimize_cmd_args, cwd=cwd, shell=True)

    minimize_cmd_args[-2] = "barycenter"
    subprocess.run(minimize_cmd_args, cwd=cwd, shell=True)


def create_batch_jobs(in_directory_path: str, out_csv_path: str):
    files = os.listdir(in_directory_path)

    standard_run_cmds = [
        "qsub",
        "-N",
        "crossing_minimization_gaps",
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
        standard_run_cmds[-2] = alg_name
        for filename in files:
            standard_run_cmds[-4] = filename
            subprocess.run(standard_run_cmds)


if __name__ == "__main__":
    # create_batch_jobs(
    #     os.path.realpath("./performance_tests/in"),
    #     os.path.realpath("./performance_tests/out/out2.csv"),
    # )
    run_regular("testcase2")

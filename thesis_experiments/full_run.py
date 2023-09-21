import csv
import os
import subprocess

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

thesis_experiments_dirname = "thesis_experiments"


def in_dir_name(test_case_name: str):
    return f"{thesis_experiments_dirname}/local_tests/{test_case_name}/in"


# TODO parameterize this
def create_graphs(test_case_name: str):
    # graph_gen_count = 20
    graph_gen_count = 20
    # for real_node_count in (10, 20, 30, 40, 50):
    for real_node_count in (50,):
        # for real_node_count in (10, 15, 20):
        virtual_node_count = real_node_count // 2
        generate_cmd_args = [
            "python",
            "-m",
            f"{thesis_experiments_dirname}.generate_2_layer_test_instances",
            f"{graph_gen_count}",
            f"{real_node_count}",
            f"{virtual_node_count}",
            "0.2",
            in_dir_name(test_case_name),
        ]
        print(f"generating {graph_gen_count} graphs with {real_node_count=}")
        subprocess.run(generate_cmd_args, cwd=cwd, shell=True)


def create_csv_out(test_case_name: str) -> str:
    out_csv_file = f"{thesis_experiments_dirname}/local_tests/{test_case_name}/out.csv"
    os.makedirs(os.path.dirname(out_csv_file), exist_ok=True)

    with open(out_csv_file, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            (
                "alg_name",
                "gap_type",
                "gaps",
                "real_nodes_per_layer_count",
                "virtual_nodes_l2_count",
                "real_edge_density",
                "instance_name",
                "crossings",
                "time_s",
            )
        )
    return out_csv_file


def run_regular_side_gaps(test_case_name: str):
    create_graphs(test_case_name)
    out_csv_file = create_csv_out(test_case_name)

    minimize_cmd_args = [
        "python",
        "-m",
        f"{thesis_experiments_dirname}.minimize_crossings",
        "--sidegaps",
        "--in_dir",
        in_dir_name(test_case_name),
        "<<alg_name>>",
        out_csv_file,
    ]
    for alg_name in ["median", "barycenter", "ilp"]:
        minimize_cmd_args[-2] = alg_name
        subprocess.run(minimize_cmd_args, cwd=cwd, shell=True)


def run_regular_k_gaps(test_case_name: str):
    create_graphs(test_case_name)
    out_csv_file = create_csv_out(test_case_name)

    for k in (1, 2, 3, 100):
        # for k in (2, 3, 100):
        minimize_cmd_args = [
            "python",
            "-m",
            f"{thesis_experiments_dirname}.minimize_crossings",
            "--kgaps",
            f"{k}",
            "--in_dir",
            in_dir_name(test_case_name),
            "<<alg_name>>",
            out_csv_file,
        ]
        for alg_name in ["median", "barycenter", "ilp"]:
            minimize_cmd_args[-2] = alg_name
            # print(f"{minimize_cmd_args=}")
            subprocess.run(minimize_cmd_args, cwd=cwd, shell=True)


def run_batch(test_case_name: str):
    create_graphs(test_case_name)
    out_csv_file = create_csv_out(test_case_name)
    files = os.listdir(in_dir_name(test_case_name))

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
        f"<<file_name>>",
        "<<algorithm_name>>",
        f"{out_csv_file}",
    ]
    for alg_name in ["median", "barycenter", "ilp"]:
        standard_run_cmds[-2] = alg_name
        for filename in files:
            standard_run_cmds[-3] = filename
            subprocess.run(standard_run_cmds)


if __name__ == "__main__":
    # create_batch_jobs(
    #     os.path.realpath("./performance_tests/in"),
    #     os.path.realpath("./performance_tests/out/out2.csv"),
    # )
    run_regular_k_gaps("testcase_50_kgaps")
    # run_regular_k_gaps("testcase_temp")

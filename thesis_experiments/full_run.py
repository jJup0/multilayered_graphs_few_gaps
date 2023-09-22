import csv
import os
import subprocess
from typing import Iterable

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

thesis_experiments_dirname = "thesis_experiments"


def in_dir_name(test_case_name: str):
    # return os.path.realpath(f"local_tests/{test_case_name}/in")
    return os.path.realpath(
        f"{thesis_experiments_dirname}/local_tests/{test_case_name}/in"
    )


def out_csv_path(test_case_name: str):
    # return f"local_tests/{test_case_name}/out.csv"
    return os.path.realpath(
        f"{thesis_experiments_dirname}/local_tests/{test_case_name}/out.csv"
    )


def log_path(test_case_name: str):
    # return f"local_tests/{test_case_name}/out.csv"
    return os.path.realpath(
        f"{thesis_experiments_dirname}/local_tests/{test_case_name}/log.txt"
    )


# TODO parameterize this
def create_graphs(
    test_case_name: str,
    *,
    graph_gen_count: int,
    real_node_counts: Iterable[int],
    virtual_node_counts: Iterable[int],
    real_edge_density: float,
):
    print(
        f"about to generate {graph_gen_count * len(real_node_counts)} graph instances"
    )
    for real_node_count, vnode_count in zip(
        real_node_counts, virtual_node_counts, strict=True
    ):
        generate_cmd_args = [
            "python",
            "-m",
            f"{thesis_experiments_dirname}.generate_2_layer_test_instances",
            f"{graph_gen_count}",
            f"{real_node_count}",
            f"{vnode_count}",
            f"{real_edge_density}",
            in_dir_name(test_case_name),
        ]
        # print(f"generating {graph_gen_count} graphs with {real_node_count=}")
        subprocess.run(generate_cmd_args, cwd=cwd)


def create_csv_out(test_case_name: str) -> str:
    out_csv_file = out_csv_path(test_case_name)
    os.makedirs(os.path.dirname(out_csv_file), exist_ok=True)

    with open(out_csv_file, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            (
                "alg_name",
                "gap_type",
                "gap_count",
                "real_nodes_per_layer_count",
                "virtual_nodes_per_layer_count",
                "real_edge_density",
                "instance_name",
                "crossings",
                "time_s",
            )
        )
    return out_csv_file


def run_regular_side_gaps(test_case_name: str):
    create_graphs(
        test_case_name,
        graph_gen_count=20,
        real_node_counts=[10, 20, 30, 40, 50],
        virtual_node_counts=[5, 10, 15, 20, 25],
        real_edge_density=0.1,
    )
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
        subprocess.Popen(minimize_cmd_args, cwd=cwd)


def run_regular_k_gaps(test_case_name: str):
    create_graphs(
        test_case_name,
        graph_gen_count=20,
        real_node_counts=[10, 20, 30, 40, 50],
        virtual_node_counts=[5, 10, 15, 20, 25],
        real_edge_density=0.1,
    )
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
            subprocess.Popen(minimize_cmd_args, cwd=cwd)


def run_sidegaps_batch(
    test_case_name: str,
    *,
    graph_gen_count: int,
    real_node_counts: Iterable[int],
    virtual_node_counts: Iterable[int],
    real_edge_density: float,
):
    create_graphs(
        test_case_name,
        graph_gen_count=graph_gen_count,
        real_node_counts=real_node_counts,
        virtual_node_counts=virtual_node_counts,
        real_edge_density=real_edge_density,
    )
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
        log_path(test_case_name),
        "-o",
        log_path(test_case_name),
        "minimize_crossings_wrapper.sh",
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
            subprocess.Popen(standard_run_cmds)


if __name__ == "__main__":
    # run_regular_k_gaps("testcase_50_kgaps2")
    # create_csv_out("testcase_50_kgaps2")
    # run_regular_k_gaps("testcase_temp")
    run_sidegaps_batch(
        "testcase_10-50_sidegaps_batched",
        graph_gen_count=5,
        real_node_counts=[50],
        virtual_node_counts=[25],
        real_edge_density=0.1,
    )


# tests to do
# r = 50, v = 30, p=0.1, kgaps, k

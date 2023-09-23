import csv
import logging
import os
import subprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

thesis_experiments_dirname = "thesis_experiments"

GL_OPEN_PROCESSES: list[tuple[list[str], subprocess.Popen[bytes]]] = []


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

def get_qsub_args(alggap_type_as_flag: str, gap_count: int | None = None):
    return standard_run_cmds = [
                "qsub",
                "-N",
                "crossing_minimization_gaps",
                # memory
                "-l",
                "s_vmem={}G",
                "-l",
                "h_vmem={}G",
                "-l",
                "mem_free={}G",
                # restart on fail
                "-r",
                "y",
                # output
                "-e",
                log_path(test_case_name),
                "-o",
                log_path(test_case_name),
                minimize_crossings_wrapper_path,
                **gap_type_and_args,
                "--in_file",
                f"{filepath}",
                f"{alg_name}",
                f"{out_csv_file}",
            ]


minimize_crossings_wrapper_path = os.path.realpath(
    f"{thesis_experiments_dirname}/minimize_crossings_wrapper.sh"
)


def create_graphs(
    test_case_name: str,
    *,
    graph_gen_count: int,
    real_node_counts: list[int],
    virtual_node_counts: list[int],
    real_edge_density: float,
):
    logger.info(
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
        logger.info(f"generating {graph_gen_count} graphs with {real_node_count=}")
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
        GL_OPEN_PROCESSES.append(
            (minimize_cmd_args, subprocess.Popen(minimize_cmd_args, cwd=cwd))
        )


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
            GL_OPEN_PROCESSES.append(
                (minimize_cmd_args, subprocess.Popen(minimize_cmd_args, cwd=cwd))
            )


def run_sidegaps_batch(
    test_case_name: str,
    *,
    graph_gen_count: int,
    real_node_counts: list[int],
    virtual_node_counts: list[int],
    real_edge_density: float,
    run_k_gaps: bool,
    gap_counts: list[int] = []
):
    # create_graphs(
    #     test_case_name,
    #     graph_gen_count=graph_gen_count,
    #     real_node_counts=real_node_counts,
    #     virtual_node_counts=virtual_node_counts,
    #     real_edge_density=real_edge_density,
    # )
    # out_csv_file = create_csv_out(test_case_name)
    out_csv_file = out_csv_path(test_case_name)
    files = os.listdir(in_dir_name(test_case_name))

    with open(log_path(test_case_name), "w"):
        pass


    # for alg_name in ["median", "barycenter", "ilp"]:
    # for alg_name in ["ilp"]:
    if not run_k_gaps:
        gap_counts=[-1]
    
    for alg_name in ["median"]:
        for filename in files:
            filepath = os.path.realpath(
                os.path.join(in_dir_name(test_case_name), filename)
            )
            filesize = os.path.getsize(filepath)
            if alg_name == "ilp":
                mem_required = 0.5 + 0.0001 * filesize
            else:
                mem_required = 0.1 + 0.00003 * filesize
            print(f"{filename=} {alg_name=} {mem_required=}")

            get_standard_qsub_args()

            # GL_OPEN_PROCESSES.append(
            #     (standard_run_cmds, subprocess.Popen(standard_run_cmds))
            # )


if __name__ == "__main__":
    real_node_counts = list(range(10, 11, 10))
    virtual_node_counts = [c // 2 for c in real_node_counts]
    run_sidegaps_batch(
        "testcase1",
        graph_gen_count=5,
        real_node_counts=real_node_counts,
        virtual_node_counts=virtual_node_counts,
        real_edge_density=0.01,
    )

    timeout_s = 1
    while GL_OPEN_PROCESSES:
        _path, process = GL_OPEN_PROCESSES[-1]
        try:
            exit_code = process.wait(timeout=timeout_s)
            logger.info(f"process finished with {exit_code=}")
            GL_OPEN_PROCESSES.pop()
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Subprocess did not complete within the {timeout_s}s timeout."
            )


# tests to do
# r = 50, v = 30, p=0.1, kgaps, k

import csv
import logging
import os
import subprocess
import sys

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


def create_log_file(test_case_name: str):
    with open(log_path(test_case_name), "w"):
        pass


minimize_crossings_wrapper_path = os.path.realpath(
    f"{thesis_experiments_dirname}/minimize_crossings_wrapper.sh"
)


def create_graphs(
    test_case_name: str,
    *,
    graph_gen_count: int,
    real_node_counts: list[int],
    virtual_node_ratios: list[float],
    average_node_degrees: list[float],
):
    logger.info(
        f"about to generate {graph_gen_count * len(real_node_counts)} graph instances"
    )

    create_graph_proccesses: list[subprocess.Popen[bytes]] = []

    for real_node_count, vnode_percent, average_node_degree in zip(
        real_node_counts, virtual_node_ratios, average_node_degrees, strict=True
    ):
        generate_cmd_args = [
            "python",
            "-m",
            f"{thesis_experiments_dirname}.generate_2_layer_test_instances",
            f"{graph_gen_count}",
            f"{real_node_count}",
            f"{vnode_percent}",
            f"{average_node_degree}",
            in_dir_name(test_case_name),
        ]
        logger.info(f"generating {graph_gen_count} graphs with {real_node_count=}")

        create_graph_proccesses.append(subprocess.Popen(generate_cmd_args, cwd=cwd))

    for process in create_graph_proccesses:
        process.wait()


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
        virtual_node_ratios=[5, 10, 15, 20, 25],
        average_node_degrees=[0.1],
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
        graph_gen_count=5,
        real_node_counts=[10],
        virtual_node_ratios=[0.1],
        average_node_degrees=[4],
    )
    out_csv_file = create_csv_out(test_case_name)

    for k in (2, 3):
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


def get_qsub_args(
    test_case_name: str,
    file_name: str,
    alg_name: str,
    gap_type_as_flag: str,
    gap_count: int | None = None,
) -> list[str]:
    # handle k- or side-gaps
    if gap_type_as_flag == "--kgaps":
        gap_type_and_args = ["--kgaps", f"{gap_count}"]
    elif gap_type_as_flag == "--sidegaps":
        gap_type_and_args = ["--sidegaps"]
    else:
        assert False, f"gap_type_as_flag must be --kgaps or --sidegaps"

    # determine memory limit
    filepath = os.path.realpath(os.path.join(in_dir_name(test_case_name), file_name))
    filesize = os.path.getsize(filepath)
    # if alg_name == "ilp":
    #     mem_required = 0.5 + 0.0001 * filesize
    # else:
    #     mem_required = 0.1 + 0.00003 * filesize
    mem_required = 2
    # print(f"{file_name=} {alg_name=} {mem_required=}")

    return [
        "qsub",
        "-N",
        "crossing_minimization_gaps",
        # memory
        "-l",
        f"s_vmem={(mem_required - 0.05):.3f}G",
        "-l",
        f"h_vmem={mem_required:.3f}G",
        "-l",
        f"mem_free={mem_required:.3f}G",
        # restart on fail
        "-r",
        "y",
        # output
        "-e",
        log_path(test_case_name),
        "-o",
        log_path(test_case_name),
        minimize_crossings_wrapper_path,
        "--in_file",
        f"{filepath}",
        *gap_type_and_args,
        alg_name,
        f"{out_csv_path(test_case_name)}",
    ]


def run_sidegaps_batch(
    test_case_name: str,
    *,
    graph_gen_count: int,
    real_node_counts: list[int],
    virtual_node_ratios: list[float],
    average_node_degrees: list[float],
    run_k_gaps: bool,
    gap_counts: list[int] = [],
):
    create_graphs(
        test_case_name,
        graph_gen_count=graph_gen_count,
        real_node_counts=real_node_counts,
        virtual_node_ratios=virtual_node_ratios,
        average_node_degrees=average_node_degrees,
    )
    create_csv_out(test_case_name)

    if not run_k_gaps:
        gap_counts = [-1]

    create_log_file(test_case_name)

    files = os.listdir(in_dir_name(test_case_name))
    # for alg_name in ["median", "barycenter", "ilp"]:
    for alg_name in ["median"]:
        for file_name in files:
            for gap_count in gap_counts:
                standard_run_cmds = get_qsub_args(
                    test_case_name, file_name, alg_name, "--kgaps", gap_count
                )
                GL_OPEN_PROCESSES.append(
                    (standard_run_cmds, subprocess.Popen(standard_run_cmds))
                )


def wait_for_processes_to_finish():
    timeout_s = 1
    while GL_OPEN_PROCESSES:
        _args, process = GL_OPEN_PROCESSES[-1]
        try:
            exit_code = process.wait(timeout=timeout_s)
            logger.info(f"process finished with {exit_code=}")
            GL_OPEN_PROCESSES.pop()
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Subprocess did not complete within the {timeout_s}s timeout."
            )


class ClusterExperiments:
    """Not a real class, just a container for all experiments that should be run for the thesis."""

    STANDARD_GRAPH_GEN_COUNT = 1

    @classmethod
    def vary_gap_count(cls, test_case_suffix: str = ""):
        real_node_counts = [5]
        virtual_node_ratios = [0.1]
        average_node_degrees = [2.0]
        run_k_gaps = True
        # gap_counts = [1, 2, 3, 4, 5, 10, 15]
        gap_counts = [10]
        run_sidegaps_batch(
            f"testcase_k_gaps_count_variation_{test_case_suffix}",
            graph_gen_count=cls.STANDARD_GRAPH_GEN_COUNT,
            real_node_counts=real_node_counts,
            virtual_node_ratios=virtual_node_ratios,
            average_node_degrees=average_node_degrees,
            run_k_gaps=run_k_gaps,
            gap_counts=gap_counts,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_case_suffix = sys.argv[1]
    else:
        test_case_suffix = ""
    ClusterExperiments.vary_gap_count(test_case_suffix)
    # run_regular_k_gaps(f"regular_{test_case_suffix}")
    wait_for_processes_to_finish()


# tests to do
# r = 50, v = 30, p=0.1, kgaps, k

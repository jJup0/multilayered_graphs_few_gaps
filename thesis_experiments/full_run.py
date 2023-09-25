import csv
import json
import logging
import os
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

thesis_experiments_dirname = "thesis_experiments"

GL_OPEN_PROCESSES: list[tuple[list[str], subprocess.Popen[bytes]]] = []


def test_case_base_dir(test_case_name: str):
    # return os.path.realpath(f"local_tests/{test_case_name}/in")
    return os.path.realpath(
        f"{thesis_experiments_dirname}/local_tests/{test_case_name}"
    )


def in_dir_name(test_case_name: str):
    return os.path.realpath(os.path.join(test_case_base_dir(test_case_name), f"in"))


def out_csv_path(test_case_name: str):
    return os.path.realpath(
        os.path.join(test_case_base_dir(test_case_name), f"out.csv")
    )


def log_path(test_case_name: str):
    return os.path.realpath(
        os.path.join(test_case_base_dir(test_case_name), f"log.txt")
    )


def test_case_info_path(test_case_name: str):
    return os.path.realpath(
        os.path.join(test_case_base_dir(test_case_name), f"info.json")
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
    nodes_per_layer: list[int],
    virtual_node_ratios: list[float],
    average_node_degrees: list[float],
):
    logger.info(
        f"about to generate {graph_gen_count * len(nodes_per_layer)} graph instances"
    )

    create_graph_proccesses: list[subprocess.Popen[bytes]] = []

    for real_node_count, vnode_percent, average_node_degree in zip(
        nodes_per_layer, virtual_node_ratios, average_node_degrees, strict=True
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
                "nodes_per_layer",
                "virtual_node_ratio",
                "average_node_degree",
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
        nodes_per_layer=[10, 20, 30, 40, 50],
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
        nodes_per_layer=[10],
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
    # filesize = os.path.getsize(filepath)

    if alg_name == "ilp":
        # mem_required = 1 + 0.0001 * filesize
        mem_required = 10
    else:
        # mem_required = 1 + 0.00003 * filesize
        mem_required = 1
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


def create_testcase_info_json(
    test_case_name: str,
    *,
    nodes_per_layer: list[int],
    virtual_node_ratios: list[float],
    average_node_degrees: list[float],
    run_k_gaps: bool,
    gap_counts: list[int],
    graph_title: str = "",
):
    data_constants: dict[str, Any] = {}
    graph_params_and_verbose_names_csv_titles = [
        (nodes_per_layer, "nodes per layer"),
        (virtual_node_ratios, "virtual node ratio"),
        (average_node_degrees, "average node degree"),
        (gap_counts, "gap count"),
    ]

    variables = graph_params_and_verbose_names_csv_titles.copy()
    for iterable_verbose_tuple in graph_params_and_verbose_names_csv_titles:
        param_iterable, verbose_name = iterable_verbose_tuple
        first_item = param_iterable[0]
        if all(first_item == item for item in param_iterable):
            data_constants[verbose_name] = param_iterable[0]
            variables.remove(iterable_verbose_tuple)

    assert len(variables) == 1, "Exactly one parameter should varied"

    variable_iterable, variable_verbose_name = variables[0]
    data_variable = [variable_verbose_name, variable_iterable]

    if graph_title:
        data_graph_title = graph_title
    else:
        oscm_type = "OSCM-KG" if run_k_gaps else "OSCM-SG"
        # TODO what else to include in title
        data_graph_title = f"{oscm_type} ..."

    graph_info: dict[str, Any] = {
        "constants": data_constants,
        "variable": data_variable,  # variable will be on y axis
        "graph_title": data_graph_title,
    }

    with open(test_case_info_path(test_case_name), "w") as f:
        json.dump(graph_info, f)


def run_batch(
    test_case_name: str,
    *,
    graph_gen_count: int,
    nodes_per_layer: list[int],
    virtual_node_ratios: list[float],
    average_node_degrees: list[float],
    run_k_gaps: bool,
    gap_counts: list[int] = [2],
    graph_title: str = "",
):
    create_graphs(
        test_case_name,
        graph_gen_count=graph_gen_count,
        nodes_per_layer=nodes_per_layer,
        virtual_node_ratios=virtual_node_ratios,
        average_node_degrees=average_node_degrees,
    )
    create_csv_out(test_case_name)

    create_testcase_info_json(
        test_case_name,
        nodes_per_layer=nodes_per_layer,
        virtual_node_ratios=virtual_node_ratios,
        average_node_degrees=average_node_degrees,
        run_k_gaps=run_k_gaps,
        gap_counts=gap_counts,
    )

    create_log_file(test_case_name)

    if not run_k_gaps:
        gap_counts = [2]
    files = os.listdir(in_dir_name(test_case_name))
    for alg_name in ["median", "barycenter", "ilp"]:
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

    # STANDARD_GRAPH_GEN_COUNT = 30
    STANDARD_GRAPH_GEN_COUNT = 5
    # STANDARD_NODE_COUNT = 50
    STANDARD_NODE_COUNT = 20
    STANDARD_VIRTUAL_NODE_RATIO = 0.1
    # STANDARD_AVERAGE_NODE_DEGREE = 5.0
    STANDARD_AVERAGE_NODE_DEGREE = 2.0

    @classmethod
    def vary_gap_count(cls, test_case_suffix: str = ""):
        test_case_name = f"testcase_k_gaps_count_variation{test_case_suffix}"
        nodes_per_layer = [cls.STANDARD_NODE_COUNT]
        virtual_node_ratios = [cls.STANDARD_VIRTUAL_NODE_RATIO]
        average_node_degrees = [cls.STANDARD_AVERAGE_NODE_DEGREE]
        run_k_gaps = True
        gap_counts = [1, 2, 3, 4, 5, 10, 15, nodes_per_layer[0]]
        run_batch(
            test_case_name,
            graph_gen_count=cls.STANDARD_GRAPH_GEN_COUNT,
            nodes_per_layer=nodes_per_layer,
            virtual_node_ratios=virtual_node_ratios,
            average_node_degrees=average_node_degrees,
            run_k_gaps=run_k_gaps,
            gap_counts=gap_counts,
        )

    @classmethod
    def side_gaps_vs_arbitrary_2_gaps(cls, test_case_suffix: str = ""):
        test_case_name = f"testcase_2_gaps_vs_side_gaps{test_case_suffix}"
        nodes_per_layer = list(range(10, 71, 10))
        virtual_node_ratios = [cls.STANDARD_VIRTUAL_NODE_RATIO] * len(nodes_per_layer)
        average_node_degrees = [cls.STANDARD_AVERAGE_NODE_DEGREE] * len(nodes_per_layer)
        run_k_gaps = True
        gap_counts = [2]
        # run k-gaps first
        run_batch(
            test_case_name,
            graph_gen_count=cls.STANDARD_GRAPH_GEN_COUNT,
            nodes_per_layer=nodes_per_layer,
            virtual_node_ratios=virtual_node_ratios,
            average_node_degrees=average_node_degrees,
            run_k_gaps=run_k_gaps,
            gap_counts=gap_counts,
            graph_title="OSCM side-gaps vs. 2 arbitrary gaps",
        )

        # manually run side gaps
        files = os.listdir(in_dir_name(test_case_name))
        for alg_name in ["median", "barycenter", "ilp"]:
            for file_name in files:
                for gap_count in gap_counts:
                    standard_run_cmds = get_qsub_args(
                        test_case_name, file_name, alg_name, "--sidegaps", gap_count
                    )
                    GL_OPEN_PROCESSES.append(
                        (standard_run_cmds, subprocess.Popen(standard_run_cmds))
                    )

    @classmethod
    def vary_virtual_node_ratio(cls, test_case_suffix: str = ""):
        test_case_name = f"testcase_side_gaps_virtual_node_variation{test_case_suffix}"
        virtual_node_ratios = list(ratio / 10 for ratio in range(10))
        nodes_per_layer = [cls.STANDARD_NODE_COUNT] * len(virtual_node_ratios)
        average_node_degrees = [cls.STANDARD_AVERAGE_NODE_DEGREE] * len(
            virtual_node_ratios
        )
        run_k_gaps = False
        run_batch(
            test_case_name,
            graph_gen_count=cls.STANDARD_GRAPH_GEN_COUNT,
            nodes_per_layer=nodes_per_layer,
            virtual_node_ratios=virtual_node_ratios,
            average_node_degrees=average_node_degrees,
            run_k_gaps=run_k_gaps,
        )

    @classmethod
    def vary_node_degree(cls, test_case_suffix: str = ""):
        test_case_name = f"testcase_side_gaps_vary_node_degree{test_case_suffix}"
        average_node_degrees = [2.0, 3.0, 4.0] + list(range(5, 41, 5))
        nodes_per_layer = [cls.STANDARD_NODE_COUNT] * len(average_node_degrees)
        virtual_node_ratios = [cls.STANDARD_VIRTUAL_NODE_RATIO] * len(
            average_node_degrees
        )
        run_k_gaps = False
        run_batch(
            test_case_name,
            graph_gen_count=cls.STANDARD_GRAPH_GEN_COUNT,
            nodes_per_layer=nodes_per_layer,
            virtual_node_ratios=virtual_node_ratios,
            average_node_degrees=average_node_degrees,
            run_k_gaps=run_k_gaps,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_case_suffix = sys.argv[1]
    else:
        test_case_suffix = ""

    ClusterExperiments.vary_gap_count(test_case_suffix)
    ClusterExperiments.vary_node_degree(test_case_suffix)
    ClusterExperiments.vary_virtual_node_ratio(test_case_suffix)
    ClusterExperiments.side_gaps_vs_arbitrary_2_gaps(test_case_suffix)

    wait_for_processes_to_finish()

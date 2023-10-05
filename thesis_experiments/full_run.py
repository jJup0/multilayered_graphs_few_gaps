import json
import logging
import math
import os
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.DEBUG)

STANDARD_CWD = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

thesis_experiments_dirname = "thesis_experiments"

GL_OPEN_PROCESSES: list[tuple[list[str], subprocess.Popen[str]]] = []


def popen_wrapper(arguments: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
    p = subprocess.Popen(arguments, **kwargs)
    GL_OPEN_PROCESSES.append((arguments, p))
    return p


def test_case_base_dir(test_case_name: str):
    # return os.path.realpath(f"local_tests/{test_case_name}/in")
    return os.path.realpath(
        f"{thesis_experiments_dirname}/local_tests/{test_case_name}"
    )


def in_dir_name(test_case_name: str):
    return os.path.realpath(os.path.join(test_case_base_dir(test_case_name), f"in"))


def get_out_csv_path(test_case_name: str):
    return os.path.realpath(
        os.path.join(test_case_base_dir(test_case_name), f"out.csv")
    )


def out_csv_dir(test_case_name: str):
    return os.path.realpath(os.path.join(test_case_base_dir(test_case_name), "out"))


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

        create_graph_proccesses.append(
            popen_wrapper(generate_cmd_args, cwd=STANDARD_CWD)
        )

    for process in create_graph_proccesses:
        logger.info("waiting on graph generation process")
        process.wait()
    logger.info("graph generation done")


def get_qsub_args(
    test_case_name: str,
    *,
    file_name: str,
    alg_name: str,
    gap_type_as_flag: str,
    gap_count: int | None = None,
    two_sided: bool = False,
    two_sided_iterations: int = 1,
) -> list[str]:
    # handle k- or side-gaps
    if gap_type_as_flag == "--kgaps":
        gap_type_and_args = ["--kgaps", f"{gap_count}"]
    elif gap_type_as_flag == "--sidegaps":
        gap_type_and_args = ["--sidegaps"]
    else:
        assert False, f"gap_type_as_flag must be --kgaps or --sidegaps"

    # handle two_sided
    if two_sided:
        two_sided_args = ["--two_sided", str(two_sided_iterations)]
    else:
        two_sided_args = []

    # determine memory limit
    filepath = os.path.realpath(os.path.join(in_dir_name(test_case_name), file_name))
    # filesize = os.path.getsize(filepath)

    if alg_name == "ilp":
        # mem_required = 1 + 0.0001 * filesize
        mem_required = 12
    else:
        # mem_required = 1 + 0.00003 * filesize
        mem_required = 1
    # print(f"{file_name=} {alg_name=} {mem_required=}")

    # out_csv_path = out_csv_path(test_case_name)
    csv_out_dir = out_csv_dir(test_case_name)
    os.makedirs(csv_out_dir, exist_ok=True)
    out_csv_path = os.path.join(
        csv_out_dir,
        f"{file_name}{alg_name}{gap_type_as_flag}{gap_count}{two_sided_iterations}.out",
    )

    return [
        "qsub",
        "-N",
        "crossing_minimization_gaps",
        # memory
        "-l",
        f"s_vmem={(mem_required - 0.2):.3f}G",
        "-l",
        f"h_vmem={mem_required:.3f}G",
        "-l",
        f"mem_free={mem_required:.3f}G",
        "-l",
        "bc4",
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
        *two_sided_args,
        alg_name,
        out_csv_path,
    ]


def create_testcase_info_json(
    test_case_name: str,
    *,
    graph_gen_count: int,
    nodes_per_layer: list[int],
    virtual_node_ratios: list[float],
    average_node_degrees: list[float],
    run_k_gaps: bool,
    gap_counts: list[int],
    two_sided_iterations: list[int],
    graph_title: str = "",
    only_heuristic: bool,
):
    data_constants: dict[str, Any] = {}
    graph_params_and_verbose_names_csv_titles = [
        (nodes_per_layer, "nodes per layer"),
        (virtual_node_ratios, "virtual node fraction"),
        (average_node_degrees, "average node degree"),
        (gap_counts, "gap count"),
        (two_sided_iterations, "up and down iterations"),
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
        data_graph_title = f"{oscm_type} ..."

    # 3 algorithms, each solve `graph_gen_count` graphs with `len(data_variable[1])` different parameters
    alg_count = 2 if only_heuristic else 3
    expected_results_count = alg_count * graph_gen_count * len(data_variable[1])

    graph_info: dict[str, Any] = {
        "constants": data_constants,
        "variable": data_variable,  # variable will be on y axis
        "graph_title": data_graph_title,
        "expected_results_count": expected_results_count,
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
    only_heuristic: bool = False,
    two_sided: bool = False,
    two_sided_iterations: list[int] = [1],
    graph_title: str = "",
):
    create_graphs(
        test_case_name,
        graph_gen_count=graph_gen_count,
        nodes_per_layer=nodes_per_layer,
        virtual_node_ratios=virtual_node_ratios,
        average_node_degrees=average_node_degrees,
    )

    create_testcase_info_json(
        test_case_name,
        graph_gen_count=graph_gen_count,
        nodes_per_layer=nodes_per_layer,
        virtual_node_ratios=virtual_node_ratios,
        average_node_degrees=average_node_degrees,
        run_k_gaps=run_k_gaps,
        gap_counts=gap_counts,
        graph_title=graph_title,
        two_sided_iterations=two_sided_iterations,
        only_heuristic=only_heuristic,
    )

    create_log_file(test_case_name)

    if not run_k_gaps:
        gap_counts = [2]
    dispatch_minimize(
        test_case_name,
        gap_counts=gap_counts,
        only_heuristic=only_heuristic,
        run_k_gaps=run_k_gaps,
        two_sided=two_sided,
        two_sided_iterations=two_sided_iterations,
    )


def dispatch_minimize(
    test_case_name: str,
    *,
    gap_counts: list[int],
    run_k_gaps: bool,
    only_heuristic: bool = False,
    two_sided: bool = False,
    two_sided_iterations: list[int] = [1],
):
    alg_names = ["median", "barycenter"]
    if not only_heuristic:
        alg_names.append("ilp")

    files = os.listdir(in_dir_name(test_case_name))

    if run_k_gaps:
        gap_type_as_flag = "--kgaps"
    else:
        gap_type_as_flag = "--sidegaps"

    for alg_name in alg_names:
        for file_name in files:
            for gap_count in gap_counts:
                for _two_sided_iterations in two_sided_iterations:
                    standard_run_cmds = get_qsub_args(
                        test_case_name,
                        file_name=file_name,
                        alg_name=alg_name,
                        gap_type_as_flag=gap_type_as_flag,
                        gap_count=gap_count,
                        two_sided=two_sided,
                        two_sided_iterations=_two_sided_iterations,
                    )
                    popen_wrapper(standard_run_cmds)


def wait_for_processes_to_finish():
    timeout_s = 1
    while GL_OPEN_PROCESSES:
        _args, process = GL_OPEN_PROCESSES[-1]
        try:
            exit_code = process.wait(timeout=timeout_s)
            if exit_code == 0:
                logger.debug(f"process finished with {exit_code=}")
            else:
                logger.warning(f"process finished with {exit_code=}, {_args=}")

            GL_OPEN_PROCESSES.pop()
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Subprocess did not complete within the {timeout_s}s timeout."
            )


class ClusterExperiments:
    """Not a real class, just a container for all experiments that should be run for the thesis."""

    STANDARD_GRAPH_GEN_COUNT = 20
    # STANDARD_GRAPH_GEN_COUNT = 1
    STANDARD_NODE_COUNT = 40
    STANDARD_VIRTUAL_NODE_RATIO = 0.2
    STANDARD_AVERAGE_NODE_DEGREE = 3.0

    @classmethod
    def _test_case_name(cls, base_name: str, test_case_version: str):
        # return f"testcase_{test_case_version}_{base_name}"
        return f"testcase_{test_case_version}_{base_name}"

    @classmethod
    def vary_gap_count(cls, test_case_suffix: str = ""):
        test_case_name = cls._test_case_name("k_gaps_count_variation", test_case_suffix)
        nodes_per_layer = [cls.STANDARD_NODE_COUNT]
        virtual_node_ratios = [cls.STANDARD_VIRTUAL_NODE_RATIO]
        average_node_degrees = [cls.STANDARD_AVERAGE_NODE_DEGREE]
        run_k_gaps = True
        gap_counts = [1, 2, 3, 4]
        max_virtual_nodes = math.ceil(nodes_per_layer[0] * virtual_node_ratios[0])
        # add gap counts in steps of 5
        gap_count_steps = 3
        next_gap_count = gap_counts[-1] - gap_count_steps + 1
        while next_gap_count < max_virtual_nodes:
            next_gap_count += gap_count_steps
            gap_counts.append(next_gap_count)

        run_batch(
            test_case_name,
            graph_gen_count=cls.STANDARD_GRAPH_GEN_COUNT,
            nodes_per_layer=nodes_per_layer,
            virtual_node_ratios=virtual_node_ratios,
            average_node_degrees=average_node_degrees,
            run_k_gaps=run_k_gaps,
            gap_counts=gap_counts,
        )
        logger.info("finished %s", test_case_name)
        return test_case_name

    @classmethod
    def side_gaps_vs_arbitrary_2_gaps(cls, test_case_suffix: str = ""):
        test_case_name = cls._test_case_name("2_gaps_vs_side_gaps", test_case_suffix)
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
        dispatch_minimize(
            test_case_name, run_k_gaps=False, gap_counts=[2], only_heuristic=False
        )

        # overwrite info file
        with open(test_case_info_path(test_case_name)) as f:
            info_json = json.load(f)
        info_json["expected_results_count"] *= 2
        with open(test_case_info_path(test_case_name), "w") as f:
            json.dump(info_json, f)

        logger.info("finished %s", test_case_name)
        return test_case_name

    @classmethod
    def side_gaps_vary_virtual_node_ratio(cls, test_case_suffix: str = ""):
        test_case_name = cls._test_case_name(
            "side_gaps_virtual_node_variation", test_case_suffix
        )
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
        logger.info("finished %s", test_case_name)
        return test_case_name

    @classmethod
    def side_gaps_vary_node_degree(cls, test_case_suffix: str = ""):
        test_case_name = cls._test_case_name(
            "side_gaps_vary_node_degree", test_case_suffix
        )
        average_node_degrees = [2.0, 3.0, 4.0] + list(
            range(5, cls.STANDARD_NODE_COUNT - 1, 5)
        )
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
        logger.info("finished %s", test_case_name)
        return test_case_name

    @classmethod
    def side_gaps_vary_node_count(cls, test_case_suffix: str = ""):
        test_case_name = cls._test_case_name(
            "side_gaps_vary_node_count", test_case_suffix
        )
        nodes_per_layer = list(range(10, 71, 10))
        average_node_degrees = [cls.STANDARD_AVERAGE_NODE_DEGREE] * len(nodes_per_layer)
        virtual_node_ratios = [cls.STANDARD_VIRTUAL_NODE_RATIO] * len(nodes_per_layer)
        run_k_gaps = False
        run_batch(
            test_case_name,
            graph_gen_count=cls.STANDARD_GRAPH_GEN_COUNT,
            nodes_per_layer=nodes_per_layer,
            virtual_node_ratios=virtual_node_ratios,
            average_node_degrees=average_node_degrees,
            run_k_gaps=run_k_gaps,
        )
        logger.info("finished %s", test_case_name)
        return test_case_name

    @classmethod
    def tscm_sg(cls, test_case_suffix: str = ""):
        test_case_name = cls._test_case_name("tscm_sg", test_case_suffix)
        nodes_per_layer = list(range(5, 41, 5))
        average_node_degrees = [cls.STANDARD_AVERAGE_NODE_DEGREE] * len(nodes_per_layer)
        virtual_node_ratios = [cls.STANDARD_VIRTUAL_NODE_RATIO] * len(nodes_per_layer)
        run_batch(
            test_case_name,
            graph_gen_count=cls.STANDARD_GRAPH_GEN_COUNT,
            nodes_per_layer=nodes_per_layer,
            virtual_node_ratios=virtual_node_ratios,
            average_node_degrees=average_node_degrees,
            run_k_gaps=False,
            two_sided=True,
        )
        logger.info("finished %s", test_case_name)
        return test_case_name

    @classmethod
    def tscm_sg_vary_up_and_down(cls, test_case_suffix: str = ""):
        test_case_name = cls._test_case_name(
            "tscm_sg_vary_up_and_down", test_case_suffix
        )
        # single up down is included in manual dispatch for only ilp
        up_and_down_iterations = [2, 3, 4, 5]
        nodes_per_layer = [cls.STANDARD_NODE_COUNT] * len(up_and_down_iterations)
        average_node_degrees = [cls.STANDARD_AVERAGE_NODE_DEGREE] * len(
            up_and_down_iterations
        )
        virtual_node_ratios = [cls.STANDARD_VIRTUAL_NODE_RATIO] * len(
            up_and_down_iterations
        )
        run_batch(
            test_case_name,
            graph_gen_count=cls.STANDARD_GRAPH_GEN_COUNT,
            nodes_per_layer=nodes_per_layer,
            virtual_node_ratios=virtual_node_ratios,
            average_node_degrees=average_node_degrees,
            run_k_gaps=False,
            two_sided=True,
            two_sided_iterations=up_and_down_iterations,
            only_heuristic=True,
        )

        # run ILP separately, as it does not matter how many iterations
        dispatch_minimize(
            test_case_name,
            gap_counts=[2],
            run_k_gaps=False,
            only_heuristic=False,
            two_sided=True,
            two_sided_iterations=[1],
        )

        logger.info("finished %s", test_case_name)
        return test_case_name

    LARGER_INSTANCE_MAX_NODES = 1_000
    LARGER_INSTANCE_AVG_NODE_DEGREE = 3.0

    @classmethod
    def oscm_k_gaps_large_instances(cls, test_case_suffix: str = ""):
        test_case_name = cls._test_case_name(
            "oscm_k_gaps_large_instances", test_case_suffix
        )
        # NOTE: HIGHER AVERAGE NODE DEGREE AND GAP COUNT
        nodes_per_layer = list(range(100, cls.LARGER_INSTANCE_MAX_NODES + 1, 100))
        average_node_degrees = [cls.LARGER_INSTANCE_AVG_NODE_DEGREE] * len(
            nodes_per_layer
        )
        virtual_node_ratios = [cls.STANDARD_VIRTUAL_NODE_RATIO] * len(nodes_per_layer)
        gap_counts = [5] * len(nodes_per_layer)
        run_batch(
            test_case_name,
            graph_gen_count=cls.STANDARD_GRAPH_GEN_COUNT,
            nodes_per_layer=nodes_per_layer,
            virtual_node_ratios=virtual_node_ratios,
            average_node_degrees=average_node_degrees,
            run_k_gaps=True,
            gap_counts=gap_counts,
            only_heuristic=True,
        )

        logger.info("finished %s", test_case_name)
        return test_case_name

    @classmethod
    def oscm_side_gaps_large_instances(cls, test_case_suffix: str = ""):
        test_case_name = cls._test_case_name(
            "oscm_side_gaps_large_instances", test_case_suffix
        )
        # NOTE: HIGHER AVERAGE NODE DEGREE AND GAP COUNT
        nodes_per_layer = list(range(100, cls.LARGER_INSTANCE_MAX_NODES + 1, 100))
        average_node_degrees = [cls.LARGER_INSTANCE_AVG_NODE_DEGREE] * len(
            nodes_per_layer
        )
        virtual_node_ratios = [cls.STANDARD_VIRTUAL_NODE_RATIO] * len(nodes_per_layer)
        run_batch(
            test_case_name,
            graph_gen_count=cls.STANDARD_GRAPH_GEN_COUNT,
            nodes_per_layer=nodes_per_layer,
            virtual_node_ratios=virtual_node_ratios,
            average_node_degrees=average_node_degrees,
            run_k_gaps=False,
            only_heuristic=True,
        )

        logger.info("finished %s", test_case_name)
        return test_case_name

    @classmethod
    def run_micro(cls, test_case_suffix: str = ""):
        # SHOULD NOT BE INCLUDED IN RUN
        test_case_name = cls._test_case_name("run_micro", test_case_suffix)
        graph_gen_count = 3
        nodes_per_layer = [10]
        average_node_degrees = [2.0] * len(nodes_per_layer)
        virtual_node_ratios = [0.2] * len(nodes_per_layer)
        run_k_gaps = True
        gap_counts = [1, 2]
        run_batch(
            test_case_name,
            graph_gen_count=graph_gen_count,
            nodes_per_layer=nodes_per_layer,
            virtual_node_ratios=virtual_node_ratios,
            average_node_degrees=average_node_degrees,
            run_k_gaps=run_k_gaps,
            gap_counts=gap_counts,
        )
        logger.info("finished %s", test_case_name)
        return test_case_name


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_case_suffix = sys.argv[1]
    else:
        test_case_suffix = "temp"

    ClusterExperiments.vary_gap_count(test_case_suffix)
    ClusterExperiments.side_gaps_vary_node_degree(test_case_suffix)
    ClusterExperiments.side_gaps_vary_virtual_node_ratio(test_case_suffix)
    ClusterExperiments.side_gaps_vs_arbitrary_2_gaps(test_case_suffix)
    ClusterExperiments.side_gaps_vary_node_count(test_case_suffix)
    ClusterExperiments.tscm_sg(test_case_suffix)
    ClusterExperiments.tscm_sg_vary_up_and_down(test_case_suffix)
    # ClusterExperiments.oscm_side_gaps_large_instances(test_case_suffix)
    # ClusterExperiments.oscm_k_gaps_large_instances(test_case_suffix)

    ##### ClusterExperiments.run_micro(test_case_suffix)

    wait_for_processes_to_finish()

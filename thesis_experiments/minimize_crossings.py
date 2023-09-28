# cli args: <flags> <algorithm_to_use> <out_dir>
# flags:
#   --sidegaps
#   --kgaps <how_many>
#   --in_dir <dir_name>
#   --file <file_name>
# either --sidegaps or --kgaps needs to be given
# either --in_dir or --file needs to be given
#
# example:
# python -m thesis_experiments.minimize_crossings --sidegaps --in_dir thesis_experiments/local_tests/in median thesis_experiments/local_tests/out/out1.csv
#


import csv
import getopt
import os
import sys
import time
from typing import NoReturn
import fcntl

from crossing_minimization.barycenter_heuristic import BarycenterImprovedSorter
from crossing_minimization.gurobi_int_lin import GurobiSorter
from crossing_minimization.median_heuristic import ImprovedMedianSorter
from multilayered_graph.multilayered_graph import MultiLayeredGraph
import logging

logger = logging.getLogger(__name__)


def log_and_exit(string: str, err_code: int = 1) -> NoReturn:
    logger.error(string)
    exit(err_code)


alg_names_to_algs = {
    "barycenter": BarycenterImprovedSorter,
    "median": ImprovedMedianSorter,
    "ilp": GurobiSorter,
    "none": None,
}


try:
    # Parsing argument
    argument_list = sys.argv[1:]
    options = ""
    long_options = ["sidegaps", "kgaps=", "in_dir=", "in_file="]
    options_and_values, normal_args = getopt.getopt(
        argument_list, options, long_options
    )
except getopt.error as err:
    log_and_exit(str(err))

side_gaps = k_gaps = in_dir = in_file = None


# checking each argument
for current_argument, current_value in options_and_values:
    if current_argument == "--sidegaps":
        side_gaps = True

    elif current_argument == "--kgaps":
        k_gaps = int(current_value)

    elif current_argument == "--in_dir":
        in_dir = current_value

    elif current_argument == "--in_file":
        in_file = current_value

if not ((side_gaps is None) ^ (k_gaps is None)):
    log_and_exit(
        f"either --sidegaps or --kgaps needs to be given. {side_gaps=}, {k_gaps=}. {sys.argv=}"
    )

if not ((in_dir is None) ^ (in_file is None)):
    log_and_exit("either --in_dir or --file needs to be given")

if len(normal_args) != 2:
    log_and_exit(
        f"Expected 2 arguments, received {len(normal_args)} arguments:."
        f"\n{normal_args}"
        f"\nUsage: python minimize_crossings_cli.py <flags> <algorithm_to_use> <out_dir>"
    )

alg_name, out_csv_file = normal_args
if alg_name not in alg_names_to_algs:
    log_and_exit(f"Only the following algorithms can be used: {alg_names_to_algs}")

# if not os.path.isdir(out_csv_file):
#     log_and_exit(f'"{out_csv_file}" is not a directory')


in_file_paths: list[str] = []
if in_dir is not None:
    if not os.path.isdir(in_dir):
        log_and_exit(f"{in_dir=} is not a directory")
    in_file_paths = os.listdir(in_dir)
else:
    in_dir = "."

if in_file is not None:
    if not os.path.isfile(in_file):
        log_and_exit(f"{in_file=} is not a file")
    in_file_paths = [in_file]

if side_gaps is None:
    side_gaps = False
    assert k_gaps is not None
    gap_count = k_gaps
    gap_type = "kgaps"
else:
    gap_type = "sidegaps"
    gap_count = 2

alg = alg_names_to_algs[alg_name]
for file_path in in_file_paths:
    ml_graph = MultiLayeredGraph.from_proprietary_serialized(
        os.path.join(in_dir, file_path)
    )

    start_ns = time.perf_counter_ns()
    if alg is not None:
        alg().sort_graph(
            ml_graph,
            max_iterations=1,
            only_one_up_iteration=True,
            side_gaps_only=side_gaps,
            max_gaps=gap_count,
        )
    total_s = (time.perf_counter_ns() - start_ns) / 1_000_000_000

    all_nodes_as_list = ml_graph.all_nodes_as_list()

    real_nodes_count = file_path[file_path.find("_r=") + 3 : file_path.find("_v=")]
    virtual_nodes_count = file_path[file_path.find("_v=") + 3 : file_path.find("_p=")]
    average_node_degree = file_path[file_path.find("_p=") + 3 : file_path.find("_id=")]
    instance = file_path.strip(".json")
    crossings = ml_graph.get_total_crossings()
    field_names = [
        "alg_name",
        "gap_type",
        "gap_count",
        "real_nodes_per_layer_count",
        "virtual_nodes_per_layer_count",
        "average_node_degree",
        "instance_name",
        "crossings",
        "time_s",
    ]
    with open(out_csv_file, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # lock file with write access
        f.seek(0, os.SEEK_END)  # seek to end, just in case
        csv_writer = csv.DictWriter(f, fieldnames=field_names)
        csv_writer.writerow(
            {
                "alg_name": alg_name,
                "gap_type": gap_type,
                "gap_count": gap_count,
                "real_nodes_per_layer_count": real_nodes_count,
                "virtual_nodes_per_layer_count": virtual_nodes_count,
                "average_node_degree": average_node_degree,
                "instance_name": instance,
                "crossings": crossings,
                "time_s": total_s,
            }
        )
        fcntl.flock(f, fcntl.LOCK_UN)  # unlock the file

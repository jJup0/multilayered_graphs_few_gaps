import csv
import json
import logging
import os
import sys

logging.basicConfig()
logger = logging.getLogger(__name__)

TEST_COLLECTION_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "local_tests"
)

test_case_pattern = sys.argv[1]


def collect_csv(test_case_name: str):
    test_case_dir = os.path.join(TEST_COLLECTION_DIR, test_case_name)
    single_csv_outs_dir = os.path.join(test_case_dir, "out")
    collected_csv_out_filename = os.path.join(test_case_dir, "out.csv")

    all_lines: list[str] = []
    for single_row_csv_filename in os.listdir(single_csv_outs_dir):
        single_row_csv_file_path = os.path.join(
            single_csv_outs_dir, single_row_csv_filename
        )
        with open(single_row_csv_file_path, "r") as single_row_f:
            line = single_row_f.readline()
            if line.count(",") < 2:
                logger.warning(
                    "%s is most likely faulty. Line= %s", single_row_csv_filename, line
                )
            all_lines.append(line)

    with open(os.path.join(test_case_dir, "info.json")) as f:
        info_json = json.load(f)

    if info_json["expected_results_count"] != len(all_lines):
        logger.warn(
            "Expected %d results for %s found %d",
            info_json["expected_results_count"],
            test_case_name,
            len(all_lines),
        )

    with open(collected_csv_out_filename, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            (
                "alg_name",
                "gap_type",
                "gap_count",
                "nodes_per_layer",
                "virtual_node_fraction",
                "average_node_degree",
                "instance_name",
                "up_and_down_iterations",
                "crossings",
                "time_s",
            )
        )

    with open(collected_csv_out_filename, "a") as f:
        f.writelines(all_lines)


for test_case_dir_name in os.listdir(TEST_COLLECTION_DIR):
    if test_case_pattern in test_case_dir_name:
        collect_csv(test_case_dir_name)

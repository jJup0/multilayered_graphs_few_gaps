import json
import os
import sys
from typing import TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)



class TestCaseInfo(TypedDict):
    constants: dict[str, int | float]
    variable: tuple[str, list[int | float]]
    graph_title: str

def create_graph(test_case_directory:str):
    with open(os.path.join(test_case_directory, "info.json")) as f:
        test_case_info: TestCaseInfo = json.load(f)

    y_data_str = "crossings"
    # y_data_str = "time_s"
    x_data_str = test_case_info["variable"][0].replace(" ", "_")

    df = pd.read_csv(os.path.join(test_case_directory, "out.csv"))
    sns.lineplot(data=df, x=x_data_str, y=y_data_str, hue="alg_name")

    plt.xlabel(x_data_str)
    plt.ylabel(y_data_str)
    plt.title(test_case_info["graph_title"])

    # plt.yscale("log")

    plt.legend(title="Algorithms", loc="best")


    _, test_case_name = os.path.split(os.path.realpath(test_case_directory))
    # Show the plot
    plt.savefig(os.path.join(f"{test_case_name}.png"))


if __name__ == "__main__":
    assert len(sys.argv) > 1
    test_case_name_match = sys.argv[1]

    logger.info("creating plots")
    test_case_root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "local_tests")
    found_test_cases = 0
    for dirname in os.listdir(test_case_root_dir):
        test_case_dir_path = os.path.realpath(os.path.join(test_case_root_dir, dirname))
        if not os.path.isdir(test_case_dir_path):
            continue
        if test_case_name_match in test_case_dir_path:
            logger.info("found matching test case")
            found_test_cases += 1
            create_graph(test_case_dir_path)
    if found_test_cases == 0:
        logger.warning("no testcases found matching %s in %s", test_case_name_match, os.listdir(test_case_root_dir))

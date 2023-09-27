import json
import logging
import os
import sys
import warnings
from typing import TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig()
logger = logging.getLogger(__name__)

# filter out FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


test_case_root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "local_tests"
)


class TestCaseInfo(TypedDict):
    constants: dict[str, int | float]
    variable: tuple[str, list[int | float]]
    graph_title: str


def out_dir(test_case_name_match: str, test_case_directory: str):
    saved_plots_abs_dir = os.path.join(
        os.path.dirname(test_case_root_dir), "saved_plots"
    )
    assert os.path.isdir(saved_plots_abs_dir)
    return os.path.join(
        saved_plots_abs_dir,
        test_case_name_match,
        test_case_directory,
    )


def create_graph(test_case_name_match: str, test_case_directory: str):
    with open(os.path.join(test_case_directory, "info.json")) as f:
        test_case_info: TestCaseInfo = json.load(f)

    df = pd.read_csv(os.path.join(test_case_directory, "out.csv"))
    x_data_str = test_case_info["variable"][0].replace(" ", "_")

    _, test_case_name = os.path.split(os.path.realpath(test_case_directory))
    for y_data_str in ["crossings", "time_s"]:
        sns.lineplot(data=df, x=x_data_str, y=y_data_str, hue="alg_name")

        plt.xlabel(x_data_str)
        plt.ylabel(y_data_str)
        plt.title(test_case_info["graph_title"])
        # plt.yscale("log")
        plt.legend(title="Algorithms", loc="best")

        # save plot to disk
        plt.savefig(
            os.path.join(
                out_dir(test_case_name_match, test_case_directory),
                f"{test_case_name}_{y_data_str}.png",
            )
        )
        plt.clf()

    # separate the data into optimal and heuristic datasets
    # optimal_data = df[df["alg_name"] == "ilp"]

    # calculate ilp ratio
    df["ratio"] = -1
    heuristic_df = df[df["alg_name"] != "ilp"]
    for index, row in heuristic_df.iterrows():
        row["ratio"] = 1
        ilp_rows = df[
            (df["alg_name"] == "ilp")
            & (df["instance_name"] == row["instance_name"])
            & (df["gap_count"] == row["gap_count"])
        ]
        assert len(ilp_rows) == 1, (
            f"NOT EXACTLY ONE RESULT FOR ILP: {ilp_rows.head()=}"
            f'{row["instance_name"], }'
        )
        heuristic_df.at[index, "ratio"] = (
            row["crossings"] / ilp_rows.iloc[0]["crossings"]
        )

    for y_data_str in ["crossings"]:
        sns.lineplot(data=heuristic_df, x=x_data_str, y="ratio", hue="alg_name")

        plt.xlabel(x_data_str)
        plt.ylabel(f"{y_data_str} ratio to ILP")
        plt.title(test_case_info["graph_title"] + "compared to ILP")
        # plt.yscale("log")
        plt.legend(title="Algorithms", loc="best")

        # save plot to disk
        plt.savefig(
            os.path.join(
                out_dir(test_case_name_match, test_case_directory),
                f"{test_case_name}_ratio_{y_data_str}.png",
            )
        )


def find_matching_test_case_dirs_and_plot_data(test_case_name_match: str):
    """
    Finds all testcase directories with `test_case_name_match`
    as a substring and calls create_graph().
    """

    logger.info("creating plots")

    found_test_cases = 0
    for dirname in os.listdir(test_case_root_dir):
        test_case_dir_path = os.path.realpath(os.path.join(test_case_root_dir, dirname))
        if not os.path.isdir(test_case_dir_path):
            continue
        if test_case_name_match in test_case_dir_path:
            logger.info("found matching test case")
            found_test_cases += 1
            create_graph(test_case_name_match, test_case_dir_path)
    if found_test_cases == 0:
        logger.warning(
            "no testcases found matching %s in %s",
            test_case_name_match,
            os.listdir(test_case_root_dir),
        )


if __name__ == "__main__":
    # assert len(sys.argv) > 1
    # test_case_name_match = sys.argv[1]
    test_case_name_match = "v7"
    # argv[1] should be a substring to search testcases for
    find_matching_test_case_dirs_and_plot_data(test_case_name_match)

import io
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
logger.setLevel(logging.DEBUG)

# filter out FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="plt.legend")


test_case_root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "local_tests"
)


class TestCaseInfo(TypedDict):
    constants: dict[str, int | float]
    variable: tuple[str, list[int | float]]
    graph_title: str
    expected_results_count: int


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


def _read_csv_no_filter(csv_real_file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_real_file_path)
    except Exception as err:
        str_err = str(err)
        if "Error tokenizing data. C error: Expected " in str_err:
            line_str_idx = str_err.find("line")
            line_nr = int(str_err[line_str_idx + 4 : str_err.find(",", line_str_idx)])
            raise Exception(f"{csv_real_file_path}:{line_nr} has faulty row: {err}")
        raise err
    return df

def sidegaps_vs_2_gaps_preprocessing(df: pd.DataFrame):
    # changes alg_name from something like "median" to "median sidegaps"
    # todo more optimal way to check if two different values
    if df[df[""]]

def create_graph(test_case_name_match: str, test_case_directory: str):
    # read info file
    with open(os.path.join(test_case_directory, "info.json")) as f:
        test_case_info: TestCaseInfo = json.load(f)
    x_data_str = test_case_info["variable"][0].replace(" ", "_")
    expected_row_count = test_case_info["expected_results_count"]

    # read csv
    csv_real_file_path = os.path.join(test_case_directory, "out.csv")
    df = _read_csv_no_filter(csv_real_file_path)

    # filter invalid data
    row_count = len(df)
    if row_count != expected_row_count:
        _, test_case_name = os.path.split(test_case_directory)
        logger.warning(
            f"Expected {expected_row_count} rows, received {row_count} for {test_case_name}"
        )

    sidegaps_vs_2_gaps_preprocessing(df)

    create_regular_plots(
        test_case_name_match, test_case_directory, test_case_info, x_data_str, df
    )

    create_ratio_plots(
        test_case_name_match,
        test_case_directory,
        test_case_info,
        x_data_str,
        csv_real_file_path,
        df,
    )


def create_ratio_plots(
    test_case_name_match: str,
    test_case_directory: str,
    test_case_info: TestCaseInfo,
    x_data_str: str,
    csv_real_file_path: str,
    df: pd.DataFrame,
):
    _, test_case_name = os.path.split(os.path.realpath(test_case_directory))

    # calculate ilp ratio
    df["ratio-crossings"] = -1
    df["ratio-time_s"] = -1
    heuristic_df = df[df["alg_name"] != "ilp"]
    for index, row in heuristic_df.iterrows():
        ilp_rows = df[
            (df["alg_name"] == "ilp")
            & (df["instance_name"] == row["instance_name"])
            & (df["gap_type"] == row["gap_type"])
            & (df["gap_count"] == row["gap_count"])
        ]
        assert (
            len(ilp_rows) == 1
        ), f"NOT EXACTLY ONE RESULT FOR ILP: {len(ilp_rows)=}\n{row['instance_name']=}\n{csv_real_file_path}:{index}"
        heuristic_df.at[index, "ratio-crossings"] = (
            row["crossings"] / ilp_rows.iloc[0]["crossings"]
        )
        heuristic_df.at[index, "ratio-time_s"] = (
            row["time_s"] / ilp_rows.iloc[0]["time_s"]
        )
        plt.clf()

    for y_data_str in ["crossings", "time_s"]:
        sns.lineplot(
            data=heuristic_df, x=x_data_str, y=f"ratio-{y_data_str}", hue="alg_name"
        )

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
        plt.clf()


def create_regular_plots(
    test_case_name_match: str,
    test_case_directory: str,
    test_case_info: TestCaseInfo,
    x_data_str: str,
    df: pd.DataFrame,
):
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
    assert len(sys.argv) > 1
    test_case_name_match = sys.argv[1]
    # argv[1] should be a substring to search testcases for
    find_matching_test_case_dirs_and_plot_data(test_case_name_match)

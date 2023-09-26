import json
import os
import sys
from typing import TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class TestCaseInfo(TypedDict):
    constants: dict[str, int | float]
    variable: tuple[str, list[int | float]]
    graph_title: str


assert len(sys.argv) > 1
test_case_directory = sys.argv[1]

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

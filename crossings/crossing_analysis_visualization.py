# matplotlib has unknown types for everything, just ignore
# pyright: reportUnknownMemberType=false
from dataclasses import dataclass
from typing import Iterable, TypeAlias

import matplotlib.pyplot as plt

Number_T: TypeAlias = int | float


@dataclass(frozen=True)
class DataSet:
    label: str
    y_values: Iterable[Number_T]


@dataclass(frozen=True)
class GraphLabels:
    xlabel: str
    ylabel: str
    title: str


def draw_crossing_analysis_graph(
    xs: Iterable[Number_T],
    data_sets: Iterable[DataSet],
    graph_labels: GraphLabels,
    *,
    save_to_disk: bool = False,
    save_to_disk_filename: str = "",
    save_to_disk_format: str = "svg"
):
    # plot the datasets
    for dataset in data_sets:
        plt.plot(xs, dataset.y_values, label=dataset.label)

    # logarithmic scale
    # plt.yscale("log")

    # Add labels and title
    plt.xlabel(graph_labels.xlabel)
    plt.ylabel(graph_labels.ylabel)
    plt.title(graph_labels.title)

    # add legend and show in window
    plt.legend()
    plt.show()

    if save_to_disk:
        plt.savefig(save_to_disk_filename, format=save_to_disk_format)

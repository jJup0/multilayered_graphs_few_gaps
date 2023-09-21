import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# TODO maybe automatically store x_data in input

y_data_str = "crossings"
# y_data_str = "time_s"
# x_data_str = "real_nodes_per_layer_count"
x_data_str = "gaps"

df = pd.read_csv(r"thesis_experiments\local_tests\testcase_10-50_kgaps\out.csv")
sns.lineplot(data=df, x=x_data_str, y=y_data_str, hue="alg_name")
# sns.lineplot(data=df, x="real_nodes_per_layer_count ", y="time_s", hue="alg_name")

# Optionally, you can customize the plot using Seaborn functions or Matplotlib settings
# For example, setting labels and title:
plt.xlabel(x_data_str)
plt.ylabel(y_data_str)
plt.title("OSCM")

plt.yscale("log")

plt.legend(title="Algorithms", loc="best")

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# TODO aggregate csvs

y_data_str = "crossings"
df = pd.read_csv(r"thesis_experiments\local_tests\testcase3\out.csv")
sns.lineplot(data=df, x="real_nodes_per_layer_count", y=y_data_str, hue="alg_name")
# sns.lineplot(data=df, x="real_nodes_per_layer_count ", y="time_s", hue="alg_name")

# Optionally, you can customize the plot using Seaborn functions or Matplotlib settings
# For example, setting labels and title:
plt.xlabel("Real node per Layer")
plt.ylabel(y_data_str)
plt.title("OSCM")


plt.legend(title="Algorithms", loc="best")

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# TODO aggregate csvs

df = pd.read_csv(r"thesis_experiments\local_tests\testcase2\out.csv")
sns.lineplot(data=df, x="real_nodes_per_layer_count", y="crossings", hue="alg_name")
# sns.lineplot(data=df, x="real_nodes_per_layer_count ", y="time_s", hue="alg_name")

# Optionally, you can customize the plot using Seaborn functions or Matplotlib settings
# For example, setting labels and title:
plt.xlabel("Real node per Layer")
plt.ylabel("Time taken to minimize")
plt.title("OSCM")


plt.legend(title="Algorithms", loc="best")

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# TODO aggregate csvs

df = pd.read_csv("thesis_experiments/local_tests/out/out2.csv")


sns.lineplot(data=df, x="time_s", y="crossings", hue="alg_name")

# Optionally, you can customize the plot using Seaborn functions or Matplotlib settings
# For example, setting labels and title:
plt.xlabel("X-Axis Label")
plt.ylabel("Y-Axis Label")
plt.title("Line Plot")


plt.legend(title="Z Values", loc="best")

# Show the plot
plt.show()

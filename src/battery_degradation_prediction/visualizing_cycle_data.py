"""visualization module"""

import matplotlib.pyplot as plt
import seaborn as sns

# visulizing_cycle_data(dataframe)

# We want to display the heading names of the dataframe as a list,
# and using that list plot features
# column_headers = list(dataframe.columns.values)

# Now we want to plot certain features... in particular, we want

# i) discharge cap vs cycle
def plot_cycle_capacity(df_discharge):
    """plot cycle capacity"""
    _, ax = plt.subplots(3, 1, figsize=(14, 26))
    sns.scatterplot(data=df_discharge, x="cycle", y="capacity", ax=ax[0])
    return 0


# ii) temperatrue_measured vs time
def plot_temperatrue_time(df_discharge):
    """plot temperatrue time"""
    _, ax = plt.subplots(3, 1, figsize=(14, 26))
    sns.scatterplot(
        data=df_discharge,
        x="test_time(h)",
        y="temperatrue_measured",
        hue="cycle",
        ax=ax[1],
    )
    return 0


# iii) discharge cap vs time of cycle

# iv) V vs time
def plot_temperatrue_time_second(df_discharge):
    """plot temperatrue time"""
    _, ax = plt.subplots(3, 1, figsize=(14, 26))
    sns.scatterplot(
        data=df_discharge, x="test_time(h)", y="voltage_measured", hue="cycle", ax=ax[2]
    )
    return 0


# v) V vs capacity

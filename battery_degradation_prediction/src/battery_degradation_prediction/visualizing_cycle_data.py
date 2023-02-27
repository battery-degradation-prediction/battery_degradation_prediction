# visulizing_cycle_data(dataframe)
fig, ax = plt.subplots(3,1, figsize = (14,26))

# We want to display the heading names of the dataframe as a list, and using that list plot features
# column_headers = list(dataframe.columns.values)

# Now we want to plot certain features... in particular, we want 

# i) discharge cap vs cycle
sns.scatterplot(data=df_discharge, x="cycle", y="capacity", ax=ax[0])

# ii) temperatrue_measured vs time
sns.scatterplot(data=df_discharge, x="test_time(h)", y="temperatrue_measured", hue='cycle', ax=ax[1])

# iii) discharge cap vs time of cycle

# iv) V vs time
sns.scatterplot(data=df_discharge, x="test_time(h)", y="voltage_measured", hue='cycle', ax=ax[2])

#v) V vs capacity

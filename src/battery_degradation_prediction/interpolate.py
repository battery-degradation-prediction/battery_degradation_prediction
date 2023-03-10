# features_list = ['temperature_measured', 'voltage_measured', 'current_measured', 'capacity']
def spine_interpolate(dataframe, features_list) -> pd.DataFrame:
    """
    Spine interpolate in a discharge dataframe using cubic spline interpolation.

    Args:
    - df_discharge_new (pandas.DataFrame): discharge dataframe containing at least the following columns:
        - cycle (int): cycle number
        - elapsed_time_per_cycle (float): elapsed time within each cycle
        - features_list (list): list of column names to interpolate

    Returns:
    - final_df (pandas.DataFrame): interpolated dataframe with the following columns:
        - cycle (int): cycle number
        - time (float): elapsed time within each cycle
        - columns specified in features_list (float): interpolated values for each feature
    """
    end_cycle = dataframe['cycle'].max()
    dfs = []
    for i in range(1, end_cycle+1):
        cycle = dataframe.loc[dataframe['cycle'] == i]
        df_features = []
        x_discharge = cycle['elapsed_time_per_cycle']
        time_x = np.linspace(0,x_discharge.max(),num=100)
        for feature in features_list:
            y_discharge = cycle[feature]        
            y_try = y_discharge.to_numpy()
            x_try = x_discharge.to_numpy()
            cs = CubicSpline(x_try, y_try)
            cs_list = cs(time_x)
            df_feature = pd.DataFrame(cs_list, columns=[feature])
            df_features.append(df_feature)
        df_cycle = pd.concat(df_features, axis=1)
        df_cycle['time'] = time_x
        df_cycle['cycle'] = i
        dfs.append(df_cycle)

    final_df = pd.concat(dfs, ignore_index=True)
    return final_df
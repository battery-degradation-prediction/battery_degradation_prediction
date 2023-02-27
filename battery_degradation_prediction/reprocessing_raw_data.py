import numpy as np
import pandas as pd
from datetime import datetime
from load_data import load_data

# func to convert datetimes in dataframe to time objects    WORKS
def convert_datetime_str_to_obj(date_time_str:str):
    """This function converts a string into a datatime object
    
    Parameters
    ----------
    date_time_str : str
        A string object that records the datetime information in the form
        YYYY-mm-HH-MM-SS

    Returns
    -------
    time_object : datetime
        A datetime object containing year, month, day, hour, minute, and second.
    """

    time_object = datetime.strptime(date_time_str, '%Y-%m-%d-%H-%M-%S')
    return time_object

# func to remove current outliers (ie current that's is close to zero) from dataframe    NOT SURE
def find_outlier(dataframe, column):
    """
    """
    # Find first and third quartile
    q1 = dataframe[column].quantile(0.1)  # return Series
    q3 = dataframe[column].quantile(0.9)

    # Find interquartile range
    IQR = q3 - q1

    # Find lower and upper bound
    q_low = q1 - 1.5*IQR
    q_high = q3 + 1.5*IQR
    return q_low, q_high

def remove_outlier(dataframe, column):
    """
    """
    # Find outliers Series
    q_low, q_high = find_outlier(dataframe, column)

    # Remove outliers
    dataframe[column] = dataframe[column][dataframe[column] > q_low]
    dataframe[column] = dataframe[column][dataframe[column] < q_high]

    return dataframe

# func to remove NaN in columns
def remove_nan_from_data(dataframe):
    dataframe.dropna(subset=list(dataframe.columns.values))
    return dataframe

# func to calc test time in hours
def calc_test_time_from_datetime(target_time:datetime, start_time:datetime) -> float:
    """
    
    Returns
    -------
    test_time : datetime.datetime
        unit: hour
    """
    #test_time = []
    second_to_hour = 1./3600.
    #for i in range(len(dataframe)):
    #    test_time.append(((dataframe['time'][i] - dataframe['time'][0]).seconds)*second_to_hour)
    time_elapsed = (target_time - start_time).seconds * second_to_hour
    #dataframe['test_time(h)'] = test_time
    return time_elapsed

# func to look only at the discharge cycles within dataframe
def isolate_discharge_cyc_data(dataframe):
    dataframe = dataframe[dataframe['type'] == 'discharging']
    return dataframe

def add_elapsed_time_per_cycle(df:pd.DataFrame) -> list[float]:
    """
    """
    time_elasped_list = []
    for time in df.groupby("cycle")["time"]:
        start_time = time[1].iloc[0]
        #print('='* 15)
        for target_time in time[1]:
            #print(f"{start_time} / {target_time}")
            time_elasped = calc_test_time_from_datetime(target_time, start_time)
            time_elasped_list.append(time_elasped)
    return time_elasped_list
def main():
    path = '../data/B0005.csv'
    df = load_data(path)
    #df = df.iloc[13000:15000]
    
    df["time"] = df["datetime"].apply(convert_datetime_str_to_obj)
    df["elapsed_time"] = df["time"].apply(calc_test_time_from_datetime, args=(df['time'].iloc[0],))
    df_discharge = isolate_discharge_cyc_data(df)
    df_discharge["elapsed_time_per_cycle"] = add_elapsed_time_per_cycle(df_discharge)
    #remove_outlier(df, "current_measured")
    
    #add_time_elapsed_col(df_discharge)
    for group in df_discharge.groupby("cycle"):
        print(group[1].head())
        print("=" * 15)
    return 

if __name__ == "__main__":
    main()



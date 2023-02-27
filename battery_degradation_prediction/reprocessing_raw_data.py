import numpy as np
import pandas as pd
from datetime import datetime

# func to convert datetimes in dataframe to time objects    WORKS
def convert_date_time_str_to_obj(date_time_str:str):
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
    # Find first and third quartile
    q1 = dataframe[column].quantile(0.1)
    q3 = dataframe[column].quantile(0.9)

    # Find interquartile range
    IQR = q3 - q1

    # Find lower and upper bound
    q_low = q1 - 1.5*IQR
    q_high = q3 + 1.5*IQR

    # Remove outliers
    dataframe[column] = dataframe[column][dataframe[column] > q_low]
    dataframe[column] = dataframe[column][dataframe[column] < q_high]

    #dataframe[f"{column}_clean"] = dataframe[column][dataframe[column] > q_low]
    #dataframe[f"{column}_clean"] = dataframe[column][dataframe[column] < q_high]
    # dataframe[f"{column}_clean"] = dataframe[column][dataframe[column] > q_low or dataframe[column] < q_high]

    return dataframe

# func to remove NAN in columns
def remove_nan_from_data(dataframe):
    dataframe.dropna(subset=list(dataframe.columns.values))
    return dataframe

# func to calc test time in hours
def calc_test_time_in_h_from_datetime(dataframe):
    test_time = []
    second_to_hour = (1/3600)
    for i in range(len(dataframe)):
        test_time.append(((dataframe['time'][i] - dataframe['time'][0]).seconds)*second_to_hour)
    dataframe['test_time(h)'] = test_time
    return dataframe

# func to look only at the discharge cycles within dataframe
def isolating_discharge_cyc_data(dataframe):
    dataframe = dataframe[dataframe['type'] == 'discharging']
    return dataframe

def main():

if __name__ == "__main__":
    main()



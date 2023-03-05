"""preprocessing data module"""
from typing import Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from battery_degradation_prediction.load_data import load_data

# func to convert datetimes in dataframe to time objects    WORKS
def convert_datetime_str_to_obj(date_time_str: str) -> datetime:
    """This function converts a string into a datatime object.

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

    time_object = datetime.strptime(date_time_str, "%Y-%m-%d-%H-%M-%S")
    return time_object


# func to remove current outliers (ie current that's is close to zero) from dataframe    NOT SURE
def find_outlier(dataframe: pd.DataFrame, column: str) -> Tuple[pd.Series, pd.Series]:
    """Find the samples that are outside the lower and upper bound of IQR.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing outliers
    column : str
        The column name for searching outliers

    Returns
    -------
    q_low : pd.Series
        The pd.Series that are lower than 10% of the column.
    q_high : pd.Series
        The pd.Series that are higher than 90% of the column.
    """
    # Find first and third quartile
    quantile_1 = dataframe[column].quantile(0.1)  # return Series
    quantile_3 = dataframe[column].quantile(0.9)

    # Find interquartile range
    iqr = quantile_3 - quantile_1

    # Find lower and upper bound
    q_low = quantile_1 - 1.5 * iqr
    q_high = quantile_3 + 1.5 * iqr
    return q_low, q_high


def remove_outlier(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove outliers in column of the dataframe and replace them with NaN, and the
    inplace is always True.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing outliers
    column : str
        The column name for searching outliers

    Returns
    -------
    dataframe : pd.DataFrame
        The dataframe with a clean column.
    """
    # Find outliers Series
    q_low, q_high = find_outlier(dataframe, column)

    # Remove outliers
    dataframe[column] = dataframe[column][dataframe[column] > q_low]
    dataframe[column] = dataframe[column][dataframe[column] < q_high]

    return dataframe


# func to remove NaN in columns
def remove_nan_from_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove NaN from dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing outliers

    Returns
    -------
    dataframe : pd.DataFrame
        The dataframe with no NaNs in columns
    """
    dataframe.dropna(subset=list(dataframe.columns.values))
    return dataframe


# func to calc test time in hours
def calc_test_time_from_datetime(target_time: datetime, start_time: datetime) -> float:
    """Compute elapsed time.

    Parameters
    ----------
    target_time : datetime
        target time
    start_time : datetime
        start time

    Returns
    -------
    test_time : datetime
        unit: hour
    """

    second_to_hour = 1.0 / 3600.0
    time_elapsed = (target_time - start_time).seconds * second_to_hour
    return time_elapsed


# func to look only at the discharge cycles within dataframe
def isolate_discharge_cyc_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return the dataframe with only discharge cycle.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing entire cycle data

    Returns
    -------
    dataframe : pd.DataFrame
        The dataframe containing only the discharging data
    """

    dataframe = dataframe[dataframe["type"] == "discharging"]
    return dataframe


def add_elapsed_time_per_cycle(df: pd.DataFrame) -> list[float]:
    """Return the elapsed time per cycle.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the continuous time column

    Returns
    -------
    time_elasped_list : list[float]
    """
    time_elasped_list = []
    for time in df.groupby("cycle")["time"]:
        start_time = time[1].iloc[0]
        for target_time in time[1]:
            time_elasped = calc_test_time_from_datetime(target_time, start_time)
            time_elasped_list.append(time_elasped)
    return time_elasped_list


def remove_jump_voltage(df_discharge: pd.DataFrame):
    """Remove the rows that jump back from minima inplacely.

    Parameters
    ----------
    df_discharge : pd.DataFrame
        The dataframe containing only the discharge cycles

    Returns
    -------
    df_discharge : pd.DataFrame
        The dataframe containing only the discharge cycles without jump back rows
    """
    cummulative_num = 0
    drop_ranges = []
    for voltage_group in df_discharge.groupby("cycle")["voltage_measured"]:
        min_voltage_index = np.argmin(voltage_group[1])
        num_group = voltage_group[1].shape[0]
        drop_ranges.append(
            range(cummulative_num + min_voltage_index + 1, cummulative_num + num_group)
        )
        cummulative_num += num_group
    drop_list = [r for ranges in drop_ranges for r in ranges]
    df_discharge.drop(drop_list, inplace=True)
    return df_discharge


def capacity_during_discharge(df_discharge):
    """TODO"""
    capcity_during_discharge_list = []
    for i in range(len(df_discharge)):
        capcity_during_discharge_list.append(abs(df_discharge['elapsed_time_per_cycle'][i] * df_discharge['current_measured'][i]))
    df_discharge["capcity_during_discharge"] = capcity_during_discharge_list
    return 

def plot_remove_jump_voltage(df_discharge):
    _, ax = plt.subplots(1, 2, figsize=(18, 5))
    sns.scatterplot(data=df_discharge, x="time", y="voltage_measured", hue="cycle", ax=ax[0])
    remove_jump_voltage(df_discharge)
    sns.scatterplot(data=df_discharge, x="time", y="voltage_measured", hue="cycle", ax=ax[1])
    ax[0].set_title("Raw voltage data")
    ax[1].set_title("Remove jump voltage")
    plt.show()
    return


def main():
    """Main function"""
    path = "../../data/B0005.csv"
    df = load_data(path)
    df = df.iloc[:10500]

    df["time"] = df["datetime"].apply(convert_datetime_str_to_obj)
    df["elapsed_time"] = df["time"].apply(calc_test_time_from_datetime, args=(df["time"].iloc[0],))
    df_discharge = isolate_discharge_cyc_data(df)
    time_elasped_list = add_elapsed_time_per_cycle(df_discharge)
    df_discharge.insert(len(df_discharge.columns), "elapsed_time_per_cycle", time_elasped_list)
    capacity_during_discharge(df_discharge)
    df_discharge.reset_index(drop=True, inplace=True)

    return df




if __name__ == "__main__":
    main()

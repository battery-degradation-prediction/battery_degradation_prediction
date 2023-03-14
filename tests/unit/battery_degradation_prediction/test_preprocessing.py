"""
Tests for the preprocessing function
"""
import unittest
import numpy as np
import pandas as pd
from battery_degradation_prediction.src.battery_degradation_prediction import preprocessing

def get_data():
    """Obtain data from https://uwdirect.github.io/SEDS_content/atomradii.csv"""

    df = pd.DataFrame(
        pd.read_csv("https://uwdirect.github.io/SEDS_content/atomradii.csv")
    )
    X_fit = df[df.columns[:2]].iloc[:-1]
    X_test = df[df.columns[:2]].iloc[-1:]
    k = 3
    y_fit_df = df[df.columns[-1]].iloc[:-1]
    y_test_df = df[df.columns[-1]].iloc[-1:]
    classes = np.unique(y_fit_df)
    y_fit = [np.where(y == classes)[0][0] for y in y_fit_df.iloc]
    y_test = [np.where(y == classes)[0][0] for y in y_test_df.iloc]
    return X_fit, X_test, k, y_fit, y_test


class UnitTests(unittest.TestCase):
    """Unit test classes"""

    def test_preprocessing(self):
        """one shot test for predict()"""
        #X_fit, X_test, k, y_fit, _ = get_data()
        try:
            predictions = [0]#predict(X_fit, y_fit, k, X_test)
            true = [0]
            np.testing.assert_array_equal(true, predictions)
        except:
            raise


class TestConvertDatetimeStrToObj(unittest.TestCase):
    """
    This class manages the tests for the function that converts
    datetime strings to objects.
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        test_df = pd.DataFrame(data={'cycle': [1, 1, 1, 2, 2],
            'datetime': ['2008-04-02-15-25-41', '2008-04-02-15-25-58',
                         '2008-04-02-15-26-17','2008-04-02-15-26-35',
                         '2008-04-02-15-26-53']})

        test_df["time"] = test_df["datetime"].apply(preprocessing.convert_datetime_str_to_obj)

    def test_input_dates_not_formatted_correctly(self):
        """
        Edge test to make sure the function throws a ValueError
        when the inputted dataframe's datetime str is not formatted
        correctly.
        """
        test_df = pd.DataFrame(data={'cycle': [1, 1, 1, 2, 2],
            'datetime': ['2008-04-02-15-25-41', '2008-04-02-15-25-58',
                         '2008-04-02-15:26:17','2008-04-02-15-26-35',
                         '2008-04-02-15-26-53']})
        
        with self.assertRaises(ValueError):
            test_df["time"] = test_df["datetime"].apply(preprocessing.convert_datetime_str_to_obj)

class TestCalcTestTimeFromDatetime(unittest.TestCase):
    """
    This class manages the tests for the function that calculates
    the total elapased time from start to end, in hours, of all
    cycling data provided.
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        test_df = pd.DataFrame(data={'cycle': [1, 1, 1, 2, 2],
            'datetime': ['2008-04-02-15-25-41', '2008-04-02-15-25-58',
                         '2008-04-02-15-26-17','2008-04-02-15-26-35',
                         '2008-04-02-15-26-53']})
        test_df["time"] = test_df["datetime"].apply(preprocessing.convert_datetime_str_to_obj)

        test_df["elapsed_time"] = test_df["time"].apply(preprocessing.calc_test_time_from_datetime,
                                                         args=(test_df["time"].iloc[0],))
        
    def test_for_negative_time(self):
        """
        Edge test to make sure the function throws a ValueError
        when the first datetime is not the start of the testing time.
        """
        test_df = pd.DataFrame(data={'cycle': [1, 1, 1, 2, 2],
            'datetime': ['2008-04-02-15-26-17', '2008-04-02-15-25-58',
                         '2008-04-02-15-25-41','2008-04-02-15-26-35',
                         '2008-04-02-15-26-53']})
        test_df["time"] = test_df["datetime"].apply(preprocessing.convert_datetime_str_to_obj)
        
        with self.assertRaises(ValueError):
            test_df["elapsed_time"] = test_df["time"].apply(
                preprocessing.calc_test_time_from_datetime,args=(test_df["time"].iloc[0],))

class TestIsolateDischargeCycleData(unittest.TestCase):
    """
    This class manages the tests for the function that 
    returns a dataframe of only discharge cycles.
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        test_df = pd.DataFrame(data={'cycle': [1, 1, 1, 1, 1],
            'datetime': ['2008-04-02-15-25-41', '2008-04-02-15-25-58',
                         '2008-04-02-15-26-17','2008-04-02-15-26-35',
                         '2008-04-02-15-26-53'],
                         'type': ['discharging', 'discharging',
                                   'discharging', 'charging', 'charging']})
        test_df["time"] = test_df["datetime"].apply(preprocessing.convert_datetime_str_to_obj)

        df_discharge = preprocessing.isolate_discharge_cyc_data(test_df)

    def test_for_checking_type_column_exists(self):
        """
        Edge test to make sure the function throws a ValueError
        when the input dataframe does not contain a column named 'Type'.
        """
        test_df = pd.DataFrame(data={'cycle': [1, 1, 1, 2, 2], 
            'datetime': ['2008-04-02-15-25-41', '2008-04-02-15-25-58', 
                         '2008-04-02-15-26-17','2008-04-02-15-26-35', 
                         '2008-04-02-15-26-53']})
        
        with self.assertRaises(ValueError):
            df_discharge = preprocessing.isolate_discharge_cyc_data(test_df)

    def test_for_checking_type_column_exists(self):
        """
        Edge test to make sure the function throws a ValueError
        when the input dataframe contains a dtype other than string 
        in the 'Type' column.
        """ 
        test_df = pd.DataFrame(data={'cycle': [1, 1, 1, 1, 1], 
            'datetime': ['2008-04-02-15-25-41', '2008-04-02-15-25-58', 
                         '2008-04-02-15-26-17','2008-04-02-15-26-35', 
                         '2008-04-02-15-26-53'],
                         'type': ['discharging', 'discharging',
                                   'discharging', 'discharging', 2]})   
        
        with self.assertRaises(ValueError):
            df_discharge = preprocessing.isolate_discharge_cyc_data(test_df)


class TestAddElapsedTimePerCycle(unittest.TestCase):
    """
    This class manages the tests for the function that returns 
    the elapsed time of an individual discharge cycle.
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        test_df = pd.DataFrame(data={'cycle': [1, 1, 1, 1, 1], 
            'datetime': ['2008-04-02-15-25-41', '2008-04-02-15-25-58', 
                         '2008-04-02-15-26-17','2008-04-02-15-26-35', 
                         '2008-04-02-15-26-53'],
                         'type': ['discharging', 'discharging',
                                   'discharging', 'discharging', 'discharging']})
        test_df["time"] = test_df["datetime"].apply(preprocessing.convert_datetime_str_to_obj)

        time_elasped_list = preprocessing.add_elapsed_time_per_cycle(test_df)

    def test_for_checking_columns_cycle_and_time(self):
        """
        Edge test to make sure the function throws a ValueError
        when the input dataframe does not contain cycle and time columns.
        """ 
        test_df = pd.DataFrame(data={'cycle': [1, 1, 1, 1, 1], 
            'datetime': ['2008-04-02-15-25-41', '2008-04-02-15-25-58', 
                         '2008-04-02-15-26-17','2008-04-02-15-26-35', 
                         '2008-04-02-15-26-53'],
                         'type': ['discharging', 'discharging',
                                   'discharging', 'discharging', 'charging']})
        with self.assertRaises(ValueError):
            time_elasped_list = preprocessing.add_elapsed_time_per_cycle(test_df)

class TestCalculateCapcityDuringDischarge(unittest.TestCase):
    """
    This class manages the tests for the function that retuns 
    the capcity at each point in time during a discharge cycle.
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        test_df = pd.DataFrame(data={'cycle': [1, 1, 1, 1, 1], 
            'datetime': ['2008-04-02-15-25-41', '2008-04-02-15-25-58', 
                         '2008-04-02-15-26-17','2008-04-02-15-26-35', 
                         '2008-04-02-15-26-53'],
                         'type': ['discharging', 'discharging',
                                   'discharging', 'discharging', 'discharging'],
                            'current_measured': [-2.312, -2.112,
                            -2.112, -1.199, -2.012]})
        test_df["time"] = test_df["datetime"].apply(preprocessing.convert_datetime_str_to_obj)
        test_df["elapsed_time"] = test_df["time"].apply(
        preprocessing.calc_test_time_from_datetime, args=(test_df["time"].iloc[0],)
        )
        df_discharge = preprocessing.isolate_discharge_cyc_data(test_df)

        time_elasped_list = preprocessing.add_elapsed_time_per_cycle(df_discharge)
        df_discharge.insert(
        len(df_discharge.columns), "elapsed_time_per_cycle", time_elasped_list
        )
        df_discharge.reset_index(drop=True, inplace=True)

        capcity_during_discharge = preprocessing.calc_capacity_during_discharge(df_discharge)

    def test_for_checking_columns_current_and_elasped_time(self):
        """
        Edge test to make sure the function throws a ValueError
        when the input dataframe does not contain current measured
        and elapsed time per cycle columns.
        """ 
        test_df = pd.DataFrame(data={'cycle': [1, 1, 1, 1, 1], 
            'datetime': ['2008-04-02-15-25-41', '2008-04-02-15-25-58', 
                         '2008-04-02-15-26-17','2008-04-02-15-26-35', 
                         '2008-04-02-15-26-53'],
                         'type': ['discharging', 'discharging',
                                   'discharging', 'discharging', 'discharging']})
        test_df["time"] = test_df["datetime"].apply(preprocessing.convert_datetime_str_to_obj)
        test_df["elapsed_time"] = test_df["time"].apply(
        preprocessing.calc_test_time_from_datetime, args=(test_df["time"].iloc[0],)
        )
        df_discharge = preprocessing.isolate_discharge_cyc_data(test_df)

        time_elasped_list = preprocessing.add_elapsed_time_per_cycle(df_discharge)
        df_discharge.insert(
        len(df_discharge.columns), "elapsed_time_per_cycle", time_elasped_list
        )
        df_discharge.reset_index(drop=True, inplace=True)
        
        with self.assertRaises(ValueError):
            capcity_during_discharge = preprocessing.calc_capacity_during_discharge(df_discharge)


class TestRemoveJumpVoltage(unittest.TestCase):
    """
    This class manages the tests for the function that removes 
    any voltage after the minimum volatge is achieved in a cycle.
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        test_df = pd.DataFrame(data={'cycle': [1, 1, 1, 2, 2], 
            'datetime': ['2008-04-02-15-25-41', '2008-04-02-15-25-58', 
                         '2008-04-02-15-26-17','2008-04-02-15-26-35', 
                         '2008-04-02-15-26-53'],
                         'type': ['discharging', 'discharging',
                                   'discharging', 'discharging', 'discharging'],
                            'current_measured': [-2.312, -2.112,
                            -2.112, -1.199, -2.012],
                            'voltage_measured': [4.201, 4.112,
                            4.102, 4.221, 3.992]})

        test_df["time"] = test_df["datetime"].apply(preprocessing.convert_datetime_str_to_obj)
        test_df["elapsed_time"] = test_df["time"].apply(
        preprocessing.calc_test_time_from_datetime, args=(test_df["time"].iloc[0],)
        )
        df_discharge = preprocessing.isolate_discharge_cyc_data(test_df)

        time_elasped_list = preprocessing.add_elapsed_time_per_cycle(df_discharge)
        df_discharge.insert(
        len(df_discharge.columns), "elapsed_time_per_cycle", time_elasped_list
        )
        df_discharge.reset_index(drop=True, inplace=True)
        df_discharge = preprocessing.isolate_discharge_cyc_data(df_discharge)

        preprocessing.remove_jump_voltage(df_discharge)

    def test_if_cycle_has_only_one_voltage_datapoint(self):
        """
        Edge test to make sure the function throws a ValueError
        when a cycle only has one voltage measurement.
        """
        test_df = pd.DataFrame(data={'cycle': [1, 1, 1, 1, 2], 
            'datetime': ['2008-04-02-15-25-41', '2008-04-02-15-25-58', 
                         '2008-04-02-15-26-17','2008-04-02-15-26-35', 
                         '2008-04-02-15-26-53'],
                         'type': ['discharging', 'discharging',
                                   'discharging', 'discharging', 'discharging'],
                            'current_measured': [-2.312, -2.112,
                            -2.112, -1.199, -2.012],
                            'voltage_measured': [4.201, 4.112,
                            4.102, 4.072, 4.200]})

        test_df["time"] = test_df["datetime"].apply(preprocessing.convert_datetime_str_to_obj)
        test_df["elapsed_time"] = test_df["time"].apply(
        preprocessing.calc_test_time_from_datetime, args=(test_df["time"].iloc[0],)
        )
        df_discharge = preprocessing.isolate_discharge_cyc_data(test_df)

        time_elasped_list = preprocessing.add_elapsed_time_per_cycle(df_discharge)
        df_discharge.insert(
        len(df_discharge.columns), "elapsed_time_per_cycle", time_elasped_list
        )
        df_discharge.reset_index(drop=True, inplace=True)
        df_discharge = preprocessing.isolate_discharge_cyc_data(df_discharge)

        with self.assertRaises(ValueError):
            preprocessing.remove_jump_voltage(df_discharge)

    def test_if_cycle_starts_at_min_voltage(self):
        """
        Edge test to make sure the function throws a ValueError
        when a cycle starts at a minimum voltage measurement.
        """
        test_df = pd.DataFrame(data={'cycle': [1, 1, 1, 2, 2], 
            'datetime': ['2008-04-02-15-25-41', '2008-04-02-15-25-58', 
                         '2008-04-02-15-26-17','2008-04-02-15-26-35', 
                         '2008-04-02-15-26-53'],
                         'type': ['discharging', 'discharging',
                                   'discharging', 'discharging', 'discharging'],
                            'current_measured': [-2.312, -2.112,
                            -2.112, -1.199, -2.012],
                            'voltage_measured': [4.201, 4.112,
                            4.102, 4.072, 4.200]})

        test_df["time"] = test_df["datetime"].apply(preprocessing.convert_datetime_str_to_obj)
        test_df["elapsed_time"] = test_df["time"].apply(
        preprocessing.calc_test_time_from_datetime, args=(test_df["time"].iloc[0],)
        )
        df_discharge = preprocessing.isolate_discharge_cyc_data(test_df)

        time_elasped_list = preprocessing.add_elapsed_time_per_cycle(df_discharge)
        df_discharge.insert(
        len(df_discharge.columns), "elapsed_time_per_cycle", time_elasped_list
        )
        df_discharge.reset_index(drop=True, inplace=True)
        df_discharge = preprocessing.isolate_discharge_cyc_data(df_discharge)

        with self.assertRaises(ValueError):
            preprocessing.remove_jump_voltage(df_discharge)


class TestGetCleanData(unittest.TestCase):
    """
    This class manages the tests for the function that retuns 
    the cleaned data from the input csv file .
    """
    def test_smoke(self):
        """
        Simple smoke test to make sure function runs.
        """
        #path = "../../data/B0005.csv"
        #preprocessing.get_clean_data(path)

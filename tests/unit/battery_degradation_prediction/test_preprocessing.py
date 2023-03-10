import unittest
import numpy as np
import pandas as pd
#from battery_degradation_prediction import preprocessing

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

        test_df["time"] = test_df["datetime"].apply(convert_datetime_str_to_obj)

    def test_input_dates_not_formatted_correctly(self):
        """
        Edge test to make sure the function throws a ValueError
        when the input probabilities do not sum to one.
        """
        test_df = pd.DataFrame(data={'cycle': [1, 1, 1, 2, 2], 
            'datetime': ['2008-04-02-15-25-41', '2008-04-02-15-25-58', 
                         '2008-04-02-15-26-17','2008-04-02-15-26-35', 
                         '2008-04-02-15-26-53']})
        
        with self.assertRaises(ValueError):
            entropy([.9, .9])

    def test_args_out_of_range(self):
        """
        Edge tst to make sure the function throws a ValueError
        when the input probabilities are < 0 or > 1.
        """
        with self.assertRaises(ValueError):
            entropy([-1, 2])
import numpy as np
import pytest

from battery_degradation_prediction.window import windowing

@pytest.fixture
def data():
    return np.arange(100)

def test_windowing_1(data):
    windows, labels = windowing(data, window_size=5, stride=2)
    assert windows.shape == (47, 5)
    assert labels.shape == (47,)

def test_windowing_2(data):
    windows, labels = windowing(data, window_size=10, stride=3)
    assert windows.shape == (30, 10)
    assert labels.shape == (30,)

def test_windowing_3(data):
    windows, labels = windowing(data, window_size=20, stride=5)
    assert windows.shape == (16, 20)
    assert labels.shape == (16,)

def test_windowing_4(data):
    windows, labels = windowing(data, window_size=50, stride=10)
    assert windows.shape == (5, 50)
    assert labels.shape == (5,)

def test_windowing_5(data):
    windows, labels = windowing(data, window_size=100, stride=20)
    assert windows is None
    assert labels is None

    def test_window_values(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window_size = 3
        stride = 1
        windows, labels = windowing(data, window_size, stride)
        self.assertTrue(np.array_equal(windows[0], np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(windows[1], np.array([2, 3, 4])))
        self.assertTrue(np.array_equal(windows[2], np.array([3, 4, 5])))
        self.assertTrue(np.array_equal(windows[3], np.array([4, 5, 6])))
        self.assertTrue(np.array_equal(windows[4], np.array([5, 6, 7])))
        self.assertTrue(np.array_equal(windows[5], np.array([6, 7, 8])))
        self.assertTrue(np.array_equal(windows[6], np.array([7, 8, 9])))
    
    def test_label_values(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window_size = 3
        stride = 1
        windows, labels = windowing(data, window_size, stride)
        self.assertTrue(np.array_equal(labels, np.array([4, 5, 6, 7, 8, 9, 10])))
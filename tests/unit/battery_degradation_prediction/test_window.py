import numpy as np
import pytest
from battery_degradation_prediction.window import windowing_numpy


def test_windowing_numpy_invalid_inputs():
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]])
    window_size = 3
    stride = 1
    expected_output = (None, None)
    result = windowing_numpy(data, window_size, stride)
    np.testing.assert_equal(result, expected_output)



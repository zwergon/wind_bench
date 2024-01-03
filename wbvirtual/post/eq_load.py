import numpy as np
from scipy.signal import butter, filtfilt

from wbvirtual.post.rainflow import rainflow


def _turningpoints(lst):
    dx = np.diff(lst)
    ind = np.where(dx[1:] * dx[:-1] < 0)
    return np.squeeze(lst[np.array(ind) + 1])


def eq_load(values, m, n_eq=1.0e6):
    """
    Compute rainflow for specified cut off frequencies & statical informations

    Arguments
    ---------
    times : numpy.ndarray Array of time values
    values : numpy.ndarray Array of values for rainflow
    m  : exponent of Wholer curve

    Returns
    -------
    cutoffFrequencies : numpy.ndarray Cutoff frequencies values used for rainflow
    DELm3_array : numpy.ndarray Rainflow values at cuttof frequencies
    """

    # Band Pass filtering
    order = 5
    low = 0.99
    b, a = butter(order, low, btype="lowpass", analog=False)
    values_filter = filtfilt(b, a, values)

    # Rainflow
    values_points = _turningpoints(values_filter)
    if values_points.size > 0:
        values_rainflow = rainflow(values_points)
    else:
        values_rainflow = np.zeros((5, 0))

    # DEL
    cutoffN = values_rainflow[3]
    valCycles = values_rainflow[
        0
    ]  # stress ranges. it should be divided by 2 to obtain the same Matlab output (on which we have feedback)

    DEL = np.power(
        sum(cutoffN * np.power(valCycles, m)) / n_eq, 1.0 / m
    )  # attention fabien modification

    return DEL

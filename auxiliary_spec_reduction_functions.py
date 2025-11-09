import numpy as np
from scipy.signal import find_peaks, savgol_filter
from astropy.modeling.models import Linear1D

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def line_maker(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y2 - (slope * x2)
    line = Linear1D(slope = slope, intercept = intercept)
    return line


def first_local_max(arr):
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            return i
    return len(arr)


def first_local_min(arr):
    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            return i
    return len(arr)

def minimum_remover(data_x, data_y, prominence, distance, mode):
    if mode == 'absorption':
        peaks, _ = find_peaks(-data_y, prominence = prominence, distance = distance)
    elif mode == 'emission':
        peaks, _ = find_peaks(data_y, prominence = prominence, distance = distance)
    else:
        raise ValueError('Mode must be either "absorption" or "emission"')

    data_y_copy = data_y.copy()

    for i in range(len(peaks)):
        if peaks[i] < 15:
            arr1 = data_y[:peaks[i]][::-1]
        else:
            arr1 = data_y[peaks[i] - 15:peaks[i]][::-1]

        arr2 = data_y[peaks[i] + 1:peaks[i] + 16]

        if mode == 'absorption':
            max_1 = first_local_max(arr1)
            max_2 = first_local_max(arr2)

            #if the maximum is either missed in the window or the array ends before the maximum is found
            if max_1 == len(arr1):
                max_1 = len(arr1) - 1
            if max_2 == len(arr2):
                max_2 = len(arr2) - 1

            idx1 = 0 if peaks[i] - max_1 == 0 else peaks[i] - (max_1 + 1)
            idx2 = peaks[i] + (max_2 + 1)

        else:  # emission
            min_1 = first_local_min(arr1)
            min_2 = first_local_min(arr2)
            
            #if the minimum is either missed in the window or the array ends before the minimum is found
            if min_1 == len(arr1):
                min_1 = len(arr1) - 1
            if min_2 == len(arr2):
                min_2 = len(arr2) - 1

            idx1 = 0 if peaks[i] - min_1 == 0 else peaks[i] - (min_1 + 1)
            idx2 = peaks[i] + (min_2 + 1)

        # keeping the indices within array bounds regardless
        idx1 = max(0, idx1)
        idx2 = min(len(data_x), idx2)

        #line = line_maker(data_x[idx1], data_y[idx1], data_x[idx2 - 1], data_y[idx2 - 1])
        line = line_maker(data_x[idx1], data_y[idx1], data_x[idx2], data_y[idx2])
        data_y_copy[idx1:idx2] = line(data_x[idx1:idx2])

    return data_y_copy#, peaks

def cleaner(wl_arr, spectrum_arr, wl_1, wl_2):
    
    line = line_maker(wl_arr[find_nearest(wl_arr, wl_1)], spectrum_arr[find_nearest(wl_arr, wl_1)], wl_arr[find_nearest(wl_arr, wl_2)], spectrum_arr[find_nearest(wl_arr, wl_2)])
    
    spectrum_arr[find_nearest(wl_arr, wl_1):find_nearest(wl_arr, wl_2) + 1] = line(wl_arr[find_nearest(wl_arr, wl_1):find_nearest(wl_arr, wl_2) + 1])

    return spectrum_arr

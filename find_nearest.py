import numpy as np

def find_nearest(array, value):
    '''
    Find the element in an array nearest a given value
    
    Parameters
    ----------
    array: array
        array to be searched
    value: int, float
        value to be searched for in the array

    Returns
    -------
    idx: int
        index of the array element nearest the given value
    '''
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

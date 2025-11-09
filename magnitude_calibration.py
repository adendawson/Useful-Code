import numpy as np

def calibration_factor(raw_mags, reference_mags):
    avg_mag_difference = round(np.nanmean(raw_mags - reference_mags), 2)
    test_mags = np.arange(avg_mag_difference - 0.1, avg_mag_difference + 0.1, 0.01)
    small_delta_mag_count = []
    
    for i in test_mags:
        delta_mag = (raw_mags + abs(i)) - reference_mags
        delta_mag_range = (delta_mag > -0.1) & (delta_mag < 0.1)
        small_delta_mag_count.append(len(delta_mag_range))

    return abs(test_mags[np.argmax(small_delta_mag_count)])

threshold_low = 0.2
threshold_high = 0.9
output_dir = "output"
output_images_dir = "output_images"
output_filename = "output.csv"

show_image_before_save = True
show_histogram = False

filter = 'bessel' # possible values: 'bessel', 'sav-gol' or 'none
bessel_filter_cutoff = 160000
filter_window = 99
filter_poly_order = 1
show_raw = True

import pandas as pd
import copy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import glob
import os
import csv
import pathlib
from scipy.signal import savgol_filter, bessel, lfilter_zi, lfilter

def hyst(x, th_lo, th_hi, initial = False):
    hi = x >= th_hi
    lo_or_hi = (x <= th_lo) | hi
    ind = np.nonzero(lo_or_hi)[0]
    if not ind.size: # prevent index error if ind is empty
        return np.zeros_like(x, dtype=bool) | initial
    cnt = np.cumsum(lo_or_hi) # from 0 to len(x)
    return np.where(cnt, hi[ind[cnt-1]], initial)

def consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)




pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
pathlib.Path(output_images_dir).mkdir(parents=True, exist_ok=True) 

csvfile = open(os.path.join(output_dir, output_filename), 'w', newline='')
fieldnames = ['Filename',
              'Pulse Index',
              'Max Value [V]',
              'Integrated pulse area [V*us]',
              'Square pulse area [V*us]',
              'Pulse duration [us]',
              'Covariance']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

pulses_widths = []

def analyze_file(filename):
    print("Analyzing file: " + filename)

    plt.figure(0)
    plt.title(filename)
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')

    file_handle = open(filename, 'r')
    first = file_handle.readline()
    time = []
    volts_raw = []
    if first.find('_time(s)') != -1: # easyscope format
        columns = ["ch1_time(s)", "ch1_value(V)"]
        df = pd.read_csv(filename, usecols=columns)
        x = df.to_numpy()
        time, volts_raw = x.transpose()
        time = 1e6*time # in us
    else:
        # normal format
        columns = ["Time", "Volt"]
        df = pd.read_csv(filename, skiprows=1, usecols=columns)
        x = df.to_numpy()
        time, volts_raw = x.transpose()
        time = 1e6*time # in us

    minimal_value = volts_raw.min()
    volts_raw -= minimal_value
    
    dt = time[1] - time[0]

    if filter == 'sav-gol':
        if show_raw:
            plt.plot(1e-6*time, volts_raw, label="Raw")

        # Savitzky-Golay filter
        volts = savgol_filter(volts_raw, filter_window, filter_poly_order)
    elif filter == 'bessel':
        if show_raw:
            plt.plot(1e-6*time, volts_raw, label="Raw")

        # Bessel filter
        b, a = bessel(8, 2*bessel_filter_cutoff*dt*1e-6, 'low', analog=False, norm='phase')
        zi = lfilter_zi(b, a)
        volts, _ = lfilter(b, a, volts_raw, zi=zi*volts_raw[0])
    else:
        volts = volts_raw

    squared = hyst(np.array(volts), threshold_low, threshold_high)
    signal_blanked = squared * volts
    nonzero_indexes = squared.nonzero()[0]
    pulses = consecutive(nonzero_indexes)

    plt.plot(1e-6*time, volts)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    if len(nonzero_indexes) == 0:
        print("No pulses found!!")
        plt.text(1e-6*time[0], volts.max()/2, "No pulses found", fontsize=60, color="red")
        plt.show()
        return

    # if last pulse is not finished
    if nonzero_indexes[-1] == len(volts) - 1:
        pulses.pop()
    if nonzero_indexes[0] == 0:
        pulses.pop(0)

    # plt.plot(time, squared)
    # plt.plot(time, signal_blanked)

    print(f"Found {len(pulses)} pulses:")

    p_count = 1
    squared_polygon = []
    squared_polygon.append([time[0]*1e-6, 0])

    for p_c in range(len(pulses)):
        p = pulses[p_c]
        # print(f"Found pulse indexes {p[0]} to {p[-1]}")
        array_x = time[p[0]:p[-1]]
        array_y = signal_blanked[p[0]:p[-1]]
        array_raw = volts_raw[p[0]:p[-1]]

        impulse_max_value = np.max(array_y)
        impulse_time = dt*(p[-1] - p[0]) # us

        square_value = impulse_max_value * impulse_time
        sum = np.sum(array_y) * dt

        print(f'width: {impulse_time} us, max: {impulse_max_value} V, sq: {square_value} us*V, sum: {sum} us*V')

        rect = mpatches.Rectangle((1e-6*time[p[0]], 0), 1e-6*(time[p[-1]]-time[p[0]]), impulse_max_value, color='gray', fill=True)
        plt.gca().add_patch(rect)

        squared_polygon.append([1e-6*time[p[0]-1], 0])
        squared_polygon.append([1e-6*time[p[0]], impulse_max_value])
        squared_polygon.append([1e-6*time[p[-1]], impulse_max_value])
        squared_polygon.append([1e-6*time[p[-1]+1], 0])

        # covariance
        # find out covariance with respect  columns
        cov_mat = np.stack((array_y, array_raw), axis = 0)
        covariance = np.cov(cov_mat)[0][1]

        row = dict()
        row['Filename'] = filename
        row['Pulse Index'] = p_count
        row['Max Value [V]'] = impulse_max_value
        row['Integrated pulse area [V*us]'] = sum
        row['Square pulse area [V*us]'] = square_value
        row['Pulse duration [us]'] = impulse_time
        row['Covariance'] = covariance
        writer.writerow(row)
        csvfile.flush()

        if p_c != len(pulses) - 1:
            print(f"Next impulse in {time[pulses[p_c+1][0]] - time[pulses[p_c][-1]]}us")

        pulses_widths.append(impulse_time)

        p_count += 1

    squared_polygon.append([time[-1]*1e-6, 0])
    _poly = mpatches.Polygon(squared_polygon, closed=False, color='red', fill=None, lw=5)
    plt.gca().add_patch(_poly)

    plt.grid(True)
    plt.savefig(os.path.join(output_images_dir, filename + '.png'))
    
    if not show_image_before_save:
        plt.close()
    else:
        plt.show()

all_csvs = glob.glob('*.CSV')
for csv in all_csvs:
    analyze_file(csv)
csvfile.close()

if show_histogram:
    plt.figure(1)
    plt.hist(pulses_widths)
    plt.show()
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import copy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

plt.title('title name')
plt.xlabel('xAxis name')
plt.ylabel('yAxis name')
columns = ["Time", "Volt"]
df = pd.read_csv("SDS00004.CSV", skiprows=1, usecols=columns)
print("Contents in csv file:\n", df)
# plt.plot(df.Time, df.Volt)
# plt.grid(True)
# plt.show()

x = df.to_numpy()
time, volts = x.transpose()
time = 1e6*time # in us

dt = time[1] - time[0]

def hyst(x, th_lo, th_hi, initial = False):
    hi = x >= th_hi
    lo_or_hi = (x <= th_lo) | hi
    ind = np.nonzero(lo_or_hi)[0]
    if not ind.size: # prevent index error if ind is empty
        return np.zeros_like(x, dtype=bool) | initial
    cnt = np.cumsum(lo_or_hi) # from 0 to len(x)
    return np.where(cnt, hi[ind[cnt-1]], initial)

threshold_low = 0
threshold_high = 5
squared = hyst(np.array(volts), threshold_low, threshold_high)

signal_blanked = squared * volts
nonzero_indexes = squared.nonzero()[0]

print('squared:', squared)
print('nonzero:', nonzero_indexes)

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

pulses = consecutive(nonzero_indexes)
print('cons:', pulses)

# if last pulse is not finished
if nonzero_indexes[-1] == len(volts) - 1:
    pulses.pop()
if nonzero_indexes[0] == 0:
    pulses.pop(0)

# plt.plot(time, squared)
# plt.plot(time, signal_blanked)
plt.plot(time, volts)

for p in pulses:
    print(f"Found pulse indexes {p[0]} to {p[-1]}")
    array_x = time[p[0]:p[-1]]
    array_y = signal_blanked[p[0]:p[-1]]

    impulse_max_value = np.max(array_y)
    impulse_time = dt*(p[-1] - p[0]) # us

    square_value = impulse_max_value * impulse_time
    sum = np.sum(array_y) * dt

    print(f'time: {impulse_time} us, max: {impulse_max_value} V, sq: {square_value} us*V, sum: {sum} us*V')
    # plt.plot(array_x, array_y)

    rect = mpatches.Rectangle((time[p[0]], 0), time[p[-1]]-time[p[0]], impulse_max_value, color='gray', fill=True)
    plt.gca().add_patch(rect)




# plt.plot(squared)
# plt.plot(volts)
# plt.plot(signal_blanked)
plt.grid(True)
plt.show()

# wartości powinny być w jednostach podstawowych V, s itp. w notacji inzynierskiej np. s*10^-3 itp


from matplotlib import rc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fftfreq, fft

from scipy import signal

read_file = pd.read_csv(r'4.csv')
read_file.to_excel(r'new-signal.xlsx', index=None, header=True)

# Insert complete path to the excel file and index of the worksheet
df = pd.read_excel("new-signal.xlsx", sheet_name=0, )

# insert the name of the column as a string in brackets
list0 = list(df['ch1_time(s)'])
list1 = list(df['ch1_value(V)'])

# convert to arry
time = np.array(list0)
voltage = np.array(list1)

# Number of sample points
N = len(time)

# sample spacing
T = 100e-9

# samplerate
fs = 1/T

# decimage signal 50 times
q = 10
voltage = signal.decimate(voltage, q)
T = T*q
N = N//q
time = time[::q]


yf = fft(voltage)
xf = fftfreq(N, T)[:N//2]
print(max(xf))

# fig = plt.figure()
# ax = fig.add_subplot(111)
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]), '-r')
# plt.legend(['FFT'])
# fft end

f_signal = rfft(voltage)
W = fftfreq(voltage.size, d=time[1]-time[0])

cut_f_signal = f_signal.copy()
cut_f_signal[(W > 16000)] = 0  # filter all frequencies above 0.6

cut_signal = irfft(cut_f_signal)

# plot results
f, axarr = plt.subplots(1, 3, figsize=(20, 6))

# Orginal signal V(t)
axarr[0].plot(time, voltage, color='g')
axarr[0].set_xlabel("Time [s]")
axarr[0].set_ylabel("Voltage [V]")
axarr[0].grid()
axarr[0].legend(['Orginal signal'])
axarr[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


# FFT filter V(f)
axarr[1].plot(xf, 2.0/N * np.abs(yf[:N//2]), color='b')
#axarr[1].legend(('numpy fft * dt'), loc='upper right')
axarr[1].set_xlabel("f [Hz]")
axarr[1].set_ylabel("Amplitude [V]")
axarr[1].set_xlim(0, 20e4)  # range y axis value
axarr[1].grid()

# Bessel filter V(f)
b, a = signal.bessel(8, 16000, 'low', analog=True, norm='phase')
w, h = signal.freqs(b, a)
axarr[1].plot(w, np.abs(h), 'black')
axarr[1].legend(['FFT filter', 'Bessel filter - 8th order'])
axarr[1].ticklabel_format(axis="x", style="sci", scilimits=(3, 3))

b, a = signal.bessel(8, 2*16000/fs, 'low', analog=False, norm='phase')
w, h = signal.freqs(b, a)
axarr[1].plot(w, np.abs(h), 'red')
axarr[1].legend(['FFT filter', 'Bessel filter - 8th order'])
axarr[1].ticklabel_format(axis="x", style="sci", scilimits=(3, 3))

# FFT filter V(t)
# axarr[2].plot(time, cut_signal, color='m')
# axarr[2].set_xlabel("Time [s]")
# axarr[2].set_ylabel("Voltage [V]")
# axarr[2].grid()
# axarr[2].legend(['FFT filter'])
# axarr[2].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

# filter signal
# b, a = signal.butter(3, 0.05)
zi = signal.lfilter_zi(b, a)
voltage_filtered, _ = signal.lfilter(b, a, voltage, zi=zi*voltage[0])
voltage_filtered = signal.filtfilt(b, a, voltage)
axarr[2].plot(time, voltage_filtered, color='r')

# covariance
cov_mat = np.stack((voltage, voltage_filtered), axis=0)
print(np.cov(cov_mat))

plt.savefig('FFT-filter.png', bbox_inches='tight', transparent=True)
plt.savefig('FFT-filter.svg', bbox_inches='tight', transparent=True)

plt.show()

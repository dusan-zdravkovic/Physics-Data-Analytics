"""
Ion Channels Data Analysis
Author: Dusan Zdravkovic
Purpose: Analyze the ion channel data to understand the relationship between applied voltage
and open probability of ion channels.
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt


# Loading ion channel data
data = np.loadtxt("Keller1986Interpolated.csv", skiprows=1, unpack=True, delimiter=",")

time = data[0]
current_at_105mv = data[4]
current_at_135mv = data[1]
current_at_125mv = data[2]
current_at_115mv = data[3]
current_at_95mv = data[5]
current_at_85mv = data[6]
current_at_75mv = data[7]
current_at_65mv = data[8]
current_at_55mv = data[9]

# Plotting current passing through channel at -105mv vs. time
plt.figure(dpi=200, figsize=(8, 3))
plt.plot(time, current_at_105mv)
plt.xlabel("time (s)")
plt.ylabel("Current at -105mV")
plt.savefig("current_vs_time_105mv.png")


# Histogram of occurances in current value groups
plt.figure(dpi=200)
plt.hist(current_at_105mv, bins=100)  # 100 equal-width bins of current values
plt.xlabel("current (pA)")
plt.ylabel("number of occurances")  # number*dt = time spent at current value
plt.grid()
plt.savefig("histogram_current_105mv.png")

min_v = min(current_at_105mv)
max_v = max(current_at_105mv)
threshold = (min_v - max_v) / 2  # midpoint between max and min value

dI = (min_v - max_v) / 100  # delta I in mV
dt = time[1] - time[0]  # delta t in ms


# How many gates open (<threshold)
open_gates = []
for i in range(len(current_at_105mv)):
    if current_at_105mv[i] < threshold:
        open_gates.append(current_at_105mv[i])


open_gates = np.array(open_gates)
n_open = len(open_gates)
n_closed = len(current_at_105mv) - n_open

T_open = n_open * dt
T_closed = n_closed * dt

p_open105 = T_open / (T_open + T_closed)
print("Probability of being in the open state for 105mV is", p_open105 * 100, "%")


# Creating a function that calculates p_open for rest of data sets
def p_open_calculator(x):
    """Calculates p_open for a data set, input a 1D array"""

    min_v = min(x)
    max_v = max(x)
    threshold = (min_v + max_v) / 2

    open_gates = []
    for i in range(len(x)):
        if x[i] < threshold:
            open_gates.append(x[i])

    open_gates = np.array(open_gates)
    n_open = len(open_gates)
    n_closed = len(x) - n_open

    T_open = n_open * dt
    T_closed = n_closed * dt

    return T_open / (T_open + T_closed)


p_open135 = p_open_calculator(current_at_135mv)
p_open125 = p_open_calculator(current_at_125mv)
p_open115 = p_open_calculator(current_at_115mv)
p_open95 = p_open_calculator(current_at_95mv)
p_open85 = p_open_calculator(current_at_85mv)
p_open75 = p_open_calculator(current_at_75mv)
p_open65 = p_open_calculator(current_at_65mv)
p_open55 = p_open_calculator(current_at_55mv)


# Producing plot of open probability as a function of voltage
voltage_array = np.array([-135, -125, -115, -105, -95, -85, -75, -65, -55])
p_open_array = np.array(
    [
        p_open135,
        p_open125,
        p_open115,
        p_open105,
        p_open95,
        p_open85,
        p_open75,
        p_open65,
        p_open55,
    ]
)

plt.figure(dpi=200)
plt.plot(voltage_array, p_open_array, color="red", marker=".", linestyle="none")
plt.xlabel("applied voltage (mV)")
plt.ylabel("$p_{open}$")
plt.title("Open Probability as a Function of Voltage")
plt.grid()

# Curve fit for logistic function
from scipy.optimize import curve_fit


def logistic(x, l, c, k):
    return l / (1 + c * np.exp(-k * (x)))


popt, pcov = curve_fit(logistic, voltage_array, p_open_array, p0=[1, 9.2, 0.3])
voltage_array2 = np.linspace(-135, -55, num=100)
plt.plot(voltage_array2, logistic(voltage_array2, popt[0], popt[1], popt[2]))


# Error on data points using 25% changed threshold
# Rewriting function but with new threshold 25%
def p_open_new_calculator(x):
    """Calculates p_open for +25% threshold, input a 1D array"""

    min_v = min(x)
    max_v = max(x)
    threshold = min_v * 0.25

    open_gates = []
    for i in range(len(x)):
        if x[i] < threshold:
            open_gates.append(x[i])

    open_gates = np.array(open_gates)
    n_open = len(open_gates)
    n_closed = len(x) - n_open

    T_open = n_open * dt
    T_closed = n_closed * dt

    return T_open / (T_open + T_closed)


p_err135 = abs(p_open_new_calculator(current_at_135mv) - p_open135)
p_err125 = abs(p_open_new_calculator(current_at_125mv) - p_open125)
p_err115 = abs(p_open_new_calculator(current_at_115mv) - p_open115)
p_err105 = abs(p_open_new_calculator(current_at_105mv) - p_open105)
p_err95 = abs(p_open_new_calculator(current_at_95mv) - p_open95)
p_err85 = abs(p_open_new_calculator(current_at_85mv) - p_open85)
p_err75 = abs(p_open_new_calculator(current_at_75mv) - p_open75)
p_err65 = abs(p_open_new_calculator(current_at_65mv) - p_open65)
p_err55 = abs(p_open_new_calculator(current_at_55mv) - p_open55)

p_error_array = np.array(
    [
        p_err135,
        p_err125,
        p_err115,
        p_err105,
        p_err95,
        p_err85,
        p_err75,
        p_err65,
        p_err55,
    ]
)

plt.errorbar(
    voltage_array, p_open_array, yerr=p_error_array, capsize=2, linestyle="none"
)
plt.savefig("logistic_fit_probability.png")

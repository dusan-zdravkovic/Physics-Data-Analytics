""" 
Photoelectric effect 
Author: Dusan Zdravkovic
Purpose: To perform curve-fitting and analysis on photoelectric effect data to derive
key parameters and visualize the relationship between light frequency and stopping voltage
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


data = np.loadtxt(
    "../photoelectric-effect/photo_data.csv", skiprows=1, delimiter=",", unpack=True
)

# Defining data
wavelength = data[0]
delta_wavelength = data[1]
stopping_voltage = data[2]
other_voltage = data[3]
frequency = (3.0 * 10**8) / (wavelength * 10**-9)
y_error = stopping_voltage * 0.05


# Defining a model
def model_function(f, h, f_0):
    return (h / 1.6 * 10**-19) * (f - f_0)


def linear_function(x, a, b):
    return a * x + b


# Curve fit
popt, pcov = curve_fit(
    linear_function, frequency, stopping_voltage, sigma=y_error, absolute_sigma=True
)
pvar = np.diag(pcov)

# Plotting
plt.figure(figsize=(15, 8))
plt.plot(frequency, stopping_voltage, ".", label="data")
plt.plot(frequency, linear_function(frequency, popt[0], popt[1]), label="curve_fit")
plt.errorbar(frequency, stopping_voltage, yerr=y_error, capsize=5, linestyle="none")

plt.title("Stopping Voltage vs. Frequency")
plt.xlabel("Frequency(Hz)")
plt.ylabel("Stopping Voltage(V)")
plt.legend()
plt.grid()
plt.savefig("stopping_voltage_vs_frequency.png")

# Finding Quantities
calculated_plancks = popt[0] * 1.6 * 10**-19
true_plancks = 6.626 * 10**-34
error_plancks = np.sqrt(pvar[0]) * 1.6 * 10**-19


calculated_threshold = -popt[1] * (1.6 * 10**-19 / true_plancks)
error_threshold = (np.sqrt(pvar[1])) * (1.6 * 10**-19 / true_plancks)


calculated_work_function = calculated_plancks * calculated_threshold
error_work_function = calculated_work_function * np.sqrt(
    (error_plancks / calculated_plancks) ** 2
    + (error_threshold / calculated_threshold) ** 2
)

print(
    "calculated plancks is",
    calculated_plancks,
    "+-",
    error_plancks,
    ",threshold frequency is",
    calculated_threshold,
    "+-",
    error_threshold,
    "calculated work function is",
    calculated_work_function,
    "+-",
    error_work_function,
)


# Chi-squared
def chi_squared(prediction_array, data_array, error_array, data_size, parameter_size):
    chi_square = (1 / (data_size - parameter_size)) * np.sum(
        (data_array - prediction_array) ** 2 / error_array**2
    )
    return chi_square


chi_square1 = chi_squared(
    linear_function(frequency, popt[0], popt[1]),
    stopping_voltage,
    y_error,
    stopping_voltage.size,
    2,
)
print("chi_squared is =", chi_square1)

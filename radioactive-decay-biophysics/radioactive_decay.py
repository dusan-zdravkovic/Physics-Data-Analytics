"""
Radioactive Decay
Author: Dusan Zdravkovic
Due: November 18 2022
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Uploading Data
decay_data = np.loadtxt("decay.txt", skiprows=2)
background_data = np.loadtxt("background.txt", skiprows=2)

# Setting Variables
tsteps = decay_data[:, 0]
Tsim = 20
dt = Tsim / float(len(tsteps))
t = dt * tsteps

It = decay_data[:, 1]
Ib = background_data[:, 1]
Nb = np.mean(Ib)
Nb_array = np.ones_like(It) * Nb
Nf = It - Nb_array
rate = Nf / dt
uncertainty = np.sqrt(It + Ib)

# Ensuring rate is positive for logarithm calculation
rate[rate <= 0] = np.nan


# Linear Regression
def f(t, a, b):
    return b * np.exp(-a * t)


popt, pcov = curve_fit(f, t, rate, p0=[1, max(rate)])  # Initial guess for a, b

# Plotting
plt.figure(1, dpi=200)
plt.plot(t, f(t, *popt), color="red", label="Curve Fit")
plt.plot(t, rate, ".", label="Data")
plt.xlabel("Time (minutes)")
plt.ylabel("Count Rate")
plt.title("Count Rate vs. Time")
plt.errorbar(t, rate, yerr=uncertainty, linestyle="none", capsize=2)
plt.grid()
plt.legend()
plt.savefig("count_rate_vs_time.png")

# Half-Life Calculation
a = popt[0]
half_life = np.log(2) / a
print("The half life is", half_life, "minutes")

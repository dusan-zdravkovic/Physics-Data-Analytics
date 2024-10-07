""" 
Thermal Motion Analysis Script

Author: Dusan Zdravkovic

Purpose: Analyzing step sizes and diffusion constants from thermal motion experiments. 
         Calculates probability densities and performs curve fitting to model data.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
scale_factor = 0.1155
k_Boltzmann = 1.38 * 10 ** (-23)  # Boltzmann's constant in J/K


# Function to load and scale data
def load_and_scale_data(file_path, scale):
    data = np.loadtxt(file_path, skiprows=2, unpack=True)
    return data * scale


# Load and process data
num_experiments = 10
base_file_path = "../thermal-motion/object_data/Exp "
x_data = []
y_data = []
delta_d = []

for i in range(1, num_experiments + 1):
    file_path = f"{base_file_path}{i}.txt"
    x, y = load_and_scale_data(file_path, scale_factor)
    x_data.append(x)
    y_data.append(y)
    # Calculate delta_d for each dataset
    deltas = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    delta_d.append(deltas)

# Combine all step sizes
all_step_sizes = np.concatenate(delta_d)

# Probability density and step size data (from your existing script)
bin_counts, bin_edges = np.histogram(all_step_sizes, bins=70, density=True)
step_size = 0.5 * (bin_edges[1:] + bin_edges[:-1])
probability_density = bin_counts


# Curve fitting function
def model_function(step_size, diffusion_constant):
    return (step_size / (2 * diffusion_constant * 0.5)) * np.exp(
        -(step_size**2) / (4 * diffusion_constant * 0.5)
    )


popt, pcov = curve_fit(model_function, step_size, probability_density)
diffusion_constant1 = popt[0]

k1 = ((diffusion_constant1 / 10**12) * (6 * np.pi * 0.1 * (0.95 / 10**6))) / 296.5
print(k1)

# Maximum likelihood
step_size_squared = step_size**2
diffusion_constant2 = (np.sum(step_size_squared)) * (1 / 142)
k2 = ((diffusion_constant2 / 10**12) * (6 * np.pi * 0.001 * (0.95 / 10**6))) / 296.5
print(k2)

# Graph: Step Size Histogram
plt.figure(figsize=(15, 8), dpi=200)
plt.hist(all_step_sizes, density=True, bins=70)
plt.title("Step Size Histogram")
plt.xlabel("Step Size ($\mu$m)")
plt.ylabel("Probability Density")
plt.plot(step_size, model_function(step_size, popt[0]))
plt.plot(
    step_size,
    model_function(step_size, diffusion_constant1),
    label="Rayleigh Distribution 1",
)
plt.plot(
    step_size,
    model_function(step_size, diffusion_constant2),
    label="Rayleigh Distribution 2",
)
plt.legend()
plt.grid()
plt.savefig("step_size_hist.png")

# Calculate Percent Differences
percentage_difference_k1 = (np.abs(k1 - k_Boltzmann) / k_Boltzmann) * 100
percentage_difference_k2 = (np.abs(k2 - k_Boltzmann) / k_Boltzmann) * 100
print(percentage_difference_k1)
print(percentage_difference_k2)

# Mean Squared Distance vs. Time
mean_square_distance = []
for i in range(len(x_data[0])):
    mean_square_distance.append(
        np.sqrt((x_data[0][i] - x_data[0][0]) ** 2 + (y_data[0][i] - y_data[0][0]) ** 2)
    )

time_list = (np.arange(len(x_data[0]))) * 0.5


def linear_function(time, diffusion_constant):
    return diffusion_constant * time * 4


r_error = np.array(mean_square_distance) * 0.1
t_error = time_list * 0.03

popt1, pcov1 = curve_fit(linear_function, time_list, mean_square_distance)

plt.figure(figsize=(15, 8), dpi=200)
plt.title("Mean Squared Distance vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Mean Squared Distance")
plt.plot(time_list, mean_square_distance, ".", label="Data")
plt.plot(time_list, linear_function(time_list, popt1[0]), label="Curve Fit")
plt.legend()
plt.grid()
plt.savefig("mean_squared_distance.png")

diffusion_constant3 = popt1[0]
k3 = ((diffusion_constant3 / 10**12) * (6 * np.pi * 0.001 * (0.95 / 10**6))) / 296.5
print(k3)

percentage_difference_k3 = (np.abs(k3 - k_Boltzmann) / k_Boltzmann) * 100
print(percentage_difference_k3)

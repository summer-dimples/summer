import numpy as np
import matplotlib.pyplot as plt

# Define parameters
T = 1.5  # Response delay
lambda_values = [1/(1.5*np.e), 0.8/1.5, np.pi/(2*1.5), 1.65/1.5]  # Lambda values corresponding to different C values
labels = ["Non-oscillatory decay (C=1/e)", 
          "Oscillatory decay (C=0.8)", 
          "Undamped oscillations (C=π/2)", 
          "Growing oscillations (C=1.65)"]

# Initial conditions
num_vehicles = 2  # Number of vehicles in the platoon
time_steps = 2000  # Number of time steps
dt = 0.01  # Time step size
t = np.arange(0, time_steps * dt, dt)  # Time array
delay_steps = int(T / dt)  # Convert delay time to discrete steps

# Define lead vehicle's acceleration
def lead_vehicle_acceleration(t):
    return -2 * np.cos(np.pi * t) * (t <= 1)

# Compute velocity from acceleration
lead_velocity = np.zeros_like(t)
lead_velocity[0] = 20  # Initial velocity of lead vehicle

for i in range(1, len(t)):
    lead_velocity[i] = lead_velocity[i-1] + dt * lead_vehicle_acceleration(t[i-1])

# Loop through each lambda to create separate plots
for lambda_val, label in zip(lambda_values, labels):
    C = lambda_val * T  # Compute C

    # Initialise velocity arrays
    v = np.full((num_vehicles, len(t)), 20.0)  # All vehicles start at 20 m/s
    v[0, :] = lead_velocity  # Lead vehicle follows prescribed motion

    # Initial Headway of 40m Between Each Vehicle
    x = np.zeros((num_vehicles, len(t)))  # Initialize position array
    for n in range(1, num_vehicles):
        x[n, 0] = x[n-1, 0] - 40  # **Ensuring 40m headway initially**

    # Time-stepping solution with delay
    for i in range(1, len(t) - 1):
        if i >= delay_steps:
            for n in range(1, num_vehicles):  # Ignore lead vehicle
                delayed_index = max(i - delay_steps, 0)  # Ensure index does not go negative
                v[n, i+1] = v[n, i] + dt * lambda_val * (v[n-1, delayed_index] - v[n, delayed_index])

    # Compute acceleration (derivative of velocity)
    a = np.zeros_like(v)
    a[:, 1:] = np.diff(v, axis=1) / dt  # Numerical differentiation

    # Update positions via integration of velocity
    for i in range(1, len(t)):
        x[:, i] = x[:, i-1] + v[:, i] * dt  # Update positions

    # Compute headways
    S = np.zeros((num_vehicles - 1, len(t)))  # Headway vector (one less than num_vehicles)
    for n in range(1, num_vehicles):
        S[n-1, :] = x[n-1, :] - x[n, :]

    # Generate separate figures for each plot type
    # Velocity plot
    plt.figure(figsize=(8, 6))
    for n in range(num_vehicles):
        plt.plot(t, v[n, :], label=f"Vehicle {n}", linewidth=2.5)
 
    plt.xlabel("Time (s)", fontsize=24)
    plt.ylabel("Velocity (m/s)", fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22, loc="upper right")
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Headway plot
    plt.figure(figsize=(8, 6))
    for n in range(num_vehicles - 1):
        plt.plot(t, S[n, :], label=f"Vehicle {n+1}", linewidth=2.5)
   
    plt.xlabel("Time (s)", fontsize=24)
    plt.ylabel("Headway (m)", fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22, loc="upper right")
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Acceleration plot
    plt.figure(figsize=(8, 6))
    for n in range(num_vehicles):
        plt.plot(t, a[n, :], label=f"Vehicle {n}", linewidth=2.5)
  
    plt.xlabel("Time (s)", fontsize=24)
    plt.ylabel("Acceleration (m/s²)", fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22, loc="upper right")
    plt.grid()
    plt.tight_layout()
    plt.show()

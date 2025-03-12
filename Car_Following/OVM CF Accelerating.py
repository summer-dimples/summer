import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters and Time Settings
# -----------------------------
num_vehicles = 3  # Number of vehicles (lead + followers)
time_steps = 2000  # Number of time steps
dt = 0.01  # Time step size (s)
t = np.arange(0, time_steps * dt, dt)  # Time array
h0 = 1000  # Initial headway for the lead vehicle

track_length = num_vehicles * 40  # Circuit length for periodic conditions

# List of sensitivity values 'a' for OVM and their labels
a_values = [0.5]
labels = ["(a = 0.5)"]

# ----------------------------------------------
# Define the Optimal Velocity (Bando) Function
# ----------------------------------------------
def OVF(s):
    
    return 16.8 * (np.tanh(0.086 * (s - 25)) + 0.913)

# -------------------------------
# Loop over different sensitivity values
# -------------------------------
for a_val, label in zip(a_values, labels):
    
    # Initialize velocity arrays
    v = np.full((num_vehicles, len(t)), 0.0)  # Steady state velocity for 40m headway

    # Initialize positions
    x = np.zeros((num_vehicles, len(t)))
    x[0, 0] = 0.0
    for n in range(1, num_vehicles):
        x[n, 0] = x[n-1, 0] - 10.0  # Ensuring 40m headway initially

    # Initialize headway storage (now a 2D array!)
    S = np.zeros((num_vehicles, len(t)))

    # -----------------------
    # Time-stepping (Euler method)
    # -----------------------
    for i in range(0, len(t)-1):
        # Compute and store headways at each time step
        for n in range(num_vehicles):
            if n == 0:  # Lead vehicle follows last vehicle (Periodic BC)
                S[n, i] = h0
            else:
                S[n, i] = x[n-1, i] - x[n, i]

        # Update velocity and position for all vehicles
        for n in range(num_vehicles):
            v[n, i+1] = v[n, i] + dt * a_val * (OVF(S[n, i]) - v[n, i])
            x[n, i+1] = x[n, i] + dt * v[n, i]

    # ---------------------------
    # Compute acceleration (for plotting)
    # ---------------------------
    acceleration = np.zeros_like(v)
    acceleration[:, 1:] = np.diff(v, axis=1) / dt

    # ---------------------------
    # Plotting Results
    # ---------------------------

    # Velocity vs Time
    plt.figure(figsize=(8, 6))
    for n in range(num_vehicles):
        plt.plot(t, v[n, :], label=f"Vehicle {n}", linewidth=2.5)
    plt.xlabel("Time (s)", fontsize=22)
    plt.ylabel("Velocity (m/s)", fontsize=22)
    plt.xlim(0, t[-1])    
    plt.xticks(fontsize=20)
    plt.ylim(0, None)  # Ensures y-axis starts at 0
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20, loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Headway vs Time (Skip the last point)
    plt.figure(figsize=(8, 6))
    for n in range(1, num_vehicles):  # Only plot for following vehicles
        plt.plot(t[:-1], S[n, :-1], label=f"Vehicle {n}", linewidth=2.5)  # Exclude last point
    plt.xlabel("Time (s)", fontsize=22)
    plt.ylabel("Headway (m)", fontsize=22)
    plt.xlim(0, t[-1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20, loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



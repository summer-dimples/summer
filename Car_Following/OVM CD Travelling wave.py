import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define Parameters
a = 2  # Sensitivity
Vmax = 30.0  # Maximum velocity
h_0 = 30.0  # Initial headway (based on paper)
N = 50  # Number of vehicles

# Define the optimal velocity function (same as paper)
def V_bando(h):
    return Vmax / 1.913 * (np.tanh(0.086 * (h - 25)) + 0.913)

# Define the Bando model equations
def bando_model(t, y, N, a):
    x = y[:N]  # Positions
    v = y[N:]  # Velocities

    h = x[:-1] - x[1:]  # Compute headway using h_n = x_{n-1} - x_n
    dvdt = np.zeros(N)
    dvdt[1:] = a * (V_bando(h) - v[1:])  # Velocity update (excluding first vehicle)
    dxdt = v  # Position update

    return np.concatenate((dxdt, dvdt))

# Set x_0 = 0, x_1 = -h_0, x_2 = -2*h_0, ...
x0 = -np.arange(N) * h_0  # Set initial positions

# Introduce disc. perturbation
h_jump = 28  # Final headway after the jump
jump_location = N // 2  # Place the shock at the middle

x0[jump_location:] += (h_jump - h_0) * np.arange(0, N - jump_location)

# Initial velocities
v0 = np.ones(N) * 25.0  
y0 = np.concatenate((x0, v0))

# Solve 
t_span = (0, 120)  # Simulate up to 120s
t_eval = np.linspace(0, 120, 500)
sol_bando = solve_ivp(bando_model, t_span, y0, args=(N, a), t_eval=t_eval, method='RK45')

# Solutions are
positions_bando = sol_bando.y[:N, :]
velocities_bando = sol_bando.y[N:, :]

# Compute headways
headways_bando = positions_bando[:-1, :] - positions_bando[1:, :]
displacements_bando = positions_bando[:-1, :] - positions_bando[0, :]

# Define times at which to plot data
t_targets = [0, 6, 12, 18]  # Time instants for comparison
colors = ['black', 'blue', 'red', 'green']  # Different colors for each time

# Plot headway vs. displacement at those times
plt.figure(figsize=(8, 6))

for i, t_target in enumerate(t_targets):
    t_idx = np.argmin(np.abs(t_eval - t_target))
    
    # Extract headways and displacements at t_target (excluding h_0)
    headways_final = headways_bando[:, t_idx]
    displacements_final = displacements_bando[:, t_idx]

    num_cars_to_plot = min(N-1, len(headways_final))  # Ensure valid array slicing

    headways_final_subset = headways_final[:num_cars_to_plot]
    displacements_final_subset = displacements_final[:num_cars_to_plot]

    # Plot with different colors for each time
    plt.scatter(displacements_final_subset, headways_final_subset, color=colors[i], label=f"Bando Model (t={t_target}s)")

plt.xlabel("Displacement (m)", fontsize=16)
plt.ylabel("Headway (m)", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.legend()
plt.show()

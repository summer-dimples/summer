import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # For manually adding legend entries
from scipy.optimize import fsolve

# Define the equations
def equation_16(alpha, beta):
    return alpha * np.sin(beta) + beta * np.cos(beta)

def equation_18(alpha, beta, C):
    return C * np.cos(beta) + alpha * np.exp(alpha)

# Define the range of alpha and beta
alpha_vals = np.linspace(-1.5, 1.5, 500)  # Range of alpha values
beta_vals = np.linspace(-4, 4, 500)  # Range of beta values

# Create a meshgrid for plotting
A, B = np.meshgrid(alpha_vals, beta_vals)

# Compute values for Equation (16) - This is independent of C
Z_16 = equation_16(A, B)

# Define three values of C for comparison
C_values = [0.3, np.pi /2, 5]  # Example values: C1, C2, C3
colors = ['r', 'g', 'm']  # Different colors for different C values

# Create the plot
plt.figure(figsize=(10, 6))

# Plot Equation (16) (Same for all C values)
contour_16 = plt.contour(A, B, Z_16, levels=[0], colors='b', linewidths=2)

# Loop through the C values and plot Equation (18) for each
contour_18_list = []
for i, C in enumerate(C_values):
    Z_18 = equation_18(A, B, C)
    contour = plt.contour(A, B, Z_18, levels=[0], colors=colors[i], linewidths=2)
    contour_18_list.append(contour)

# Labels and formatting
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)

# Manually create legend handles
legend_elements = [
    mlines.Line2D([], [], color='b', linewidth=2, label=r'Eq. 3.12: $\alpha \sin\beta + \beta \cos\beta = 0$'),
    mlines.Line2D([], [], color='r', linewidth=2, label=f'Eq. 3.14: $C = {C_values[0]}$'),
    mlines.Line2D([], [], color='g', linewidth=2, label=r'Eq. 3.14: $C = \frac{\pi}{2}$'),
    mlines.Line2D([], [], color='m', linewidth=2, label=f'Eq. 3.14: $C = {C_values[2]}$')
]

# Add the legend

plt.savefig("my_plot.png", dpi=300, bbox_inches="tight")
# Show the plot
plt.show()



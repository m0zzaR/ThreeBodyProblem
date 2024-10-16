import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Set constants
G = 1
m1 = 7
m2 = 8
m3 = 3

# Define the three-body equations
def threebody(t, U):
    # Unpack the vector U into velocity and position components
    v1 = U[0:3]
    v2 = U[3:6]
    v3 = U[6:9]
    r1 = U[9:12]
    r2 = U[12:15]
    r3 = U[15:18]
    
    # Calculate the accelerations using Newton's law of gravitation
    v1dot = G * ((m2 / np.linalg.norm(r2 - r1)**3) * (r2 - r1) + (m3 / np.linalg.norm(r3 - r1)**3) * (r3 - r1))
    v2dot = G * ((m1 / np.linalg.norm(r1 - r2)**3) * (r1 - r2) + (m3 / np.linalg.norm(r3 - r2)**3) * (r3 - r2))
    v3dot = G * ((m1 / np.linalg.norm(r1 - r3)**3) * (r1 - r3) + (m2 / np.linalg.norm(r2 - r3)**3) * (r2 - r3))
    
    # The rate of change of position is just the velocity
    r1dot = v1
    r2dot = v2
    r3dot = v3
    
    # Return the concatenated velocity and position derivatives
    return np.concatenate([v1dot, v2dot, v3dot, r1dot, r2dot, r3dot])

# Initial conditions: velocities and positions
U0 = [3/m1, 0, 0, -3/m2, 0, 0, 0, 0, 0, 1, 1, 0, 2, 3, 4, 5, 1, 2]

# Time points
t = np.linspace(0, 10, 1000)

# Solve the system of equations using solve_ivp (similar to MATLAB's ode15s)
solution = solve_ivp(threebody, [t[0], t[-1]], U0, t_eval=t, method='LSODA')

# Extract the positions and velocities from the solution
v1 = solution.y[0:3, :].T
v2 = solution.y[3:6, :].T
v3 = solution.y[6:9, :].T
r1 = solution.y[9:12, :].T
r2 = solution.y[12:15, :].T
r3 = solution.y[15:18, :].T

# Set up the plot limits based on the results
if m3 != 0:
    xmin = min([np.min(r1[:, 0]), np.min(r2[:, 0]), np.min(r3[:, 0])])
    xmax = max([np.max(r1[:, 0]), np.max(r2[:, 0]), np.max(r3[:, 0])])
    ymin = min([np.min(r1[:, 1]), np.min(r2[:, 1]), np.min(r3[:, 1])])
    ymax = max([np.max(r1[:, 1]), np.max(r2[:, 1]), np.max(r3[:, 1])])
    zmin = min([np.min(r1[:, 2]), np.min(r2[:, 2]), np.min(r3[:, 2])])
    zmax = max([np.max(r1[:, 2]), np.max(r2[:, 2]), np.max(r3[:, 2])])
else:
    xmin = min([np.min(r1[:, 0]), np.min(r2[:, 0])])
    xmax = max([np.max(r1[:, 0]), np.max(r2[:, 0])])
    ymin = min([np.min(r1[:, 1]), np.min(r2[:, 1])])
    ymax = max([np.max(r1[:, 1]), np.max(r2[:, 1])])
    zmin = min([np.min(r1[:, 2]), np.min(r2[:, 2])])
    zmax = max([np.max(r1[:, 2]), np.max(r2[:, 2])])

# Plot the trajectories of the three bodies
for i in range(len(t)):
    plt.figure(figsize=(8, 6))
    plt.plot(r1[i, 0], r1[i, 1], r1[i, 2], 'ro', markersize=8)
    plt.plot(r1[:i, 0], r1[:i, 1], r1[:i, 2], 'r')
    
    plt.plot(r2[i, 0], r2[i, 1], r2[i, 2], 'go', markersize=8)
    plt.plot(r2[:i, 0], r2[:i, 1], r2[:i, 2], 'g')
    
    if m3 != 0:
        plt.plot(r3[i, 0], r3[i, 1], r3[i, 2], 'bo', markersize=8)
        plt.plot(r3[:i, 0], r3[:i, 1], r3[:i, 2], 'b')
    
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.grid(True)
    plt.pause(0.001)
    plt.clf()

plt.show()

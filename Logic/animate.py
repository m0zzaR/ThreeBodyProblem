import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import Celestial

# Constants
G = 6.67430e-11  # Gravitational constant
dt = 1000        # Time step in seconds

# Create celestial bodies
body1 = Celestial.Body(
    mass=5.972e24,                 # Mass of Earth in kg
    position=[0.0, 0.0],           # Starting at the origin
    velocity=[0.0, 0.0]            # Initially stationary
)

body2 = Celestial.Body(
    mass=7.348e22,                 # Mass of the Moon in kg
    position=[384400000.0, 0.0],   # Approximately 384,400 km from Earth
    velocity=[0.0, 0.0]         # 1022 in y Approximate orbital velocity of the Moon in m/s
)

# Simulation parameters
num_steps = 500  # Number of simulation steps

# Lists to store positions for animation
positions1 = []
positions2 = []

# Run the simulation and store positions
for _ in range(num_steps):
    # Store current positions
    positions1.append(body1.position.copy())
    positions2.append(body2.position.copy())
    # Perform a simulation step
    Celestial.step(dt, body1, body2, G)

positions1 = np.array(positions1)
positions2 = np.array(positions2)

# Adjust positions to be relative to Earth's position
positions1_relative = positions1 - positions1  # Earth's position relative to itself is zero
positions2_relative = positions2 - positions1  # Moon's position relative to Earth

# Set up the figure and axis for animation
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.grid(True)

# Set axis limits centered on Earth
distance = 4e8  # 400 million meters
ax.set_xlim(-distance, distance)
ax.set_ylim(-distance, distance)

# Initialize the lines representing the two objects
line1, = ax.plot([], [], 'ro', markersize=8, label='Earth')
line2, = ax.plot([], [], 'bo', markersize=5, label='Moon')

# Initialize the trails
trail1, = ax.plot([], [], 'r-', linewidth=0.5)
trail2, = ax.plot([], [], 'b-', linewidth=0.5)

ax.legend()

# Initialization function for the animation
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    trail1.set_data([], [])
    trail2.set_data([], [])
    return line1, line2, trail1, trail2

# Animation function which updates figure data
def animate(i):
    # Earth is at the center
    line1.set_data([0], [0])
    # Moon's position relative to Earth
    line2.set_data([positions2_relative[i, 0]], [positions2_relative[i, 1]])
    # Update trails
    trail1.set_data([0], [0])  # Earth doesn't move
    trail2.set_data(positions2_relative[:i+1, 0], positions2_relative[:i+1, 1])
    return line1, line2, trail1, trail2

# Create the animation
anim = FuncAnimation(fig, animate, init_func=init,
                     frames=num_steps, interval=20, blit=True)

# Display the animation
plt.show()
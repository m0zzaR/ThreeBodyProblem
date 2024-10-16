import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from collections import deque
import Celestial

# Constants
G = 1  # Gravitational constant
dt = 3600        # Time step in seconds (1 hour)

# Create celestial bodies
body1 = Celestial.Body(
    mass=1,                 # Mass of Earth in kg
    position=[0.0, 0.0],           # Starting at the origin
    velocity=[0.0, 0.0]            # Initially stationary
)

body2 = Celestial.Body(
    mass=1,                 # Mass of the Moon in kg
    position=[10, 0.0],     # Approximately 384,400 km from Earth in meters
    velocity=[0.0, 0.0]          # Approximate orbital velocity of the Moon in m/s
)

# Maximum trail length to prevent memory issues
max_trail_length = 500  # Adjust as needed

# Initialize deques to store positions for trails, including initial positions
positions1 = deque([body1.position.copy()], maxlen=max_trail_length)
positions2 = deque([body2.position.copy()], maxlen=max_trail_length)

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
    # Perform simulation step
    Celestial.step(dt, body1, body2, G)
    
    # Store positions for trails
    positions1.append(body1.position.copy())
    positions2.append(body2.position.copy())
    
    # Adjust positions to be relative to Earth's position
    position1_rel = body1.position - body1.position  # Earth at center
    position2_rel = body2.position - body1.position  # Moon relative to Earth

    # Update object positions
    line1.set_data([0], [0])
    line2.set_data([position2_rel[0]], [position2_rel[1]])
    
    # Update trails
    trail1.set_data([0], [0])  # Earth doesn't move
    trail2.set_data(
        [pos[0] - body1.position[0] for pos in positions2],
        [pos[1] - body1.position[1] for pos in positions2]
    )
    
    return line1, line2, trail1, trail2

# Create the animation
anim = FuncAnimation(fig, animate, init_func=init, frames=1000, interval=20, blit=True)

# Set up the writer
Writer = FFMpegWriter
writer = Writer(fps=30, metadata=dict(artist='m0zzaR'), bitrate=1800)

# Save the animation
anim.save('orbit_animation.mp4', writer=writer)

# Display the animation (optional)
plt.show()

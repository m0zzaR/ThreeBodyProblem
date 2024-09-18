import numpy as np
import math

# Define constants
G = 1  # Gravitational constant (normalized for simplicity)

timesteps = np.linspace(1, 15, 5)

# Initial velocities (as vectors)
v1 = np.array([0.0, 0.0])
v2 = np.array([0.0, 0.0])

# Initial positions (as vectors)
r1 = np.array([0, 0.0])
r2 = np.array([10, 0.0])

# Define masses
m1 = 10.0  # Mass of object 1
m2 = 5.0  # Mass of object 2

# Function to calculate magnitude of a 2D vector
def mag(x):
    return np.sqrt(x[0] ** 2 + x[1] ** 2)

# Example Step function (as used in previous simulations)
def step(dt):
    global v1, v2, r1, r2
    # Calculate the distance between the two objects
    dx = r2[0] - r1[0]
    dy = r2[1] - r1[1]
    r = np.sqrt(dx**2 + dy**2)
    
    # Calculate gravitational force between the two objects
    F = G * m1 * m2 / r**2  # Force magnitude
    Fx = F * dx / r  # Force in x direction
    Fy = F * dy / r  # Force in y direction
    
    # Calculate accelerations
    a1 = np.array([Fx / m1, Fy / m1])  # Acceleration on object 1
    a2 = -np.array([Fx / m2, Fy / m2])  # Acceleration on object 2 (opposite direction)
    
    # Update velocities
    v1 = v1 + a1 * dt
    v2 = v2 + a2 * dt
    
    # Update positions
    r1 = r1 + v1 * dt + 1/2 * a1 * dt ** 2
    r2 = r2 + v2 * dt + 1/2 * a2 * dt ** 2
    
    return r1, r2, v1, v2

# Main simulation loop
for step_index in range(1, len(timesteps)):
    dt = timesteps[step_index] - timesteps[step_index - 1]
    
    # Update positions and velocities
    r1, r2, v1, v2 = step(dt)
    
    # Calculate distance between the two objects after the step
    dist = mag(r2 - r1)
    
    # Output the distance
    print(f"Step {step_index}: Distance between objects: {dist}")

    # Print the position and velocity vectors for both objects
    print(f"Object 1 -> Position: {r1}, Velocity: {v1}")
    print(f"Object 2 -> Position: {r2}, Velocity: {v2}\n")


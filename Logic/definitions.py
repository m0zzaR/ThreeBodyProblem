import numpy as np
import matplotlib.pyplot as plt
import math

# Define constants
G = 6.67e-11

timesteps = np.linspace(1, 3000, 1000)
# Lists to store velocity data for plotting
# Lists to store acceleration and position data for both objects
a1_magnitudes = []
a2_magnitudes = []
r1_magnitudes = []
r2_magnitudes = []
v1_magnitudes = []
v2_magnitudes = []
time_data = []

# Initial velocities (as vectors)
v1 = np.array([0, 0])
v2 = np.array([0, 0])

# Initial positions (as vectors)
r1 = np.array([0, 0.0])
r2 = np.array([93e7, 0.0]) #Dist from Earth to Sun

# Define masses
m1 = 2e30  # Mass of Sun (kg)
m2 = 6e24  # Mass of Earth (kg)

# Function to calculate magnitude of a 2D vector
def mag(x):
    return np.sqrt(x[0] ** 2 + x[1] ** 2)

# Example Step function 
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

    # Store acceleration magnitudes
    a1_magnitudes.append(mag(a1))
    a2_magnitudes.append(mag(a2))
    
    # Update velocities
    v1 = v1 + a1 * dt
    v2 = v2 + a2 * dt

    # Store magnitudes of velocities for both objects
    v1_magnitudes.append(mag(v1))
    v2_magnitudes.append(mag(v2))
    
    # Update positions
    r1 = r1 + v1 * dt + 1/2 * a1 * dt ** 2
    r2 = r2 + v2 * dt + 1/2 * a2 * dt ** 2

    # Store position magnitudes
    r1_magnitudes.append(mag(r1))
    r2_magnitudes.append(mag(r2))


    
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

    time_data.append(timesteps[step_index])

    # Print the position and velocity vectors for both objects
    print(f"Object 1 -> Position: {r1}, Velocity: {v1}")
    print(f"Object 2 -> Position: {r2}, Velocity: {v2}\n")


# Plot velocity, acceleration, and position in a single figure with 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Plot position vs time
axs[0].plot(time_data, r1_magnitudes, label="Object 1 Position (m1 = 20)", color='blue')
axs[0].plot(time_data, r2_magnitudes, label="Object 2 Position (m2 = 5)", color='orange')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Position')
axs[0].legend()
axs[0].grid(True)

# Plot velocity vs time
axs[1].plot(time_data, v1_magnitudes, label="Object 1 Velocity (m1 = 20)", color='blue')
axs[1].plot(time_data, v2_magnitudes, label="Object 2 Velocity (m2 = 5)", color='orange')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Velocity')
axs[1].legend()
axs[1].grid(True)

# Plot acceleration vs time
axs[2].plot(time_data, a1_magnitudes, label="Object 1 Acceleration (m1 = 20)", color='blue')
axs[2].plot(time_data, a2_magnitudes, label="Object 2 Acceleration (m2 = 5)", color='orange')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Acceleration')
axs[2].legend()
axs[2].grid(True)


# Adjust layout to make room for titles
plt.tight_layout()
plt.show()

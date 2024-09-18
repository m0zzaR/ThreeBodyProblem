import numpy as np
import math

# Define constants
#G = 6.67430e-11
G = 1

timesteps = np.linspace(1, 50, 100)


# Initial velocities (as vectors)
v1 = np.array([0.0, 0.0])
v2 = np.array([0.0, 0.0])

# Initial positions (as vectors)
r1 = np.array([10.0, 0.0])
r2 = np.array([0.0, 0.0])

# Example usage
m1 = 5.0  # Mass of object 1
m2 = 10.0  # Mass of object 2

# Function to calculate magnitude of a 2D vector
def mag(x):
    return math.sqrt(x[0] ** 2 + x[1] ** 2)

# Function to calculate position update over time (assumed to use energy equation)
def funcOfTime(m, x, x0, v0):
    v0 = 0  # Setting initial velocity to 0 as per the original code
    dx = np.sqrt(v0**2 - 2 * G * m * (1/x - 1/x0))
    return dx

# Example Step function (as used in previous simulations)
def step(dt):
    global v1, v2, r1, r2
    # Assuming we are updating positions and velocities
    # Here we'll apply the inverse square law to get the new velocities and positions
    
    # Calculate force between the two objects (simplified for this example)
    dx = r2[0] - r1[0]
    dy = r2[1] - r1[1]
    r = np.sqrt(dx**2 + dy**2)
    
    print(r)
    F = G * m1 * m2 / r**2  # Force magnitude
    Fx = F * dx / r  # Force in x direction
    Fy = F * dy / r  # Force in y direction
    
    # Calculate accelerations
    a1 = np.array([Fx / m1, Fy / m1])  # Acceleration on object 1
    a2 = np.array([Fx / m2, Fy / m2])    # Acceleration on object 2
    
    # Update velocities
    v1 = v1 + a1 * dt
    v2 = v2 + a2 * dt
    
    # Update positions
    r1 = r1 + v1 * dt
    r2 = r2 + v2 * dt
    
    return




for steps in range(100):

    if steps == 0:
        continue
    else:
        dt = timesteps[steps] - timesteps[steps - 1]

    step(dt)


    # Calculate distance between the two objects after the step
    dist = mag(r2 - r1)
    
    #print(f"x1,y1 = {x1_new}, x2,y2 = {x2_new}")

    # Output the distance
    print(f"Distance between objects: {dist}")


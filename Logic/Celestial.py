import numpy as np

class Body:

    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

    def update_position(self, dt, acceleration):
        """
        Update the position of the body using Verlet integration.
        """
        self.position += self.velocity * dt + 0.5 * acceleration * dt ** 2

    def update_velocity(self, dt, acceleration_new, acceleration_old):
        """
        Update the velocity of the body using the average of old and new accelerations.
        """
        self.velocity += 0.5 * (acceleration_old + acceleration_new) * dt

def compute_grav_force(body1, body2, G): # Some point this will become a body array
    """
    Compute the gravitational force exerted on body1 by body2.
    """
    # Vector from body1 to body2
    r_vec = body2.position - body1.position
    # Distance between bodies
    r_mag = np.linalg.norm(r_vec)
    # Force magnitude
    F_mag = G * body1.mass * body2.mass / r_mag ** 2
    # Force vector
    F_vec = F_mag * (r_vec / r_mag)
    
    return F_vec

def step(dt, body1, body2, G):
    """
    Perform a single simulation step, updating the positions and velocities of both bodies.
    """
    # Calculate initial accelerations
    F12 = compute_grav_force(body1, body2, G)
    a1_old = F12 / body1.mass
    a2_old = -F12 / body2.mass  # Newton's Third Law

    # Update positions
    body1.update_position(dt, a1_old)
    body2.update_position(dt, a2_old)

    # Recalculate forces with new positions
    F12_new = compute_grav_force(body1, body2, G)
    a1_new = F12_new / body1.mass
    a2_new = -F12_new / body2.mass

    # Update velocities
    body1.update_velocity(dt, a1_new, a1_old)
    body2.update_velocity(dt, a2_new, a2_old)

# # Constants
# G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
# dt = 0.01        # Time step in seconds
# numsteps = 5

# # Create celestial bodies
# body1 = CelestialBody(
#     mass=5.0,                  # Mass in kilograms
#     position=[0.0, 0.0],       # Initial position in meters
#     velocity=[0.0, 0.0]        # Initial velocity in meters per second
# )

# body2 = CelestialBody(
#     mass=1.0,                  # Mass in kilograms
#     position=[10.0, 0.0],       # Initial position in meters
#     velocity=[0.0, 0.0]        # Initial velocity in meters per second
# )

# for _ in range(numsteps):
#     # Run the simulation for one step
#     step(dt, body1, body2, G)
#     # Output the updated states
#     print(f"Body 1 Position: {body1.position}, Velocity: {body1.velocity}")
#     print(f"Body 2 Position: {body2.position}, Velocity: {body2.velocity}")



        

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

def position_exists(position, existing_positions, tol=1e-8):
    """Checks if a given position is already in the list of existing positions."""
    for pos in existing_positions:
        if all(abs(a - b) < tol for a, b in zip(position, pos)):
            return True
    return False

def get_body_input(index, num_dimensions, existing_positions):
    """Collects input for a single body and returns a Body instance."""
    print(f"\nEnter details for Body {index + 1}:")
    while True:
        try:
            position = []
            for dim in range(num_dimensions):
                coord = float(input(f"  Position component {dim+1}: "))
                position.append(coord)
            
            # Check if position conflicts with existing positions
            if position_exists(position, existing_positions):
                print("  This position is already occupied by another body. Please enter a different position.")
                continue  # Prompt for position again

            m = float(input("  Mass: "))
            if m <= 0:
                print("  Mass must be positive.")
                continue

            velocity = []
            for dim in range(num_dimensions):
                vel = float(input(f"  Velocity component {dim+1}: "))
                velocity.append(vel)

            return Body(mass=m, position=position, velocity=velocity)
        except ValueError:
            print("  Invalid input. Please enter numerical values.")

# Collect number of dimensions
while True:
    try:
        num_dimensions = int(input("Number of dimensions (2 or 3): "))
        if num_dimensions not in [2, 3]:
            print("Please enter 2 or 3.")
            continue
        break
    except ValueError:
        print("Invalid input. Please enter 2 or 3.")

# Collect number of bodies
while True:
    try:
        num_bodies = int(input("Number of bodies: "))
        if num_bodies <= 0:
            print("Please enter a positive integer.")
            continue
        break
    except ValueError:
        print("Invalid input. Please enter an integer.")

# Collect data for each body
bodies = []
existing_positions = []
for i in range(num_bodies):
    body = get_body_input(i, num_dimensions, existing_positions)
    bodies.append(body)
    existing_positions.append(body.position)

# Gravitational constant (normalized)
G = 1

# Prepare initial conditions for ODE solver
def prepare_initial_conditions(bodies, num_dimensions):
    """Prepares the initial conditions array for the ODE solver."""
    initial_conditions = []
    for body in bodies:
        initial_conditions.extend(body.velocity)
    for body in bodies:
        initial_conditions.extend(body.position)
    return initial_conditions

# ODE function for N-body problem
def n_body_equations(t, U):
    """Compute the derivatives for the N-body problem."""
    N = len(bodies)
    positions = U[N*num_dimensions:].reshape((N, num_dimensions))
    velocities = U[:N*num_dimensions].reshape((N, num_dimensions))
    masses = np.array([body.mass for body in bodies])

    accelerations = np.zeros((N, num_dimensions))
    for i in range(N):
        for j in range(N):
            if i != j:
                diff = positions[j] - positions[i]
                distance = np.linalg.norm(diff) + 1e-8  # Avoid division by zero
                accelerations[i] += G * masses[j] * diff / distance**3

    derivatives = np.concatenate([accelerations.flatten(), velocities.flatten()])
    return derivatives

# Time span and evaluation times
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1001)

# Solve the ODE
print("Solving ODE...")
U0 = prepare_initial_conditions(bodies, num_dimensions)
sol = solve_ivp(n_body_equations, t_span, U0, method='RK45', t_eval=t_eval)
if not sol.success:
    print("ODE solver failed:", sol.message)
    exit()
print("ODE solved.")

# Extract positions for plotting
N = len(bodies)
positions = sol.y[N*num_dimensions:].reshape((N, num_dimensions, -1))  # Shape: (N_bodies, num_dimensions, N_time_steps)

# Determine plot limits
all_positions = positions.reshape(-1, num_dimensions)
mins = all_positions.min(axis=0)
maxs = all_positions.max(axis=0)

# Set up the plot
if num_dimensions == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
elif num_dimensions == 2:
    fig, ax = plt.subplots()

# Colors for plotting bodies
colors = plt.cm.jet(np.linspace(0, 1, N))

# Animation loop
for i in range(len(sol.t)):
    ax.clear()
    for idx, body in enumerate(bodies):
        # Plot trajectory up to current time
        if num_dimensions == 3:
            ax.plot(positions[idx, 0, :i+1], positions[idx, 1, :i+1], positions[idx, 2, :i+1],
                    color=colors[idx], label=f'Body {idx + 1}' if i == 0 else "")
            # Plot current position
            ax.scatter(positions[idx, 0, i], positions[idx, 1, i], positions[idx, 2, i],
                       color=colors[idx], s=50)
        elif num_dimensions == 2:
            ax.plot(positions[idx, 0, :i+1], positions[idx, 1, :i+1],
                    color=colors[idx], label=f'Body {idx + 1}' if i == 0 else "")
            # Plot current position
            ax.scatter(positions[idx, 0, i], positions[idx, 1, i],
                       color=colors[idx], s=50)
    # Set plot limits
    ax.set_xlim([mins[0], maxs[0]])
    ax.set_ylim([mins[1], maxs[1]])
    if num_dimensions == 3:
        ax.set_zlim([mins[2], maxs[2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if num_dimensions == 3:
        ax.set_zlabel('Z')
    if i == 0:
        ax.legend()
    plt.pause(0.001)  # Pause to create animation effect

plt.show()

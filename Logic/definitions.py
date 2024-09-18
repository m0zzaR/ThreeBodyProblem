import numpy as np
import math

G = 1

timestep = np.linspace(1, 100, 100)
position = [0] * 100
print(position)




# Newtonian Law of Attracted Objects
def inversesquare(m1, m2, x1, y1, x2, y2):
    F1 = [ G * m1 / x1 ** 2, G * m2 / y1 ** 2 ]
    F2 = [ G * m1 / x2 ** 2, G * m2 / y2 ** 2 ]

    return [[F1, F2]]

# xi - xf = v0 * t + 1/2 * G * m / R^2 * t^2

# Stepper function for changing R vector
def stepper():



    for t in timestep:
        print()
    return 1

def mag(x):
    return math.sqrt(sum(component**2 for component in x))

def funcOfTime(m, x, x0, v0):
    v0 = 0
    dx = np.sqrt(v0**2 - 2 * G * m * (1/x - 1/x0))
    return dx
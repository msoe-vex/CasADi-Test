import numpy as np
import matplotlib.pyplot as plt
from casadi import *

# Set up parameters
start_pos = [0, 0]
target_pos = [14, 14]
obstacles = [[5, 5], [7, 3]]  # Positions of obstacles
obstacle_radius = 1
num_steps = 100  # Number of discrete steps

# Define variables
opti = Opti()  # Optimization environment

# State variables for robot's position at each time step
x = opti.variable(num_steps + 1)
y = opti.variable(num_steps + 1)

# Control variables (velocities)
vx = opti.variable(num_steps)
vy = opti.variable(num_steps)

# Objective: Minimize the total path length and the distance to the target
objective = 0
for t in range(num_steps):
    objective += (x[t+1] - x[t])**2 + (y[t+1] - y[t])**2  # path length term
objective += (x[-1] - target_pos[0])**2 + (y[-1] - target_pos[1])**2  # target distance term

opti.minimize(objective)

# Initial position constraints
opti.subject_to(x[0] == start_pos[0])
opti.subject_to(y[0] == start_pos[1])

# Movement constraints (Euler integration)
for t in range(num_steps):
    opti.subject_to(x[t+1] == x[t] + vx[t])
    opti.subject_to(y[t+1] == y[t] + vy[t])

# Add obstacle avoidance constraints
for obs in obstacles:
    for t in range(num_steps + 1):
        # Ensure the robot stays away from each obstacle by a safe margin
        opti.subject_to((x[t] - obs[0])**2 + (y[t] - obs[1])**2 >= (obstacle_radius + 0.5)**2)

# Add maximum velocity constraints to ensure realistic motion
max_velocity = 2  # max velocity (you can adjust this)
for t in range(num_steps):
    opti.subject_to(vx[t]**2 + vy[t]**2 <= max_velocity**2)

# Solver options
opti.solver("ipopt")  # Use an interior-point optimization solver

# Solve the problem
try:
    sol = opti.solve()
    x_opt = sol.value(x)
    y_opt = sol.value(y)
except:
    print("No solution found!")
    x_opt, y_opt = [], []

# Plotting results
plt.figure(figsize=(8, 8))
plt.plot(x_opt, y_opt, '-o', label="Robot Path")
plt.scatter(*target_pos, color='green', s=100, label="Target")
plt.scatter(*start_pos, color='blue', s=100, label="Start")
for obs in obstacles:
    circle = plt.Circle(obs, obstacle_radius, color='red', alpha=0.5)
    plt.gca().add_patch(circle)
plt.xlim(-5, 20)
plt.ylim(-5, 20)
plt.legend()
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("CasADi Robot Path Optimization")
plt.grid(True)
plt.show()


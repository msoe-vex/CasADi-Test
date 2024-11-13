import numpy as np
import matplotlib.pyplot as plt
from casadi import *

# Set up parameters
start_pos = [0, 0]
target_pos = [100, 120]
start_vel = [0, 0]
obstacles = [[50, 50], [70, 30]]  # Positions of obstacles
obstacle_radius = 10
robot_radius = 10
num_steps = 50  # Number of discrete steps

# Define variables
opti = Opti()  # Optimization environment

# State variables for robot's position at each time step
x = opti.variable(num_steps + 1)
y = opti.variable(num_steps + 1)

# Control variables (velocities)
vx = opti.variable(num_steps)
vy = opti.variable(num_steps)

# Time step variable (assumed constant across all steps)
dt = opti.variable()
initial_dt = 5 / num_steps
opti.set_initial(dt, initial_dt)  # Initial guess for dt
opti.subject_to(dt * num_steps > 0.1)  # Relaxed lower bound
opti.subject_to(dt * num_steps < 5)  # Upper bound to control time step size

for t in range(num_steps + 1):
    opti.set_initial(x[t], start_pos[0] + t * (target_pos[0] - start_pos[0]) / num_steps)
    opti.set_initial(y[t], start_pos[1] + t * (target_pos[1] - start_pos[1]) / num_steps)
for t in range(num_steps):
    opti.set_initial(vx[t], (target_pos[0] - start_pos[0]) / (initial_dt * num_steps))
    opti.set_initial(vy[t], (target_pos[1] - start_pos[1]) / (initial_dt * num_steps))

# Objective: Minimize total time with reduced path length and smoothness penalties
path_length_penalty = 1e-6  # Smaller penalty for path length
velocity_smoothing_weight = 1e-4  # Smaller weight for velocity smoothness

# Define path length and velocity smoothing terms
path_length = 0
velocity_smoothing = 0

for t in range(num_steps):
    path_length += (x[t + 1] - x[t])**2 + (y[t + 1] - y[t])**2
    if t < num_steps - 1:
        velocity_smoothing += (vx[t + 1] - vx[t])**2 + (vy[t + 1] - vy[t])**2

# Combined objective function
opti.minimize(dt * num_steps + path_length_penalty * path_length + velocity_smoothing_weight * velocity_smoothing)

# Initial position and velocity constraints
opti.subject_to(x[0] == start_pos[0])
opti.subject_to(y[0] == start_pos[1])
opti.subject_to(vx[0] == start_vel[0])
opti.subject_to(vy[0] == start_vel[1])

# Movement constraints (Euler integration)
for t in range(num_steps):
    opti.subject_to(x[t + 1] == x[t] + dt * vx[t])
    opti.subject_to(y[t + 1] == y[t] + dt * vy[t])

# Increase final position and stop tolerance
position_tolerance = 1  # Increased tolerance for target position
velocity_tolerance = 0.1  # Increased tolerance for velocity stop
opti.subject_to((x[-1] - target_pos[0])**2 + (y[-1] - target_pos[1])**2 <= position_tolerance**2)
opti.subject_to(vx[num_steps - 1]**2 + vy[num_steps - 1]**2 <= velocity_tolerance**2)

# Avoid driving outside the field
opti.subject_to(x > 0)
opti.subject_to(x < 144)
opti.subject_to(y > 0)
opti.subject_to(y < 144)

# Obstacle avoidance with reduced buffer
for obs in obstacles:
    for t in range(num_steps + 1):
        opti.subject_to((x[t] - obs[0])**2 + (y[t] - obs[1])**2 >= (obstacle_radius + robot_radius - 1)**2)

# Maximum velocity constraint
max_velocity = 75
for t in range(num_steps):
    opti.subject_to(vx[t]**2 + vy[t]**2 <= max_velocity**2)

# Maximum acceleration constraints with slightly relaxed limit
max_acceleration = 50
for t in range(num_steps - 1):
    ax = (vx[t + 1] - vx[t]) / dt
    ay = (vy[t + 1] - vy[t]) / dt
    opti.subject_to(ax**2 + ay**2 <= max_acceleration**2)

# Solver options with increased iterations and feasibility tolerance
opti.solver("ipopt", {
    "ipopt.max_iter": 5000,
    "ipopt.mu_strategy": "adaptive",
    "ipopt.mu_target": 1e-4,
    "ipopt.tol": 1e-4,
    "ipopt.acceptable_tol": 1e-2,
    "ipopt.acceptable_iter": 20,
    "ipopt.linear_solver": "mumps",
})

# Solve the problem
try:
    sol = opti.solve()
    x_opt = sol.value(x)
    y_opt = sol.value(y)
    vx_opt = sol.value(vx)
    vy_opt = sol.value(vy)
    dt_opt = sol.value(dt)

    print("\n(t) (x, y) (vx, vy) (ax, ay)")
    for i in range(num_steps):
        print(f"({dt_opt*(i+1):.2f}) ({x_opt[i+1]:.4f}, {y_opt[i+1]:.4f}) ({vx_opt[i]:.4f}, {vy_opt[i]:.4f})")
    print(f"\nOptimal time step: {dt_opt}")
    print(f"Total time to reach target: {dt_opt * num_steps}")

except RuntimeError:
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
plt.xlim(-1, 145)
plt.ylim(-1, 145)
plt.legend()
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("CasADi Robot Path Optimization with Convergence Improvements")
plt.grid(True)
plt.show()

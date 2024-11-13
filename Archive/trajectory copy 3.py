import numpy as np
import matplotlib.pyplot as plt
from casadi import *

# Set up parameters
start_pos = [0, 0]
target_pos = [140, 140]
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
opti.set_initial(dt, 5 / num_steps)  # Initial guess for dt
opti.subject_to(dt * num_steps > 0)
opti.subject_to(dt * num_steps < 5)  # Upper bound for dt to help convergence

# Objective: Minimize the total time
opti.minimize(dt * num_steps)

# Initial position and velocity constraints
opti.subject_to(x[0] == start_pos[0])
opti.subject_to(y[0] == start_pos[1])
opti.subject_to(vx[0] == start_vel[0])
opti.subject_to(vy[0] == start_vel[1])

# Movement constraints (Euler integration)
for t in range(num_steps):
    opti.subject_to(x[t + 1] == x[t] + dt * vx[t])
    opti.subject_to(y[t + 1] == y[t] + dt * vy[t])

# Final position constraint
tolerance = 1
opti.subject_to((x[-1] - target_pos[0])**2 + (y[-1] - target_pos[1])**2 <= tolerance**2)

# Ensure robot comes to a stop at target
stopped_tolerance = 0.5
opti.subject_to(vx[num_steps - 1] <= stopped_tolerance)
opti.subject_to(vy[num_steps - 1] <= stopped_tolerance)

# Avoid driving outside the field
opti.subject_to(x > 0)
opti.subject_to(x < 144)
opti.subject_to(y > 0)
opti.subject_to(y < 144)

# Add obstacle avoidance constraints
for obs in obstacles:
    for t in range(num_steps + 1):
        opti.subject_to((x[t] - obs[0])**2 + (y[t] - obs[1])**2 >= (obstacle_radius + robot_radius)**2)

# Maximum velocity constraints
max_velocity = 75
for t in range(num_steps):
    opti.subject_to(vx[t]**2 + vy[t]**2 <= max_velocity**2)

# Maximum acceleration constraints
max_acceleration = 40
ax = [None] * (num_steps - 1)
ay = [None] * (num_steps - 1)
for t in range(num_steps - 1):
    ax[t] = (vx[t + 1] - vx[t]) / dt
    ay[t] = (vy[t + 1] - vy[t]) / dt
    opti.subject_to(ax[t]**2 + ay[t]**2 <= max_acceleration**2)

# Maximum jerk constraints (rate of change of acceleration)
max_jerk = 150  # Define max jerk
jx = [None] * (num_steps - 2)
jy = [None] * (num_steps - 2)
for t in range(num_steps - 2):
    jx[t] = (ax[t + 1] - ax[t]) / dt  # Jerk in x
    jy[t] = (ay[t + 1] - ay[t]) / dt  # Jerk in y
    opti.subject_to(jx[t]**2 + jy[t]**2 <= max_jerk**2)

# Solver options
opti.solver("ipopt", {
    "ipopt.max_iter": 2000,
    "ipopt.tol": 1e-6,
    "ipopt.acceptable_tol": 1e-2,
    "ipopt.acceptable_iter": 10,
    "ipopt.print_level": 3,
})

# Solve the problem
try:
    sol = opti.solve()
    x_opt = sol.value(x)
    y_opt = sol.value(y)
    vx_opt = sol.value(vx)
    vy_opt = sol.value(vy)
    dt_opt = sol.value(dt)

    print()
    print("(t) (x, y) (vx, vy) (ax, ay)")
    print(f"(0.00) ({x_opt[0]:.4f}, {y_opt[0]:.4f})")
    for i in range(num_steps):
        print(f"({dt_opt*(i+1):.2f}) ({x_opt[i+1]:.4f}, {y_opt[i+1]:.4f}) ({vx_opt[i]:.4f}, {vy_opt[i]:.4f})")
    print()
    print(f"Optimal time step: {dt_opt}")
    print(f"Total time to reach target: {dt_opt * num_steps}")
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
plt.xlim(-50, 200)
plt.ylim(-50, 200)
plt.legend()
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("CasADi Robot Path Optimization with Jerk Constraint")
plt.grid(True)
plt.show()

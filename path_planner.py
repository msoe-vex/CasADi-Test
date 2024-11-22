#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Path Planner
# Author: Tianchen Ji
# Email: tj12@illinois.edu
# Create Date: 2019-11-26
# ---------------------------------------------------------------------------

from casadi import *
import numpy as np
import matplotlib.pyplot as plt

# The num of MPC actions, here include vx and vy
NUM_OF_ACTS = 2

# The num of MPC states, here include px and py
NUM_OF_STATES = 2

NUM_OF_G_STATES = 1

# MPC parameters
lookahead_step_num = 50
lookahead_step_timeinterval = 0.1

# start point and end point
start_point = [0.1, 0.1]
end_point = [0.9, 0.9]

robot_radius = 8/144

# Define maximum allowable velocity and acceleration
max_velocity = 100/144
max_accel = 100/144  # Example value, adjust as needed

# List of obstacle coordinates
obstacles = [[3/6, 2/6], [3/6, 4/6], [2/6, 3/6], [4/6, 3/6]]

# threshold of safety
obstacle_radius = 3.5/144

class FirstStateIndex:
	def __init__(self, n):
		self.px = 0
		self.py = self.px + n
		self.vx = self.py + n
		self.vy = self.vx + n - 1

class MPC:
	def __init__(self):
		self.first_state_index_ = FirstStateIndex(lookahead_step_num)
		self.num_of_x_ = NUM_OF_STATES * lookahead_step_num + NUM_OF_ACTS * (lookahead_step_num - 1)
		self.num_of_g_ = (NUM_OF_STATES + NUM_OF_G_STATES) * lookahead_step_num + len(obstacles) * lookahead_step_num + (lookahead_step_num - 1) * NUM_OF_ACTS  # Including acceleration constraints

	def Solve(self, state):
		# Define optimization variables
		x = SX.sym('x', self.num_of_x_)

		# Define cost functions
		w_cte = 10.0
		w_dv = 1.0
		cost = 0.0

		# Initial variables
		x_ = [0] * self.num_of_x_
		x_[self.first_state_index_.px] = state[0]
		x_[self.first_state_index_.py] = state[1]

		# Penalty on states
		for i in range(lookahead_step_num - 1, lookahead_step_num):
			cte = (x[self.first_state_index_.px + i] - end_point[0])**2 + (x[self.first_state_index_.py + i] - end_point[1])**2
			cost += w_cte * cte

		# Penalty on inputs
		for i in range(lookahead_step_num - 2):
			dvx = x[self.first_state_index_.vx + i + 1] - x[self.first_state_index_.vx + i]
			dvy = x[self.first_state_index_.vy + i + 1] - x[self.first_state_index_.vy + i]
			cost += w_dv * (dvx**2) + w_dv * (dvy**2)
		
		# Additional penalty on final position
		final_x = x[self.first_state_index_.px + lookahead_step_num - 1]
		final_y = x[self.first_state_index_.py + lookahead_step_num - 1]
		cost += 50.0 * ((final_x - end_point[0])**2 + (final_y - end_point[1])**2)  # High weight to strongly encourage final stop

		# Additional penalty on final velocities to encourage a stop
		final_vx = x[self.first_state_index_.vx + lookahead_step_num - 2]
		final_vy = x[self.first_state_index_.vy + lookahead_step_num - 2]
		cost += 50.0 * (final_vx**2 + final_vy**2)  # High weight to strongly encourage final stop

		# Additional penalty on initial velocities to encourage starting from zero
		initial_vx = x[self.first_state_index_.vx]
		initial_vy = x[self.first_state_index_.vy]
		cost += 50.0 * (initial_vx**2 + initial_vy**2)  # High weight to strongly encourage final stop

		# Define lowerbound and upperbound of x
		x_lowerbound_ = [-exp(10)] * self.num_of_x_
		x_upperbound_ = [exp(10)] * self.num_of_x_
		for i in range(self.first_state_index_.vx, self.num_of_x_):
			x_lowerbound_[i] = -max_velocity
			x_upperbound_[i] = max_velocity

		# Define lowerbound and upperbound of g constraints
		g_lowerbound_ = [0] * self.num_of_g_
		g_upperbound_ = [0] * self.num_of_g_

		g_lowerbound_[self.first_state_index_.px] = state[0]
		g_upperbound_[self.first_state_index_.px] = state[0]
		g_lowerbound_[self.first_state_index_.py] = state[1]
		g_upperbound_[self.first_state_index_.py] = state[1]

		# Initialize g constraints list with SX elements
		g = [SX(0)] * self.num_of_g_

		# Set initial position constraints
		g[self.first_state_index_.px] = x[self.first_state_index_.px]
		g[self.first_state_index_.py] = x[self.first_state_index_.py]

		# Set initial and final velocity constraints (as per previous step)
		g[self.first_state_index_.vx] = x[self.first_state_index_.vx]  # Initial vx = 0
		g[self.first_state_index_.vy] = x[self.first_state_index_.vy]  # Initial vy = 0
		g[self.first_state_index_.vx + lookahead_step_num - 2] = x[self.first_state_index_.vx + lookahead_step_num - 2]  # Final vx = 0
		g[self.first_state_index_.vy + lookahead_step_num - 2] = x[self.first_state_index_.vy + lookahead_step_num - 2]  # Final vy = 0

		# Set bounds for the initial and final velocity constraints
		g_lowerbound_[self.first_state_index_.vx] = 0
		g_upperbound_[self.first_state_index_.vx] = 0
		g_lowerbound_[self.first_state_index_.vy] = 0
		g_upperbound_[self.first_state_index_.vy] = 0
		g_lowerbound_[self.first_state_index_.vx + lookahead_step_num - 2] = 0
		g_upperbound_[self.first_state_index_.vx + lookahead_step_num - 2] = 0
		g_lowerbound_[self.first_state_index_.vy + lookahead_step_num - 2] = 0
		g_upperbound_[self.first_state_index_.vy + lookahead_step_num - 2] = 0

		# Add acceleration constraints
		g_index = 1 + self.first_state_index_.py + 1 * lookahead_step_num
		for i in range(lookahead_step_num - 2):
			curr_vx_index = i + self.first_state_index_.vx
			curr_vy_index = i + self.first_state_index_.vy
			next_vx_index = curr_vx_index + 1
			next_vy_index = curr_vy_index + 1

			# Acceleration in x direction
			ax = (x[next_vx_index] - x[curr_vx_index]) / lookahead_step_timeinterval
			g[g_index] = ax
			g_lowerbound_[g_index] = -max_accel
			g_upperbound_[g_index] = max_accel
			g_index += 1

			# Acceleration in y direction
			ay = (x[next_vy_index] - x[curr_vy_index]) / lookahead_step_timeinterval
			g[g_index] = ay
			g_lowerbound_[g_index] = -max_accel
			g_upperbound_[g_index] = max_accel
			g_index += 1

		# Update obstacle constraints as previously defined
		for i in range(lookahead_step_num - 1):
			curr_px_index = i + self.first_state_index_.px
			curr_py_index = i + self.first_state_index_.py
			curr_vx_index = i + self.first_state_index_.vx
			curr_vy_index = i + self.first_state_index_.vy

			curr_px = x[curr_px_index]
			curr_py = x[curr_py_index]
			curr_vx = x[curr_vx_index]
			curr_vy = x[curr_vy_index]

			next_px = x[1 + curr_px_index]
			next_py = x[1 + curr_py_index]

			next_m_px = curr_px + curr_vx * lookahead_step_timeinterval
			next_m_py = curr_py + curr_vy * lookahead_step_timeinterval

			# Equality constraints
			g[1 + curr_px_index] = next_px - next_m_px
			g[1 + curr_py_index] = next_py - next_m_py

			# Inequality constraints for each obstacle
			for obstacle in obstacles:
				g[g_index] = (next_px - obstacle[0])**2 + (next_py - obstacle[1])**2
				g_lowerbound_[g_index] = (obstacle_radius+robot_radius)**2
				g_upperbound_[g_index] = exp(10)
				g_index += 1

		# Create the NLP
		nlp = {'x': x, 'f': cost, 'g': vertcat(*g)}

		# Solver options
		opts = {}
		opts["ipopt.print_level"] = 3
		opts["print_time"] = 0

		solver = nlpsol('solver', 'ipopt', nlp, opts)

		# Solve the NLP
		res = solver(x0=x_, lbx=x_lowerbound_, ubx=x_upperbound_, lbg=g_lowerbound_, ubg=g_upperbound_)
		return res
	
	def print_trajectory_details(self, res):
		"""
		Print the position, velocity, and acceleration of each step in the trajectory.

		Parameters:
		- res: The result object from the solver containing optimized trajectory.
		- lookahead_step_num: The number of steps in the trajectory.
		- lookahead_step_timeinterval: Time interval between each step.
		"""
		# Extract the optimized trajectory from the result
		x_opt = res['x'].full().flatten()
		
		# Print header
		print(f"{'Step':<5} {'Position (x, y)':<20}\t{'Velocity (vx, vy)':<20}\t{'Acceleration (ax, ay)':<25}")
		print("-" * 70)

		lemlib_output_string = ""
		
		# Loop through each step in the trajectory
		for i in range(lookahead_step_num):
			# Position at step i
			px = x_opt[self.first_state_index_.px + i]
			py = x_opt[self.first_state_index_.py + i]

			if i < lookahead_step_num - 1:  # Ensure we don’t go out of bounds
				vx = x_opt[self.first_state_index_.vx + i]
				vy = x_opt[self.first_state_index_.vy + i]
			else:
				vx = vy = 0  # No velocity at the last step

			# Acceleration between step i and i+1
			if i < lookahead_step_num - 2:  # Ensure we don’t go out of bounds
				next_vx = x_opt[self.first_state_index_.vx + i + 1]
				next_vy = x_opt[self.first_state_index_.vy + i + 1]
				ax = (next_vx - vx) / lookahead_step_timeinterval
				ay = (next_vy - vy) / lookahead_step_timeinterval
			else:
				ax = ay = 0  # No acceleration at the last step

			# Print the details for this step
			print(f"{i:<5} ({px:.2f}, {py:.2f})\t\t({vx:.2f}, {vy:.2f})\t\t({ax:.2f}, {ay:.2f})")

			speed = sqrt(vx*vx+vy*vy)

			lemlib_output_string += f"{px*144:.3f}, {py*144:.3f}, {speed*144:.3f}\n"
		lemlib_output_string += "endData"

		file = open('path_output.txt', 'w')
		file.write(lemlib_output_string)
		file.close()


mpc_ = MPC()
sol = mpc_.Solve(start_point)

mpc_.print_trajectory_details(sol)

# Plot results
fig = plt.figure(figsize=(7, 7))
planned_px = sol['x'][0:1 * lookahead_step_num]
planned_py = sol['x'][1 * lookahead_step_num:2 * lookahead_step_num]
plt.plot(planned_px, planned_py, 'o-', label='planned trajectory')
theta = np.arange(0, 2 * np.pi, 0.01)
for obstacle in obstacles:
	danger_x = obstacle[0] + (obstacle_radius - 0.005) * np.cos(theta)
	danger_y = obstacle[1] + (obstacle_radius - 0.005) * np.sin(theta)
	plt.plot(danger_x, danger_y, 'r--', label='danger area')
plt.plot(start_point[0], start_point[1], 'o', label='start point')
plt.plot(end_point[0], end_point[1], 'o', label='target point')
for obstacle in obstacles:
	plt.plot(obstacle[0], obstacle[1], 'o', label='obstacle')
plt.legend(loc='upper left')
plt.axis('equal')
plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.grid()
plt.show()
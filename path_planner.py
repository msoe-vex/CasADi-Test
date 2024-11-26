from casadi import *
import numpy as np
import matplotlib.pyplot as plt

# The num of MPC actions, here include vx and vy
NUM_OF_ACTS = 2

# The num of MPC states, here include px and py
NUM_OF_STATES = 2

# MPC parameters
lookahead_step_num = 30
lookahead_step_timeinterval = 0.1

# start point and end point
start_point = [0.1, 0.1]
end_point = [.6, .8]

robot_radius = 8/144

# Define maximum allowable velocity and acceleration
max_velocity = 70/144
max_accel = 200/144  # Example value, adjust as needed

# List of obstacle coordinates
obstacles = [[3/6, 2/6], [3/6, 4/6], [2/6, 3/6], [4/6, 3/6]]

# threshold of safety
obstacle_radius = 3.5/144
#obstacle_radius = 0.1

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
		#constraints for position at each obstacle, velocity, and acceleration
		self.num_of_g_ = ((lookahead_step_num-1) * len(obstacles))  + (lookahead_step_num-1)*NUM_OF_ACTS + (lookahead_step_num-1) + (lookahead_step_num - 2) 

	def Solve(self, state):
		# Define optimization variables
		x = SX.sym('x', self.num_of_x_)

		# Define cost functions
		w_cte = 10.0 # Bigger value means last point has to be closer to designated end point
		w_dv = 10 # Bigger value means smoother path
		cost = 0.0

		# Initial variables
		x_ = [0] * self.num_of_x_

		# Set initial values as a linear path
		init_x = numpy.linspace(state[0], end_point[0], lookahead_step_num)
		init_y = numpy.linspace(state[1], end_point[1], lookahead_step_num)

		#Set initial velocity to 0
		init_v = [0] * ((lookahead_step_num-1) * NUM_OF_ACTS)
		x_ = np.concatenate((init_x, init_y, init_v))

		# Penalty on inputs, smoothing path, penalizes change in velocity
		for i in range(lookahead_step_num - 2):
			dvx = x[self.first_state_index_.vx + i + 1] - x[self.first_state_index_.vx + i]
			dvy = x[self.first_state_index_.vy + i + 1] - x[self.first_state_index_.vy + i]
			cost += w_dv * (dvx**2 + dvy**2)
		
		# # Penalty on states, avoid going towards the middle of the field
		# for i in range(lookahead_step_num - 1):
		# 	px = x[self.first_state_index_.px+i]
		# 	py = x[self.first_state_index_.py+i]

		# 	cost += if_else((px-0.5)**2 + (py-0.5)**2 < (1/6)**2, 1e10, 0)

		# Define lowerbound and upperbound for position and velocity
		x_lowerbound_ = [-exp(10)] * self.num_of_x_
		x_upperbound_ = [exp(10)] * self.num_of_x_

		# Ensure path does not go outside the field
		for i in range(self.first_state_index_.px, self.first_state_index_.py+lookahead_step_num):
			x_lowerbound_[i] = 0
			x_upperbound_[i] = 1
		# Ensure velocity does not exceed the max speed
		for i in range(self.first_state_index_.vx, self.first_state_index_.vy+lookahead_step_num-1):
			x_lowerbound_[i] = -max_velocity
			x_upperbound_[i] = max_velocity
		
		#Constrain initial and final position
		x_lowerbound_[self.first_state_index_.px] = state[0]
		x_lowerbound_[self.first_state_index_.py] = state[1]
		x_lowerbound_[self.first_state_index_.px+lookahead_step_num-1] = end_point[0]
		x_lowerbound_[self.first_state_index_.py+lookahead_step_num-1] = end_point[1]
		x_upperbound_[self.first_state_index_.px] = state[0]
		x_upperbound_[self.first_state_index_.py] = state[1]
		x_upperbound_[self.first_state_index_.px+lookahead_step_num-1] = end_point[0]
		x_upperbound_[self.first_state_index_.py+lookahead_step_num-1] = end_point[1]

		#Constrain initial and final velocity
		x_lowerbound_[self.first_state_index_.vx] = 0
		x_lowerbound_[self.first_state_index_.vy] = 0
		x_lowerbound_[self.first_state_index_.vx+lookahead_step_num-2] = 0
		x_lowerbound_[self.first_state_index_.vy+lookahead_step_num-2] = 0
		x_upperbound_[self.first_state_index_.vx] = 0
		x_upperbound_[self.first_state_index_.vy] = 0
		x_upperbound_[self.first_state_index_.vx+lookahead_step_num-2] = 0
		x_upperbound_[self.first_state_index_.vy+lookahead_step_num-2] = 0

		# Define lowerbound and upperbound of g constraints
		g_lowerbound_ = [exp(-10)] * self.num_of_g_
		g_upperbound_ = [exp(10)] * self.num_of_g_

		# Initialize g constraints list with SX elements
		g = [SX(0)] * self.num_of_g_
	
		g_index = 0
		
		# Add speed constraints
		for i in range(lookahead_step_num - 1):
			curr_vx_index = self.first_state_index_.vx + i
			curr_vy_index = self.first_state_index_.vy + i
			vx = x[curr_vx_index]
			vy = x[curr_vy_index]

			# Constraint on velocity magnitude
			g[g_index] = vx**2 + vy**2
			g_lowerbound_[g_index] = 0  # Minimum speed (non-negative)
			g_upperbound_[g_index] = max_velocity**2  # Maximum speed in any direction
			g_index += 1

		# Add acceleration magnitude constraints
		for i in range(lookahead_step_num - 2):
			curr_vx_index = self.first_state_index_.vx + i
			curr_vy_index = self.first_state_index_.vy + i
			next_vx_index = curr_vx_index + 1
			next_vy_index = curr_vy_index + 1

			ax = (x[next_vx_index] - x[curr_vx_index]) / lookahead_step_timeinterval
			ay = (x[next_vy_index] - x[curr_vy_index]) / lookahead_step_timeinterval

			# Constraint on acceleration magnitude
			g[g_index] = ax**2 + ay**2
			g_lowerbound_[g_index] = 0  # Minimum acceleration (non-negative)
			g_upperbound_[g_index] = max_accel**2  # Maximum acceleration in any direction
			g_index += 1

		# Update position constraints as previously defined
		for i in range(lookahead_step_num-1):
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

			# equality constraints
			g[g_index] = next_px - next_m_px
			g_lowerbound_[g_index] = 0
			g_upperbound_[g_index] = 0
			g_index += 1

			g[g_index] = next_py - next_m_py
			g_lowerbound_[g_index] = 0
			g_upperbound_[g_index] = 0
			g_index += 1

		# Obstacle constraints
		for i in range(lookahead_step_num-1):
			curr_px_index = i + self.first_state_index_.px
			curr_py_index = i + self.first_state_index_.py

			curr_px = x[curr_px_index]
			curr_py = x[curr_py_index]

			# Inequality constraints for each obstacle
			for obstacle in obstacles:
				g[g_index] = (curr_px - obstacle[0])**2 + (curr_py - obstacle[1])**2
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
		x_opt = sol['x'].full().flatten()
		
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
	
	def plotResults(self, sol):
		# Create a figure with a flexible window size
		fig, ax = plt.subplots(figsize=(8, 8))  # The graph itself will remain square
		
		planned_px = sol['x'][0:1 * lookahead_step_num]
		planned_py = sol['x'][1 * lookahead_step_num:2 * lookahead_step_num]
		ax.plot(planned_px, planned_py, 'o-', label='path')
		theta = np.arange(0, 2 * np.pi, 0.01)
		
		ax.plot(start_point[0], start_point[1], 'o', label='start')
		ax.plot(end_point[0], end_point[1], 'o', label='target')
		
		first_obstacle = True
		for obstacle in obstacles:
			danger_x = obstacle[0] + (obstacle_radius - 0.005) * np.cos(theta)
			danger_y = obstacle[1] + (obstacle_radius - 0.005) * np.sin(theta)
			if first_obstacle:
				#ax.plot(obstacle[0], obstacle[1], 'o', color='red', label='obstacle')
				ax.plot(danger_x, danger_y, 'r-', label='obstacle')
				first_obstacle = False
			else:
				#ax.plot(obstacle[0], obstacle[1], 'o', color='red')
				ax.plot(danger_x, danger_y, 'r-')
		
		# Plot the circle in the middle of the graph with radius 1/6
		radius = 1 / 6
		center_x, center_y = 0.5, 0.5
		circle_x = center_x + radius * np.cos(theta)
		circle_y = center_y + radius * np.sin(theta)
		ax.plot(circle_x, circle_y, 'b-')
		
		# Add the legend outside the square graph
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0., frameon=False)
		
		# Ensure the graph remains square
		ax.set_aspect('equal', adjustable='box')  # Make sure the graph area is square
		ax.set_xlim(-0.1, 1.1)
		ax.set_ylim(-0.1, 1.1)
		ax.grid()
		
		# Display the plot
		plt.show()




mpc_ = MPC()
sol = mpc_.Solve(start_point)

mpc_.print_trajectory_details(sol)
mpc_.plotResults(sol)
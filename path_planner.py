from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

# The num of MPC actions, here include vx and vy
NUM_OF_ACTS = 2

# The num of MPC states, here include px and py
NUM_OF_STATES = 2

# MPC parameters
lookahead_step_num = 50
lookahead_step_timeinterval = 0.1

time_step_min = 0.05  # Minimum time step
time_step_max = 0.5   # Maximum time step

# start point and end point
start_point = [0.1, 0.1]
end_point = [.6, .8]

robot_length = 16/144
robot_width = 15/144

robot_radius = max(robot_length, robot_width)/sqrt(2)

# Define maximum allowable velocity and acceleration
max_velocity = 70/144
max_accel = 50/144  # Example value, adjust as needed

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
		self.dt = self.vy + n - 1

class MPC:
	def __init__(self):
		self.indexes = FirstStateIndex(lookahead_step_num)
		self.num_of_x_ = NUM_OF_STATES * lookahead_step_num + NUM_OF_ACTS * (lookahead_step_num - 1) + 1 # plus one for time step variable
		#constraints for position at each obstacle, velocity, and acceleration
		self.num_of_g_ = ((lookahead_step_num-1) * len(obstacles))  + (lookahead_step_num-1)*NUM_OF_ACTS + (lookahead_step_num-1) + (lookahead_step_num - 2) 

	def Solve(self, state):
		# Define optimization variables
		x = SX.sym('x', self.num_of_x_)
		self.indexes.dt = self.num_of_x_-1

		# Define cost functions
		w_cte = 10.0 # Bigger value means last point has to be closer to designated end point
		w_dv = 10 # Bigger value means smoother path
		w_time_step = 10.0  # Weight for penalizing the time step
		cost = 0.0

		# Initial variables
		x_ = [0] * self.num_of_x_

		# Set initial values as a linear path
		init_x = numpy.linspace(state[0], end_point[0], lookahead_step_num)
		init_y = numpy.linspace(state[1], end_point[1], lookahead_step_num)

		#Set initial velocity to 0
		init_v = [0] * ((lookahead_step_num-1) * NUM_OF_ACTS)
		init_time_step = lookahead_step_timeinterval  # Initial guess for the time step
		x_ = np.concatenate((init_x, init_y, init_v, [init_time_step]))

		# Penalty on inputs, smoothing path, penalizes change in velocity
		for i in range(lookahead_step_num - 2):
			dvx = x[self.indexes.vx + i + 1] - x[self.indexes.vx + i]
			dvy = x[self.indexes.vy + i + 1] - x[self.indexes.vy + i]
			cost += w_dv * (dvx**2 + dvy**2)
		
		time_step = x[self.indexes.dt]
		cost += w_time_step * time_step * lookahead_step_num
		
		# # Penalty on states, avoid going towards the middle of the field
		# for i in range(lookahead_step_num - 1):
		# 	px = x[self.indexes.px+i]
		# 	py = x[self.indexes.py+i]

		# 	cost += if_else((px-0.5)**2 + (py-0.5)**2 < (1/6)**2, 1e10, 0)

		# Define lowerbound and upperbound for position and velocity
		x_lowerbound_ = [-exp(10)] * self.num_of_x_
		x_upperbound_ = [exp(10)] * self.num_of_x_

		# Ensure path does not go outside the field
		for i in range(self.indexes.px, self.indexes.py+lookahead_step_num):
			x_lowerbound_[i] = 0
			x_upperbound_[i] = 1
		# Ensure velocity does not exceed the max speed
		for i in range(self.indexes.vx, self.indexes.vy+lookahead_step_num-1):
			x_lowerbound_[i] = -max_velocity
			x_upperbound_[i] = max_velocity
		
		#Constrain initial and final position
		x_lowerbound_[self.indexes.px] = state[0]
		x_lowerbound_[self.indexes.py] = state[1]
		x_lowerbound_[self.indexes.px+lookahead_step_num-1] = end_point[0]
		x_lowerbound_[self.indexes.py+lookahead_step_num-1] = end_point[1]
		x_upperbound_[self.indexes.px] = state[0]
		x_upperbound_[self.indexes.py] = state[1]
		x_upperbound_[self.indexes.px+lookahead_step_num-1] = end_point[0]
		x_upperbound_[self.indexes.py+lookahead_step_num-1] = end_point[1]

		#Constrain initial and final velocity
		x_lowerbound_[self.indexes.vx] = 0
		x_lowerbound_[self.indexes.vy] = 0
		x_lowerbound_[self.indexes.vx+lookahead_step_num-2] = 0
		x_lowerbound_[self.indexes.vy+lookahead_step_num-2] = 0
		x_upperbound_[self.indexes.vx] = 0
		x_upperbound_[self.indexes.vy] = 0
		x_upperbound_[self.indexes.vx+lookahead_step_num-2] = 0
		x_upperbound_[self.indexes.vy+lookahead_step_num-2] = 0

		x_lowerbound_[self.indexes.dt] = time_step_min
		x_upperbound_[self.indexes.dt] = time_step_max

		# Define lowerbound and upperbound of g constraints
		g_lowerbound_ = [exp(-10)] * self.num_of_g_
		g_upperbound_ = [exp(10)] * self.num_of_g_

		# Initialize g constraints list with SX elements
		g = [SX(0)] * self.num_of_g_
	
		g_index = 0
		
		# Add speed constraints
		for i in range(lookahead_step_num - 1):
			curr_vx_index = self.indexes.vx + i
			curr_vy_index = self.indexes.vy + i
			vx = x[curr_vx_index]
			vy = x[curr_vy_index]

			# Constraint on velocity magnitude
			g[g_index] = vx**2 + vy**2
			g_lowerbound_[g_index] = 0  # Minimum speed (non-negative)
			g_upperbound_[g_index] = max_velocity**2  # Maximum speed in any direction
			g_index += 1

		# Add acceleration magnitude constraints
		for i in range(lookahead_step_num - 2):
			curr_vx_index = self.indexes.vx + i
			curr_vy_index = self.indexes.vy + i
			next_vx_index = curr_vx_index + 1
			next_vy_index = curr_vy_index + 1

			ax = (x[next_vx_index] - x[curr_vx_index]) / time_step
			ay = (x[next_vy_index] - x[curr_vy_index]) / time_step

			# Constraint on acceleration magnitude
			g[g_index] = ax**2 + ay**2
			g_lowerbound_[g_index] = 0  # Minimum acceleration (non-negative)
			g_upperbound_[g_index] = max_accel**2  # Maximum acceleration in any direction
			g_index += 1

		# Update position constraints as previously defined
		for i in range(lookahead_step_num-1):
			curr_px_index = i + self.indexes.px
			curr_py_index = i + self.indexes.py
			curr_vx_index = i + self.indexes.vx
			curr_vy_index = i + self.indexes.vy

			curr_px = x[curr_px_index]
			curr_py = x[curr_py_index]
			curr_vx = x[curr_vx_index]
			curr_vy = x[curr_vy_index]

			next_px = x[1 + curr_px_index]
			next_py = x[1 + curr_py_index]

			next_m_px = curr_px + curr_vx * time_step
			next_m_py = curr_py + curr_vy * time_step

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
			curr_px_index = i + self.indexes.px
			curr_py_index = i + self.indexes.py

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
		"""
		# Extract the optimized trajectory from the result
		x_opt = sol['x'].full().flatten()
		
		# Print header
		print(f"{'Step':<5} {'Position (x, y)':<20}\t{'Velocity (vx, vy)':<20}\t{'Acceleration (ax, ay)':<25}")
		print("-" * 70)

		lemlib_output_string = ""

		optimized_time_step  = x_opt[self.num_of_x_-1]
		
		# Loop through each step in the trajectory
		for i in range(lookahead_step_num):
			# Position at step i
			px = x_opt[self.indexes.px + i]
			py = x_opt[self.indexes.py + i]

			if i < lookahead_step_num - 1:  # Ensure we don’t go out of bounds
				vx = x_opt[self.indexes.vx + i]
				vy = x_opt[self.indexes.vy + i]
			else:
				vx = vy = 0  # No velocity at the last step

			# Acceleration between step i and i+1
			if i < lookahead_step_num - 2:  # Ensure we don’t go out of bounds
				next_vx = x_opt[self.indexes.vx + i + 1]
				next_vy = x_opt[self.indexes.vy + i + 1]
				ax = (next_vx - vx) / optimized_time_step 
				ay = (next_vy - vy) / optimized_time_step 
			else:
				ax = ay = 0  # No acceleration at the last step

			# Print the details for this step
			print(f"{i:<5} ({px:.2f}, {py:.2f})\t\t({vx:.2f}, {vy:.2f})\t\t({ax:.2f}, {ay:.2f})")

			speed = sqrt(vx*vx+vy*vy)

			lemlib_output_string += f"{px*144:.3f}, {py*144:.3f}, {speed*144:.3f}\n"
		
		print("")
		print(f"Time step: {optimized_time_step:.2f}")
		print(f"Path time: {optimized_time_step * lookahead_step_num:.2f}")
		lemlib_output_string += "endData"

		file = open('path_output.txt', 'w')
		file.write(lemlib_output_string)
		file.close()
	
	def plotResults(self, sol):
		# Create a figure with a flexible window size
		fig, ax = plt.subplots(figsize=(8, 8))  # The graph itself will remain square

		# Convert CasADi matrices to NumPy arrays
		planned_px = np.array(sol['x'][self.indexes.px:self.indexes.py]).flatten()
		planned_py = np.array(sol['x'][self.indexes.py:self.indexes.vx]).flatten()
		planned_vx = np.array(sol['x'][self.indexes.vx:self.indexes.vy]).flatten()  # x-velocity
		planned_vy = np.array(sol['x'][self.indexes.vy:self.indexes.dt]).flatten()  # y-velocity

		planned_theta = np.arctan2(planned_vy, planned_vx)
		# Makes sure the heading for the start and end point matches that of the second and second-to-last point
		planned_theta = np.concatenate(([planned_theta[1]],planned_theta[1:-1],[planned_theta[-2],planned_theta[-2]]))

		# Plot the planned path
		ax.plot(planned_px, planned_py, 'o-', label='path')

		# Define theta for circles
		theta_list = np.linspace(0, 2 * np.pi, 100)

		# Draw the robot's radius at each planned point
		num_outlines = 3
		mod = round(lookahead_step_num / (num_outlines - 1))
		index = 0
		for px, py, theta in zip(planned_px, planned_py, planned_theta):
			rotation = Affine2D().rotate_around(px, py, theta) 
			rectangle = plt.Rectangle(
				(px - robot_length / 2, py - robot_width / 2),  # Bottom-left corner
				robot_length,  # Width of the rectangle
				robot_width,   # Height of the rectangle
				edgecolor='g',
				facecolor='none',
				alpha=0.5
			)
			rectangle.set_transform(rotation + ax.transData)

			if(index % mod == 0 or index == lookahead_step_num - 1):
				robot_circle_x = px + robot_radius * np.cos(theta_list)
				robot_circle_y = py + robot_radius * np.sin(theta_list)
				ax.plot(robot_circle_x, robot_circle_y, 'g--', alpha=0.5, label='robot radius' if index == 0 else None)
				ax.add_patch(rectangle)
			index += 1
		
		# Plot start and end points
		ax.plot(start_point[0], start_point[1], 'o', label='start')
		ax.plot(end_point[0], end_point[1], 'o', label='target')

		# Plot obstacles
		first_obstacle = True
		for obstacle in obstacles:
			danger_x = obstacle[0] + (obstacle_radius - 0.005) * np.cos(theta_list)
			danger_y = obstacle[1] + (obstacle_radius - 0.005) * np.sin(theta_list)
			if first_obstacle:
				ax.plot(danger_x, danger_y, 'r-', label='obstacle')
				first_obstacle = False
			else:
				ax.plot(danger_x, danger_y, 'r-')

		# Plot the circle in the middle of the graph with radius 1/6
		radius = 1 / 6
		center_x, center_y = 0.5, 0.5
		circle_x = center_x + radius * np.cos(theta_list)
		circle_y = center_y + radius * np.sin(theta_list)
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
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import random
from math import sqrt, exp  # Added missing imports if needed

NUM_OF_ACTS = 2   # Number of MPC actions (vx, vy)
NUM_OF_STATES = 2  # Number of MPC states (px, py)

lookahead_step_num = 30
lookahead_step_timeinterval = 0.1

max_time = 10  # Maximum time for the path
min_time = 0   # Minimum time for the path
time_step_min = min_time/lookahead_step_num  # Minimum time step
time_step_max = max_time/lookahead_step_num   # Maximum time step

start_point = [24/144, 24/144]
end_point = [120/144, 120/144]

robot_length = 15/144
robot_width = 15/144
buffer_radius = 0/144
robot_radius = sqrt(robot_length**2 + robot_width**2) / 2 + buffer_radius

max_velocity = 70/144
max_accel = 70/144
max_power = 100

center_circle_radius = 1/6+3.5/144

# =============================================================================
# Obstacle Definitions
# =============================================================================
class Obstacle:
    def __init__(self, x, y, r, i):
        self.x = x
        self.y = y
        self.radius = r
        self.ignore_collision = i

obstacles = [Obstacle(3/6, 2/6, 3.5/144, False),
             Obstacle(3/6, 4/6, 3.5/144, False),
             Obstacle(2/6, 3/6, 3.5/144, False),
             Obstacle(4/6, 3/6, 3.5/144, False)]
for i in range(5):
    obstacles.append(Obstacle(random.uniform(.1, .9), random.uniform(.1, .9), 5.75/144, False))

class FirstStateIndex:
    def __init__(self, n):
        self.px = 0
        self.py = self.px + n
        self.vx = self.py + n
        self.vy = self.vx + n - 1
        self.dt = self.vy + n - 1

# =============================================================================
# MPC Class Definitions
# =============================================================================
class MPC:
    def __init__(self):
        # -------------------------------------------------------------------------
        # Initialization: Define indexes and problem dimensions
        # -------------------------------------------------------------------------
        self.indexes = FirstStateIndex(lookahead_step_num)
        self.num_of_x_ = (lookahead_step_num)*NUM_OF_STATES + (lookahead_step_num - 1)*NUM_OF_ACTS + 1  # plus one for time step variable
        self.num_of_g_ = (lookahead_step_num)*len(obstacles)  + (lookahead_step_num-1)*(NUM_OF_ACTS+1) + (lookahead_step_num - 2)
    
    def intersects(self, x1, y1, x2, y2, r):
        # ...existing code...
        if(x1**2 + y1**2 < r**2 or x2**2 + y2**2 < r**2):
            return False  # Prevent error when start or end is inside the circle
        m = (y2 - y1) / (x2 - x1)
        return 4*((m**2+1)*r**2 - (y1-m*x1)**2) > 0

    def get_initial_path(self, x1, y1, x2, y2, r):
        # -------------------------------------------------------------------------
        # Path Initialization: Compute initial guess for the path
        # -------------------------------------------------------------------------
        x1 = x1 - 0.5; y1 = y1 - 0.5; x2 = x2 - 0.5; y2 = y2 - 0.5
        if self.intersects(x1, y1, x2, y2, r):
            start1 = 2*np.arctan((y1-sqrt(-r**2+x1**2+y1**2))/(x1+r))
            start2 = 2*np.arctan((y1+sqrt(-r**2+x1**2+y1**2))/(x1+r))
            end1 = 2*np.arctan((y2-sqrt(-r**2+x2**2+y2**2))/(x2+r))
            end2 = 2*np.arctan((y2+sqrt(-r**2+x2**2+y2**2))/(x2+r))
            x3 = r*np.cos(start1); y3 = r*np.sin(start1)
            x4 = r*np.cos(start2); y4 = r*np.sin(start2)
            x5 = r*np.cos(end1);   y5 = r*np.sin(end1)
            x6 = r*np.cos(end2);   y6 = r*np.sin(end2)
            m3 = (y3-y1)/(x3-x1)
            m4 = (y4-y1)/(x4-x1)
            m5 = (y5-y2)/(x5-x2)
            m6 = (y6-y2)/(x6-x2)
            x7 = (m3*x1 - m6*x2 - y1 + y2)/(m3-m6)
            y7 = (m3*(m6*(x1-x2)+y2)-m6*y1)/(m3-m6)
            x8 = (m4*x1 - m5*x2 - y1 + y2)/(m4-m5)
            y8 = (m4*(m5*(x1-x2)+y2)-m5*y1)/(m4-m5)
            d1 = sqrt((x7-x1)**2 + (y7-y1)**2) + sqrt((x7-x2)**2 + (y7-y2)**2)
            d2 = sqrt((x8-x1)**2 + (y8-y1)**2) + sqrt((x8-x2)**2 + (y8-y2)**2)
            if d1 < d2:
                init_x = np.linspace(x1+0.5, x7+0.5, lookahead_step_num//2)
                init_y = np.linspace(y1+0.5, y7+0.5, lookahead_step_num//2)
                init_x2 = np.linspace(x7+0.5, x2+0.5, lookahead_step_num//2)
                init_y2 = np.linspace(y7+0.5, y2+0.5, lookahead_step_num//2)
            else:
                init_x = np.linspace(x1+0.5, x8+0.5, lookahead_step_num//2)
                init_y = np.linspace(y1+0.5, y8+0.5, lookahead_step_num//2)
                init_x2 = np.linspace(x8+0.5, x2+0.5, lookahead_step_num//2)
                init_y2 = np.linspace(y8+0.5, y2+0.5, lookahead_step_num//2)
            init_x = np.concatenate((init_x, init_x2))
            init_y = np.concatenate((init_y, init_y2))
        else:
            init_x = np.linspace(x1+0.5, x2+0.5, lookahead_step_num)
            init_y = np.linspace(y1+0.5, y2+0.5, lookahead_step_num)
        return (init_x, init_y)

    def Solve(self, state):
        # -------------------------------------------------------------------------
        # Solve Optimization Problem: Set up variables, constraints, and solve NLP
        # -------------------------------------------------------------------------
        x = SX.sym('x', self.num_of_x_)
        self.indexes.dt = self.num_of_x_ - 1

        w_time_step = 100.0  # Cost weight on time step
        cost = 0.0

        # Build initial guess using an initial path and zero velocity
        init_x, init_y = self.get_initial_path(state[0], state[1], end_point[0], end_point[1], center_circle_radius)
        self.init_x = init_x  # For plotting
        self.init_y = init_y

        init_v = [0] * ((lookahead_step_num - 1) * NUM_OF_ACTS)
        init_time_step = lookahead_step_timeinterval 
        x_ = np.concatenate((init_x, init_y, init_v, [init_time_step]))

        time_step = x[self.indexes.dt]
        cost += w_time_step * time_step * lookahead_step_num

        # Define variable bounds
        x_lowerbound_ = [-exp(10)] * self.num_of_x_
        x_upperbound_ = [exp(10)] * self.num_of_x_

        for i in range(self.indexes.px, self.indexes.py + lookahead_step_num):
            x_lowerbound_[i] = 0 + robot_radius
            x_upperbound_[i] = 1 - robot_radius
        for i in range(self.indexes.vx, self.indexes.vy + lookahead_step_num - 1):
            x_lowerbound_[i] = -max_velocity
            x_upperbound_[i] = max_velocity

        # Constrain start and goal positions & velocities
        x_lowerbound_[self.indexes.px] = state[0]
        x_lowerbound_[self.indexes.py] = state[1]
        x_lowerbound_[self.indexes.px + lookahead_step_num - 1] = end_point[0]
        x_lowerbound_[self.indexes.py + lookahead_step_num - 1] = end_point[1]
        x_upperbound_[self.indexes.px] = state[0]
        x_upperbound_[self.indexes.py] = state[1]
        x_upperbound_[self.indexes.px + lookahead_step_num - 1] = end_point[0]
        x_upperbound_[self.indexes.py + lookahead_step_num - 1] = end_point[1]

        x_lowerbound_[self.indexes.vx + lookahead_step_num - 2] = 0
        x_lowerbound_[self.indexes.vy + lookahead_step_num - 2] = 0
        x_upperbound_[self.indexes.vx + lookahead_step_num - 2] = 0
        x_upperbound_[self.indexes.vy + lookahead_step_num - 2] = 0

        x_lowerbound_[self.indexes.dt] = time_step_min
        x_upperbound_[self.indexes.dt] = time_step_max

        g_lowerbound_ = [exp(-10)] * self.num_of_g_
        g_upperbound_ = [exp(10)] * self.num_of_g_

        g = [SX(0)] * self.num_of_g_
        g_index = 0

        # Speed constraints
        for i in range(lookahead_step_num - 1):
            curr_vx_index = self.indexes.vx + i
            curr_vy_index = self.indexes.vy + i
            vx = x[curr_vx_index]
            vy = x[curr_vy_index]
            g[g_index] = vx**2 + vy**2
            g_lowerbound_[g_index] = 0
            g_upperbound_[g_index] = max_velocity**2
            g_index += 1
        
        # Acceleration constraints
        for i in range(lookahead_step_num - 2):
            curr_vx_index = self.indexes.vx + i
            curr_vy_index = self.indexes.vy + i
            next_vx_index = curr_vx_index + 1
            next_vy_index = curr_vy_index + 1
            ax = (x[next_vx_index] - x[curr_vx_index]) / time_step
            ay = (x[next_vy_index] - x[curr_vy_index]) / time_step
            g[g_index] = ax**2 + ay**2
            g_lowerbound_[g_index] = 0
            g_upperbound_[g_index] = max_accel**2
            g_index += 1

        # Dynamics (position update) constraints
        for i in range(lookahead_step_num - 1):
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
            g[g_index] = next_px - next_m_px
            g_lowerbound_[g_index] = 0; g_upperbound_[g_index] = 0
            g_index += 1
            g[g_index] = next_py - next_m_py
            g_lowerbound_[g_index] = 0; g_upperbound_[g_index] = 0
            g_index += 1

        # Obstacle constraints
        for i in range(lookahead_step_num):
            curr_px_index = i + self.indexes.px
            curr_py_index = i + self.indexes.py
            curr_px = x[curr_px_index]
            curr_py = x[curr_py_index]
            for obstacle in obstacles:
                g[g_index] = (curr_px - obstacle.x)**2 + (curr_py - obstacle.y)**2
                if obstacle.ignore_collision:
                    g_lowerbound_[g_index] = exp(-10)
                else:
                    g_lowerbound_[g_index] = (obstacle.radius + robot_radius)**2
                g_upperbound_[g_index] = exp(10)
                g_index += 1

        nlp = {'x': x, 'f': cost, 'g': vertcat(*g)}
        opts = {"ipopt.print_level": 3, "print_time": 0, 'ipopt.tol': 1e-6}
        solver = nlpsol('solver', 'ipopt', nlp, opts)
        res = solver(x0=x_, lbx=x_lowerbound_, ubx=x_upperbound_, lbg=g_lowerbound_, ubg=g_upperbound_)
        self.status = solver.stats()['return_status']
        return res

    def print_trajectory_details(self, res):
        # -------------------------------------------------------------------------
        # Trajectory Output: Print details and save to a file
        # -------------------------------------------------------------------------
        x_opt = res['x'].full().flatten()
        final_cost = res['f'].full().item()
        print(f"{'Step':<5} {'Position (x, y)':<20}\t{'Velocity (vx, vy)':<20}\t{'Acceleration (ax, ay)':<25}")
        print("-" * 70)
        lemlib_output_string = ""
        optimized_time_step  = x_opt[self.num_of_x_-1]
        for i in range(lookahead_step_num):
            px = x_opt[self.indexes.px + i]
            py = x_opt[self.indexes.py + i]
            if i < lookahead_step_num - 1:
                vx = x_opt[self.indexes.vx + i]
                vy = x_opt[self.indexes.vy + i]
            else:
                vx = vy = 0
            if i < lookahead_step_num - 2:
                next_vx = x_opt[self.indexes.vx + i + 1]
                next_vy = x_opt[self.indexes.vy + i + 1]
                ax = (next_vx - vx) / optimized_time_step 
                ay = (next_vy - vy) / optimized_time_step 
            else:
                ax = ay = 0
            print(f"{i:<5} ({px*144-72:.2f}, {py*144-72:.2f})\t\t({vx*144:.2f}, {vy*144:.2f})\t\t({ax*144:.2f}, {ay*144:.2f})")
            speed = (sqrt(vx*vx+vy*vy)/max_velocity*max_power)
            lemlib_output_string += f"{px*144-72:.3f}, {py*144-72:.3f}, {speed:.3f}\n"
        print(f"\nFinal cost: {final_cost:.2f}")
        print(f"\nTime step: {optimized_time_step:.2f}")
        print(f"Path time: {optimized_time_step * lookahead_step_num:.2f}")
        print(f"\nStatus: {self.status}")
        lemlib_output_string += "endData"
        with open('path_output.txt', 'w') as file:
            file.write(lemlib_output_string)

    def plotResults(self, sol):
        # -------------------------------------------------------------------------
        # Plot Results: Display trajectory, obstacles, and robot boundaries
        # -------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 8))
        planned_px = np.array(sol['x'][self.indexes.px:self.indexes.py]).flatten()
        planned_py = np.array(sol['x'][self.indexes.py:self.indexes.vx]).flatten()
        planned_vx = np.array(sol['x'][self.indexes.vx:self.indexes.vy]).flatten()
        planned_vy = np.array(sol['x'][self.indexes.vy:self.indexes.dt]).flatten()
        planned_theta = np.arctan2(planned_vy, planned_vx)
        planned_theta = np.concatenate(([planned_theta[1]], planned_theta[1:-1], [planned_theta[-2], planned_theta[-2]]))
        ax.plot(self.init_x, self.init_y, linestyle=':', color='gray', alpha=0.7, label='initial path')
        ax.plot(planned_px, planned_py, '-o', label='path', color="blue", alpha=0.5)
        theta_list = np.linspace(0, 2 * np.pi, 100)
        num_outlines = 3
        mod = round(lookahead_step_num / (num_outlines - 1))
        index = 0
        for px, py, theta in zip(planned_px, planned_py, planned_theta):
            rotation = Affine2D().rotate_around(px, py, theta)
            rectangle = plt.Rectangle((px - robot_length / 2, py - robot_width / 2), robot_length, robot_width,
                                      edgecolor='blue', facecolor='none', alpha=1)
            rectangle.set_transform(rotation + ax.transData)
            if index % mod == 0 or index == lookahead_step_num - 1:
                robot_circle_x = px + robot_radius * np.cos(theta_list)
                robot_circle_y = py + robot_radius * np.sin(theta_list)
                ax.plot(robot_circle_x, robot_circle_y, '--', color='blue', alpha=0.5, label='robot radius' if index == 0 else None)
                ax.add_patch(rectangle)
            index += 1
        ax.plot(start_point[0], start_point[1], 'o', color='orange', label='start')
        ax.plot(end_point[0], end_point[1], 'o', color='green', label='target')
        first_obstacle = True
        for obstacle in obstacles:
            danger_x = obstacle.x + (obstacle.radius - 0.005) * np.cos(theta_list)
            danger_y = obstacle.y + (obstacle.radius - 0.005) * np.sin(theta_list)
            if first_obstacle:
                ax.plot(danger_x, danger_y, 'r-', label='obstacle')
                first_obstacle = False
            else:
                ax.plot(danger_x, danger_y, 'r-')
        radius = center_circle_radius; center_x, center_y = 0.5, 0.5
        circle_x = center_x + radius * np.cos(theta_list)
        circle_y = center_y + radius * np.sin(theta_list)
        ax.plot(circle_x, circle_y, 'r--', alpha=0.25)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0., frameon=False)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid()
        plt.show()

# =============================================================================
# Main Execution
# =============================================================================
mpc_ = MPC()
sol = mpc_.Solve(start_point)
mpc_.print_trajectory_details(sol)
mpc_.plotResults(sol)
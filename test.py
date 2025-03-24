import numpy as np
import matplotlib.pyplot as plt


l1 =1
l2 = 1

def solve_ik(x, y):
    cos_q2 = (x*x + y*y - l1*l1 - l2*l2) / (2 * l1 * l2)
    q2 = np.arccos(cos_q2)
    
    # Calculate q1
    beta = np.arctan2(y, x)
    psi = np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    q1 = beta - psi
    
    return q1, q2

# Parameters
num_points = 200
angle = np.linspace(0, 2 * np.pi, num_points) 
radius = 1.2  
wave_amplitude = 0.3  
wave_frequency = 6 

traj = []
for i in angle:
    wavy_radius = radius + wave_amplitude * np.sin(wave_frequency * i)
    traj.append([wavy_radius * np.cos(i), wavy_radius * np.sin(i)])


desired_traj = traj.copy()

robot_positions = []
for i in range(len(desired_traj)):
    robot_positions.append(solve_ik(desired_traj[i][0], desired_traj[i][1]))


plt.ion()
plt.show()

for ind in range(len(robot_positions)-1):  # Adjusted range to match available trajectory points
    theta_1_i = robot_positions[ind][0]
    theta_2_i = robot_positions[ind][1]
    
    x0 = 0
    y0 = 0
    x1 = np.round(l1 * np.cos(theta_1_i), 2)
    y1 = np.round(l1 * np.sin(theta_1_i), 2)
    x2 = np.round(l1 * np.cos(theta_1_i) + l2 * np.cos(theta_1_i + theta_2_i), 2)
    y2 = np.round(l1 * np.sin(theta_1_i) + l2 * np.sin(theta_1_i + theta_2_i), 2)
    
    plt.clf()
    
    if ind < len(desired_traj):
        plt.scatter(desired_traj[ind][0], desired_traj[ind][1], color='red')
    
    plt.plot([x0, x1], [y0, y1], color='blue', linewidth=5)
    plt.plot([x1, x2], [y1, y2], color='blue', linewidth=3)
    plt.xlim([-2.5, 2.5])
    plt.ylim([-2.5, 2.5])
    plt.pause(0.02)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import keras


r1 = 0.2  
r2 = 0.15  

model = keras.models.load_model('real_robot_model.keras')

def circle_3d():
    num_points = 400
    t = np.linspace(0, 4*np.pi, num_points)
    radius = 0.4
    height_variation = 0.5
    traj = []
    for i in t:
        x = radius * np.cos(i)
        y = radius * np.sin(i)
        z = height_variation * np.sin(i/2)
        traj.append([x, y, z])
    return traj
    
def first_quadrant_circle():
    """Generate a circle in the first quadrant of the XY plane (z=0)"""
    num_points = 200
    t = np.linspace(0, 2*np.pi, num_points)
    radius = 0.05
    center_x, center_y = 0.05, 0.24  
    traj = []
    for i in t:
        x = center_x + radius * np.cos(i)
        y = center_y + radius * np.sin(i)
        z = 0.1  
        traj.append([x, y, z])
    return traj

def spiral_3d():
    num_points = 200
    t = np.linspace(0, 6 * np.pi, num_points)
    radius_growth = 0.05
    traj = []
    
    for i in t:
        r = 0.2 + radius_growth * i
        x = r * np.cos(i)
        y = r * np.sin(i)
        z = 0.1 * i  
        traj.append([x, y, z])
        
    return traj

def pick_and_place():
    """Simulates a pick and place operation"""
    num_points = 300
    traj = []
    
    
    approach_points = num_points // 6
    for i in range(approach_points):
        t = i / approach_points
        x = t * 1.5
        y = 0
        z = 0.5 * (1 - t)  
        traj.append([x, y, z])
    
    
    pause_points = num_points // 12
    for i in range(pause_points):
        traj.append([1.5, 0, 0])
    
    
    lift_points = num_points // 6
    for i in range(lift_points):
        t = i / lift_points
        x = 1.5
        y = 0
        z = t * 0.7  
        traj.append([x, y, z])
    
    
    transport_points = num_points // 4
    for i in range(transport_points):
        t = i / transport_points
        x = 1.5 - t * 3  
        y = t * 1.0  
        z = 0.7  
        traj.append([x, y, z])
    
    
    lower_points = num_points // 6
    for i in range(lower_points):
        t = i / lower_points
        x = -1.5
        y = 1.0
        z = 0.7 * (1 - t)  
        traj.append([x, y, z])
    
    
    return_points = num_points - approach_points - pause_points - lift_points - transport_points - lower_points
    for i in range(return_points):
        t = i / return_points
        x = -1.5 + t * 1.5  
        y = 1.0 - t * 1.0  
        z = t * 0.5  
        traj.append([x, y, z])
    
    return traj

def figure_eight_3d():
    num_points = 200
    t = np.linspace(0, 2 * np.pi, num_points)
    a = 1.0  
    b = 0.5  
    traj = []
    for i in t:
        x = a * np.sin(i)
        y = b * np.sin(2*i)
        z = 0.3 + 0.2 * np.cos(3*i)  
        traj.append([x, y, z])
    return traj


desired_traj = first_quadrant_circle()  


robot_positions = []


prev_q1, prev_q2, prev_q3 = 0.0, 0.0, 0.0  

for point in desired_traj:
    x, y, z = point
    
    
    model_input = np.array([[x, y, z, prev_q1, prev_q2, prev_q3]])
    
    
    prediction = model.predict(model_input, verbose=0)
    q1, q2, q3 = prediction[0]
    
    robot_positions.append([q1, q2, q3])
    
    
    prev_q1, prev_q2, prev_q3 = q1, q2, q3


fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
plt.ion()  


actual_x = []
actual_y = []
actual_z = []

error_list = list()

for ind in range(len(robot_positions)):
    
    q1 = robot_positions[ind][0]  
    q2 = robot_positions[ind][1]  
    q3 = robot_positions[ind][2]  
    
    
    
    x0, y0, z0 = 0, 0, 0
    
    
    j1_x, j1_y, j1_z = 0, 0, 0
    
    
    q2_offset = 0.34906585
    q3_offset = 0.785398163

    x1 = r1 * np.cos(q2 + q2_offset)
    y1 = r1 * np.sin(q2 + q2_offset) * np.cos(q1)
    z1 = r1 * np.sin(q2 + q2_offset) * np.sin(q1)
    
    
    x2 = x1 + r2 * np.cos(q2 + q2_offset + np.pi - q3 - q3_offset)
    y2 = y1 + r2 * np.sin(q2 + q2_offset + np.pi - q3 - q3_offset) * np.cos(q1)
    z2 = z1 + r2 * np.sin(q2 + q2_offset + np.pi - q3 - q3_offset) * np.sin(q1)
    
    
    actual_x.append(x2)
    actual_y.append(y2)
    actual_z.append(z2)
    
    
    ax.clear()
    
    
    target_x = [pos[0] for pos in desired_traj]
    target_y = [pos[1] for pos in desired_traj]
    target_z = [pos[2] for pos in desired_traj]
    ax.plot(target_x, target_y, target_z, 'r--', alpha=0.5, label='Desired Path')
    
    
    ax.plot(actual_x, actual_y, actual_z, 'g-', alpha=0.7, label='Actual Path')
    
    
    if ind < len(desired_traj):
        ax.scatter(desired_traj[ind][0], desired_traj[ind][1], desired_traj[ind][2], 
                  color='red', s=50, zorder=5)
    
    
    axis_length = 0.3
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, label='X axis')
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, label='Y axis')
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, label='Z axis')
    
    
    
    ax.plot([x0, j1_x], [y0, j1_y], [z0, j1_z], 'k-', linewidth=3)
    
    
    ax.plot([j1_x, x1], [j1_y, y1], [j1_z, z1], 'navy', linewidth=5, label='Link 1')
    
    
    ax.plot([x1, x2], [y1, y2], [z1, z2], 'blue', linewidth=5, label='Link 2')
    
    
    ax.scatter([j1_x, x1, x2], [j1_y, y1, y2], [j1_z, z1, z2], 
              color=['black', 'navy', 'blue'], s=[80, 60, 60])
    
    
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(f'3DOF Robot Arm - Frame {ind+1}/{len(robot_positions)}')
    
    
    if ind == 0:
        ax.legend(loc='upper right')
    
    
    angle_info = f'Joint 1 (q1): {q1:.2f} rad\nJoint 2 (q2): {q2:.2f} rad\nJoint 3 (q3): {q3:.2f} rad'
    ax.text2D(0.05, 0.05, angle_info, transform=ax.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    if ind < len(desired_traj):
        target_pos = desired_traj[ind]
        actual_pos = (x2, y2, z2)
        error = np.sqrt((target_pos[0]-actual_pos[0])**2 + 
                         (target_pos[1]-actual_pos[1])**2 + 
                         (target_pos[2]-actual_pos[2])**2)
        error_list.append(error)
        error_info = f'Position Error: {error:.4f}'
        ax.text2D(0.75, 0.05, error_info, transform=ax.transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    plt.pause(0.02)

plt.ioff()
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(error_list)
plt.grid(True, alpha=0.3)
plt.title("Error")
plt.xlabel("Time sample")
plt.ylabel("Error value [rad]")
plt.show()

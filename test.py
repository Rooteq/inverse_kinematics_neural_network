import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

l1 = 1
l2 = 1

model = keras.models.load_model('robot_kinematics_rnn_model.keras')

def first_quadrant_circle():
    num_points = 200
    angle = np.linspace(0, 2*np.pi, num_points)  
    radius = 0.4
    traj = []
    for i in angle:
        traj.append([1 + radius * np.cos(i), 1 + radius * np.sin(i)])
    return traj


def figure_eight():
    num_points = 200
    t = np.linspace(0, 2 * np.pi, num_points)
    a = 0.6  
    b = 0.4  
    traj = []
    for i in t:
        x = 0.6 + a * np.sin(i)
        y = 0.6 + b * np.sin(2*i)
        traj.append([x, y])
    return traj

def square_trajectory():
    num_points = 200
    side_len = 0.2
    points_per_side = num_points // 4
    
    traj = []
    
    for i in range(points_per_side):
        x = 1 + side_len - (2 * side_len * i / points_per_side)
        y = 1 -side_len
        traj.append([x, y])
    
    
    for i in range(points_per_side):
        x = 1 -side_len
        y = 1 -side_len + (2 * side_len * i / points_per_side)
        traj.append([x, y])
    
    
    for i in range(points_per_side):
        x = 1 - side_len + (2 * side_len * i / points_per_side)
        y = 1 + side_len
        traj.append([x, y])
    
    
    for i in range(points_per_side):
        x = 1 + side_len
        y = 1 + side_len - (2 * side_len * i / points_per_side)
        traj.append([x, y])
    
    return traj


def spiral():
    num_points = 200
    t = np.linspace(0, 6 * np.pi, num_points)
    radius_growth = 0.05
    traj = []
    for i in t:
        r = 0.2 + radius_growth * i
        x = r * np.cos(i)
        y = r * np.sin(i)
        traj.append([x, y])
    return traj



num_points = 200
angle = np.linspace(0, 2 * np.pi, num_points)
radius = 1.2
wave_amplitude = 0.3
wave_frequency = 6
traj = []
for i in angle:
    wavy_radius = radius + wave_amplitude * np.sin(wave_frequency * i)
    traj.append([wavy_radius * np.cos(i), wavy_radius * np.sin(i)])



desired_traj = first_quadrant_circle()


sequence_length = 5  
sequence = []
previous_joint_angles = []


for i in range(sequence_length):
    x, y = desired_traj[i]
    
    cos_q2 = (x*x + y*y - l1*l1 - l2*l2) / (2 * l1 * l2)
    q2 = np.arccos(cos_q2)
    beta = np.arctan2(y, x)
    psi = np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    q1 = beta - psi
    
    sequence.append([x, y, q1, q2])
    previous_joint_angles.append([q1, q2])


robot_positions = previous_joint_angles.copy()  

for i in range(sequence_length, len(desired_traj)):
    x, y = desired_traj[i]
    
    
    sequence.pop(0)  
    sequence.append([x, y, robot_positions[-1][0], robot_positions[-1][1]])
    
    
    model_input = np.array([sequence])
    
    
    prediction = model.predict(model_input, verbose=0)
    q1, q2 = prediction[0]
    
    
    robot_positions.append([q1, q2])
    
    
    sequence[-1][2] = q1
    sequence[-1][3] = q2


plt.ion()
plt.show()

for ind in range(len(robot_positions)-1):
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
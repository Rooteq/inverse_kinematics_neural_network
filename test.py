import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# Robot parameters
l1 = 1
l2 = 1

# Choose which method to use
use_neural_network = True  # Set to True to use neural network, False for analytical

# Load trained model if using neural network
if use_neural_network:
    model = keras.models.load_model('robot_kinematics_model.keras')

# Inverse kinematics function
def solve_ik(x, y):
    if use_neural_network:
        # Neural network approach
        # Assuming your model expects [x, y, prev_theta1, prev_theta2]
        global prev_theta1, prev_theta2
        input_data = np.array([[x, y, prev_theta1, prev_theta2]])
        prediction = model.predict(input_data, verbose=1)
        q1, q2 = prediction[0]
        
        # Update previous angles for next iteration
        prev_theta1, prev_theta2 = q1, q2
    else:
        # Analytical approach
        cos_q2 = (x*x + y*y - l1*l1 - l2*l2) / (2 * l1 * l2)
        q2 = np.arccos(cos_q2)
        # Calculate q1
        beta = np.arctan2(y, x)
        psi = np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        q1 = beta - psi
    
    return q1, q2

# Initialize previous angles for neural network method
prev_theta1, prev_theta2 = 0, 0

# Generate trajectory
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

# Solve IK for each point in the trajectory
robot_positions = []
for i in range(len(desired_traj)):
    robot_positions.append(solve_ik(desired_traj[i][0], desired_traj[i][1]))

# Visualization
plt.ion()
plt.show()

for ind in range(len(robot_positions)-1):
    theta_1_i = robot_positions[ind][0]
    theta_2_i = robot_positions[ind][1]
    
    # Forward kinematics to get end-effector position
    x0 = 0
    y0 = 0
    x1 = np.round(l1 * np.cos(theta_1_i), 2)
    y1 = np.round(l1 * np.sin(theta_1_i), 2)
    x2 = np.round(l1 * np.cos(theta_1_i) + l2 * np.cos(theta_1_i + theta_2_i), 2)
    y2 = np.round(l1 * np.sin(theta_1_i) + l2 * np.sin(theta_1_i + theta_2_i), 2)
    
    plt.clf()
    
    # Plot desired trajectory point
    if ind < len(desired_traj):
        plt.scatter(desired_traj[ind][0], desired_traj[ind][1], color='red')
    
    # Plot robot arm
    plt.plot([x0, x1], [y0, y1], color='blue', linewidth=5)
    plt.plot([x1, x2], [y1, y2], color='blue', linewidth=3)
    
    # Set plot limits
    plt.xlim([-2.5, 2.5])
    plt.ylim([-2.5, 2.5])
    plt.title(f"Robot Arm Trajectory - {'Neural Network' if use_neural_network else 'Analytical'}")
    plt.pause(0.02)
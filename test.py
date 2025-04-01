import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# Robot parameters
l1 = 1
l2 = 1

# Load the RNN model
model = keras.models.load_model('robot_kinematics_rnn_model.keras')

def first_quadrant_circle():
    num_points = 200
    angle = np.linspace(0, 2*np.pi, num_points)  # 0 to 90 degrees
    radius = 0.2
    traj = []
    for i in angle:
        traj.append([1 + radius * np.cos(i), 1 + radius * np.sin(i)])
    return traj

# Option 2: Figure-8 trajectory
def figure_eight():
    num_points = 200
    t = np.linspace(0, 2 * np.pi, num_points)
    a = 0.6  # horizontal amplitude
    b = 0.4  # vertical amplitude
    traj = []
    for i in t:
        x = 0.6 + a * np.sin(i)
        y = 0.6 + b * np.sin(2*i)
        traj.append([x, y])
    return traj

# Option 3: Square trajectory
def square_trajectory():
    num_points = 200
    side_len = 0.2
    points_per_side = num_points // 4
    
    traj = []
    # Bottom side (right to left)
    for i in range(points_per_side):
        x = 1 + side_len - (2 * side_len * i / points_per_side)
        y = 1 -side_len
        traj.append([x, y])
    
    # Left side (bottom to top)
    for i in range(points_per_side):
        x = 1 -side_len
        y = 1 -side_len + (2 * side_len * i / points_per_side)
        traj.append([x, y])
    
    # Top side (left to right)
    for i in range(points_per_side):
        x = 1 - side_len + (2 * side_len * i / points_per_side)
        y = 1 + side_len
        traj.append([x, y])
    
    # Right side (top to bottom)
    for i in range(points_per_side):
        x = 1 + side_len
        y = 1 + side_len - (2 * side_len * i / points_per_side)
        traj.append([x, y])
    
    return traj

# Option 4: Spiral trajectory
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
# desired_traj = traj.copy()

# Create a dictionary of trajectories
trajectories = {
    "First Quadrant Circle": first_quadrant_circle(),
    "Figure-8": figure_eight(),
    "Square": square_trajectory(),
    "Spiral": spiral(),
    "Wavy Circle": traj  # Your original trajectory
}

# Choose which trajectory to test
trajectory_name = "Spiral"  # Change this to test different trajectories
desired_traj = trajectories[trajectory_name]

# Initialize sequence with analytical IK for the first few points
sequence_length = 5  # Must match the training sequence length
sequence = []
previous_joint_angles = []

# Use analytical IK to initialize the sequence
for i in range(sequence_length):
    x, y = desired_traj[i]
    # Analytical IK
    cos_q2 = (x*x + y*y - l1*l1 - l2*l2) / (2 * l1 * l2)
    q2 = np.arccos(cos_q2)
    beta = np.arctan2(y, x)
    psi = np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    q1 = beta - psi
    
    sequence.append([x, y, q1, q2])
    previous_joint_angles.append([q1, q2])

# Solve IK using RNN for the rest of trajectory
robot_positions = previous_joint_angles.copy()  # Start with analytical solutions

for i in range(sequence_length, len(desired_traj)):
    x, y = desired_traj[i]
    
    # Update the sequence with the new position and previous angles
    sequence.pop(0)  # Remove oldest entry
    sequence.append([x, y, robot_positions[-1][0], robot_positions[-1][1]])
    
    # Convert sequence to model input format
    model_input = np.array([sequence])
    
    # Get prediction from RNN
    prediction = model.predict(model_input, verbose=0)
    q1, q2 = prediction[0]
    
    # Store the predicted joint angles
    robot_positions.append([q1, q2])
    
    # Update the sequence with the new prediction for next iteration
    sequence[-1][2] = q1
    sequence[-1][3] = q2

# Visualization code remains the same
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
    plt.title("Robot Arm Trajectory - RNN Model")
    plt.pause(0.02)
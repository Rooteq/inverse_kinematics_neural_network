import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# Link lengths
l1 = 1
l2 = 1

# Load the trained model
model = keras.models.load_model('robot_kinematics_rnn_model.keras')

# Trajectory generation functions
def first_quadrant_circle():
    num_points = 400
    angle = np.linspace(0, 4*np.pi, num_points)
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
        y = 1 - side_len
        traj.append([x, y])
        
    for i in range(points_per_side):
        x = 1 - side_len
        y = 1 - side_len + (2 * side_len * i / points_per_side)
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

def wavy_circle():
    num_points = 200
    angle = np.linspace(0, 2 * np.pi, num_points)
    radius = 1.2
    wave_amplitude = 0.3
    wave_frequency = 6
    traj = []
    
    for i in angle:
        wavy_radius = radius + wave_amplitude * np.sin(wave_frequency * i)
        traj.append([wavy_radius * np.cos(i), wavy_radius * np.sin(i)])
        
    return traj

# Select the desired trajectory
desired_traj = first_quadrant_circle()

# Check if the model is an RNN (requires sequence input)
is_rnn = any(isinstance(layer, keras.layers.RNN) for layer in model.layers)
sequence_length = 5  # If using an RNN model

# Generate joint angles using the neural network model
robot_positions = []

if is_rnn:
    # RNN approach - uses sequences for prediction
    # Initialize with zeros
    prev_q1, prev_q2 = 0.0, 0.0
    sequence = []
    
    # Build initial sequence
    for i in range(min(sequence_length, len(desired_traj))):
        x, y = desired_traj[i]
        sequence.append([x, y, prev_q1, prev_q2])
        
        # If we have a full sequence, make a prediction
        if len(sequence) == sequence_length:
            model_input = np.array([sequence])
            prediction = model.predict(model_input, verbose=0)
            prev_q1, prev_q2 = prediction[0]
            robot_positions.append([prev_q1, prev_q2])
            
            # Update the sequence with the predicted angles
            sequence[-1][2] = prev_q1
            sequence[-1][3] = prev_q2
    
    # Continue with the rest of the trajectory
    for i in range(sequence_length, len(desired_traj)):
        x, y = desired_traj[i]
        
        # Remove the oldest entry and add the new position with previous angles
        sequence.pop(0)
        sequence.append([x, y, prev_q1, prev_q2])
        
        # Make prediction
        model_input = np.array([sequence])
        prediction = model.predict(model_input, verbose=0)
        prev_q1, prev_q2 = prediction[0]
        robot_positions.append([prev_q1, prev_q2])
        
        # Update the sequence with the predicted angles
        sequence[-1][2] = prev_q1
        sequence[-1][3] = prev_q2
else:
    # Normal feedforward neural network approach
    for point in desired_traj:
        x, y = point
        
        # Create input for the model - format depends on what the model expects
        # Typically [x, y, prev_q1, prev_q2] for a 2DOF robot
        model_input = np.array([[x, y]])  # Adjust this format if needed
        
        # Make prediction
        prediction = model.predict(model_input, verbose=0)
        q1, q2 = prediction[0]
        
        robot_positions.append([q1, q2])

# Set up the visualization
plt.figure(figsize=(10, 8))
plt.ion()  # Interactive mode on

# Initialize lists to track the actual end effector path
actual_x = []
actual_y = []

# Animation loop
for ind in range(len(robot_positions)):
    # Get joint angles
    theta_1 = robot_positions[ind][0]
    theta_2 = robot_positions[ind][1]
    
    # Calculate joint positions
    x0, y0 = 0, 0  # Origin (robot base)
    x1 = l1 * np.cos(theta_1)
    y1 = l1 * np.sin(theta_1)
    x2 = x1 + l2 * np.cos(theta_1 + theta_2)
    y2 = y1 + l2 * np.sin(theta_1 + theta_2)
    
    # Store the end effector position
    actual_x.append(x2)
    actual_y.append(y2)
    
    # Clear the figure
    plt.clf()
    
    # Plot the target trajectory
    target_x = [pos[0] for pos in desired_traj]
    target_y = [pos[1] for pos in desired_traj]
    plt.plot(target_x, target_y, 'r--', alpha=0.5, label='Desired Path')
    
    # Plot the actual trajectory
    plt.plot(actual_x, actual_y, 'g-', alpha=0.7, label='Actual Path')
    
    # Highlight current target point
    if ind < len(desired_traj):
        plt.scatter(desired_traj[ind][0], desired_traj[ind][1], color='red', s=50, zorder=5)
    
    # Draw the robot
    plt.plot([x0, x1], [y0, y1], 'b-', linewidth=3, label='Link 1')
    plt.plot([x1, x2], [y1, y2], 'c-', linewidth=3, label='Link 2')
    
    # Draw joints
    plt.scatter([x0, x1, x2], [y0, y1, y2], color=['k', 'b', 'c'], s=[30, 20, 20])
    
    # Set plot properties
    plt.xlim([-2.5, 2.5])
    plt.ylim([-2.5, 2.5])
    plt.grid(True, alpha=0.3)
    plt.title(f'2DOF Robot Following Trajectory - Frame {ind+1}/{len(robot_positions)}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Only show legend on first frame to avoid redrawing it
    if ind == 0:
        plt.legend(loc='upper right')
    
    # Add joint angle information
    angle_info = f'Joint 1: {theta_1:.2f} rad\nJoint 2: {theta_2:.2f} rad'
    plt.annotate(angle_info, xy=(0.05, 0.05), xycoords='figure fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    # Add error information
    if ind < len(desired_traj):
        target_pos = desired_traj[ind]
        actual_pos = (x2, y2)
        error = np.sqrt((target_pos[0]-actual_pos[0])**2 + (target_pos[1]-actual_pos[1])**2)
        error_info = f'Error: {error:.4f}'
        plt.annotate(error_info, xy=(0.75, 0.05), xycoords='figure fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
    
    plt.pause(0.02)

# Keep the final plot visible
plt.ioff()
plt.show()
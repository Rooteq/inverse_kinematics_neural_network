import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import keras


MODEL_PATH = "3dof_model.keras"
L1 = 1.0
L2 = 1.0
L3 = 1.0


def generate_circle_trajectory(num_points=200):
    angle = np.linspace(0, 4 * np.pi, num_points)
    radius = 0.6
    trajectory = []
    for i in angle:
        trajectory.append([1.5 + radius * np.cos(i), 0.5 + radius * np.sin(i)])
    return trajectory


def forward_kinematics(theta_1, theta_2, theta_3):
    x0, y0 = 0.0, 0.0
    x1 = L1 * np.cos(theta_1)
    y1 = L1 * np.sin(theta_1)
    x2 = x1 + L2 * np.cos(theta_1 + theta_2)
    y2 = y1 + L2 * np.sin(theta_1 + theta_2)
    x3 = x2 + L3 * np.cos(theta_1 + theta_2 + theta_3)
    y3 = y2 + L3 * np.sin(theta_1 + theta_2 + theta_3)
    return (x0, y0), (x1, y1), (x2, y2), (x3, y3)


def predict_trajectory(model, desired_trajectory):
    model_inputs = []
    for pos in desired_trajectory:
        x, y = pos
        model_inputs.append([x, y, 0.0, 0.0, 0.0])
    
    robot_positions = []
    for i in range(len(model_inputs)):
        input_data = np.array([model_inputs[i]])
        prediction = model.predict(input_data, verbose=0)
        q1, q2, q3 = prediction[0]
        robot_positions.append([q1, q2, q3])
        
        if i < len(model_inputs) - 1:
            next_x, next_y = desired_trajectory[i + 1]
            model_inputs[i + 1] = [next_x, next_y, q1, q2, q3]
    
    return robot_positions


def calculate_error(desired_trajectory, robot_positions):
    errors = []
    for i, (desired_pos, joint_angles) in enumerate(zip(desired_trajectory, robot_positions)):
        _, _, _, (x3, y3) = forward_kinematics(*joint_angles)
        error = np.sqrt((desired_pos[0] - x3) ** 2 + (desired_pos[1] - y3) ** 2)
        errors.append(error)
    return errors


def animate_robot(desired_trajectory, robot_positions):
    plt.ion()
    plt.figure(figsize=(10, 8))
    
    actual_trajectory_x = []
    actual_trajectory_y = []
    
    for ind, joint_angles in enumerate(robot_positions):
        theta_1, theta_2, theta_3 = joint_angles
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = forward_kinematics(theta_1, theta_2, theta_3)
        
        actual_trajectory_x.append(x3)
        actual_trajectory_y.append(y3)
        
        plt.clf()
        
        desired_x = [point[0] for point in desired_trajectory]
        desired_y = [point[1] for point in desired_trajectory]
        plt.plot(desired_x, desired_y, "r--", alpha=0.5, label="Desired")
        plt.plot(actual_trajectory_x, actual_trajectory_y, "b-", alpha=0.7, label="Actual")
        
        if ind < len(desired_trajectory):
            plt.scatter(desired_trajectory[ind][0], desired_trajectory[ind][1], color="blue", s=50, alpha=0.5)
        
        plt.plot([x0, x1], [y0, y1], "b-", linewidth=4)
        plt.plot([x1, x2], [y1, y2], "g-", linewidth=3)
        plt.plot([x2, x3], [y2, y3], "r-", linewidth=2)
        
        plt.scatter([x0], [y0], color="black", s=60)
        plt.scatter([x1], [y1], color="blue", s=40)
        plt.scatter([x2], [y2], color="green", s=40)
        plt.scatter([x3], [y3], color="red", s=40)
        
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.grid(True, alpha=0.3)
        plt.title(f"3DOF Robot - Frame {ind + 1}/{len(robot_positions)}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc="upper right")
        
        if ind < len(desired_trajectory):
            target_pos = desired_trajectory[ind]
            error = np.sqrt((target_pos[0] - x3) ** 2 + (target_pos[1] - y3) ** 2)
            plt.annotate(
                f"Error: {error:.4f}",
                xy=(0.75, 0.05),
                xycoords="figure fraction",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )
        
        plt.pause(0.01)
    
    plt.ioff()
    plt.show()


def plot_error(errors):
    plt.figure(figsize=(10, 6))
    plt.plot(errors)
    plt.grid(True, alpha=0.3)
    plt.title("Tracking Error")
    plt.xlabel("Time Sample")
    plt.ylabel("Error [m]")
    plt.show()


def main():
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Run 3_dof_train.py first to train the model")
        sys.exit(1)
    
    model = keras.models.load_model(MODEL_PATH)
    desired_trajectory = generate_circle_trajectory()
    robot_positions = predict_trajectory(model, desired_trajectory)
    errors = calculate_error(desired_trajectory, robot_positions)
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    print(f"Mean error: {mean_error:.6f}m, Max error: {max_error:.6f}m")
    
    animate_robot(desired_trajectory, robot_positions)
    plot_error(errors)


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras


l1 = 1
l2 = 1


model = keras.models.load_model("robot_kinematics_rnn_model.keras")


def first_quadrant_circle():
    num_points = 200
    angle = np.linspace(0, 4 * np.pi, num_points)
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
        y = 0.6 + b * np.sin(2 * i)
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


desired_traj = first_quadrant_circle()


is_rnn = any(isinstance(layer, keras.layers.RNN) for layer in model.layers)
sequence_length = 5


robot_positions = []

if is_rnn:

    prev_q1, prev_q2 = 0.0, 0.0
    sequence = []

    for i in range(min(sequence_length, len(desired_traj))):
        x, y = desired_traj[i]
        sequence.append([x, y, prev_q1, prev_q2])

        if len(sequence) == sequence_length:
            model_input = np.array([sequence])
            prediction = model.predict(model_input, verbose=0)
            prev_q1, prev_q2 = prediction[0]
            robot_positions.append([prev_q1, prev_q2])

            sequence[-1][2] = prev_q1
            sequence[-1][3] = prev_q2

    for i in range(sequence_length, len(desired_traj)):
        x, y = desired_traj[i]

        sequence.pop(0)
        sequence.append([x, y, prev_q1, prev_q2])

        model_input = np.array([sequence])
        prediction = model.predict(model_input, verbose=0)
        prev_q1, prev_q2 = prediction[0]
        robot_positions.append([prev_q1, prev_q2])

        sequence[-1][2] = prev_q1
        sequence[-1][3] = prev_q2
else:

    for point in desired_traj:
        x, y = point

        model_input = np.array([[x, y]])

        prediction = model.predict(model_input, verbose=0)
        q1, q2 = prediction[0]

        robot_positions.append([q1, q2])


plt.figure(figsize=(10, 8))
plt.ion()


actual_x = []
actual_y = []

error_list = list()

for ind in range(len(robot_positions)):

    theta_1 = robot_positions[ind][0]
    theta_2 = robot_positions[ind][1]

    x0, y0 = 0, 0
    x1 = l1 * np.cos(theta_1)
    y1 = l1 * np.sin(theta_1)
    x2 = x1 + l2 * np.cos(theta_1 + theta_2)
    y2 = y1 + l2 * np.sin(theta_1 + theta_2)

    actual_x.append(x2)
    actual_y.append(y2)

    plt.clf()

    target_x = [pos[0] for pos in desired_traj]
    target_y = [pos[1] for pos in desired_traj]
    plt.plot(target_x, target_y, "r--", alpha=0.5, label="Desired Path")

    plt.plot(actual_x, actual_y, "g-", alpha=0.7, label="Actual Path")

    if ind < len(desired_traj):
        plt.scatter(
            desired_traj[ind][0], desired_traj[ind][1], color="red", s=50, zorder=5
        )

    plt.plot([x0, x1], [y0, y1], "b-", linewidth=3, label="Link 1")
    plt.plot([x1, x2], [y1, y2], "c-", linewidth=3, label="Link 2")

    plt.scatter([x0, x1, x2], [y0, y1, y2], color=["k", "b", "c"], s=[30, 20, 20])

    plt.xlim([-2.5, 2.5])
    plt.ylim([-2.5, 2.5])
    plt.grid(True, alpha=0.3)
    plt.title(f"2DOF Robot Following Trajectory - Frame {ind+1}/{len(robot_positions)}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")

    if ind == 0:
        plt.legend(loc="upper right")

    angle_info = f"Joint 1: {theta_1:.2f} rad\nJoint 2: {theta_2:.2f} rad"
    plt.annotate(
        angle_info,
        xy=(0.05, 0.05),
        xycoords="figure fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    if ind < len(desired_traj):
        target_pos = desired_traj[ind]
        actual_pos = (x2, y2)
        error = np.sqrt(
            (target_pos[0] - actual_pos[0]) ** 2 + (target_pos[1] - actual_pos[1]) ** 2
        )
        error_info = f"Error: {error:.4f}"
        error_list.append(error)
        plt.annotate(
            error_info,
            xy=(0.75, 0.05),
            xycoords="figure fraction",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

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
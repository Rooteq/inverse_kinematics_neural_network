import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras


l1 = 1
l2 = 1
l3 = 1


model = keras.models.load_model("3dof_model.keras")


def first_quadrant_circle():
    num_points = 200
    angle = np.linspace(0, 4 * np.pi, num_points)
    radius = 0.6
    traj = []
    for i in angle:
        traj.append([1.5 + radius * np.cos(i), 0.5 + radius * np.sin(i)])
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


model_inputs = []
for pos in desired_traj:
    x, y = pos

    dummy_joint_1 = 0.0
    dummy_joint_2 = 0.0
    dummy_joint_3 = 0.0

    model_inputs.append([x, y, dummy_joint_1, dummy_joint_2, dummy_joint_3])


robot_positions = []
for i in range(len(model_inputs)):

    input_data = np.array([model_inputs[i]])

    prediction = model.predict(input_data, verbose=0)
    q1, q2, q3 = prediction[0]

    robot_positions.append([q1, q2, q3])

    if i < len(model_inputs) - 1:

        next_x, next_y = desired_traj[i + 1]

        model_inputs[i + 1] = [next_x, next_y, q1, q2, q3]


actual_trajectory_x = []
actual_trajectory_y = []


plt.ion()
plt.figure(figsize=(10, 8))

error_list = list()

for ind in range(len(robot_positions)):

    theta_1 = robot_positions[ind][0]
    theta_2 = robot_positions[ind][1]
    theta_3 = robot_positions[ind][2]

    x0, y0 = 0, 0

    x1 = l1 * np.cos(theta_1)
    y1 = l1 * np.sin(theta_1)

    x2 = x1 + l2 * np.cos(theta_1 + theta_2)
    y2 = y1 + l2 * np.sin(theta_1 + theta_2)

    x3 = x2 + l3 * np.cos(theta_1 + theta_2 + theta_3)
    y3 = y2 + l3 * np.sin(theta_1 + theta_2 + theta_3)

    actual_trajectory_x.append(x3)
    actual_trajectory_y.append(y3)

    plt.clf()

    desired_x = [point[0] for point in desired_traj]
    desired_y = [point[1] for point in desired_traj]
    plt.plot(desired_x, desired_y, "r--", alpha=0.5, label="Desired")

    plt.plot(actual_trajectory_x, actual_trajectory_y, "b-", alpha=0.7, label="Actual")

    if ind < len(desired_traj):
        plt.scatter(
            desired_traj[ind][0], desired_traj[ind][1], color="blue", s=50, alpha=0.5
        )

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
    plt.title(f"3DOF Robot Animation - Frame {ind+1}/{len(robot_positions)}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend(loc="upper right")

    if ind < len(desired_traj):
        target_pos = desired_traj[ind]
        actual_pos = (x3, y3)
        error = np.sqrt(
            (target_pos[0] - actual_pos[0]) ** 2 + (target_pos[1] - actual_pos[1]) ** 2
        )
        error_list.append(error)
        error_info = f"Error: {error:.4f}"
        plt.annotate(
            error_info,
            xy=(0.75, 0.05),
            xycoords="figure fraction",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    plt.pause(0.01)

plt.ioff()
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(error_list)
plt.grid(True, alpha=0.3)
plt.title("Error")
plt.xlabel("Time sample")
plt.ylabel("Error value [rad]")
plt.show()

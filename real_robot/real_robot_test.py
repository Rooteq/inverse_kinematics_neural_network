import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import keras


r1 = 0.2
r2 = 0.15

model = keras.models.load_model("real_robot_model.keras")


def circle_3d():
    num_points = 400
    t = np.linspace(0, 4 * np.pi, num_points)
    radius = 0.4
    height_variation = 0.5
    traj = []
    for i in t:
        x = radius * np.cos(i)
        y = radius * np.sin(i)
        z = height_variation * np.sin(i / 2)
        traj.append([x, y, z])
    return traj


def first_quadrant_circle():
    """Generate a circle in the first quadrant of the XY plane (z=0)"""
    num_points = 200
    t = np.linspace(0, 2 * np.pi, num_points)
    radius = 0.05
    center_x, center_y = 0.05, 0.24
    traj = []
    for i in t:
        x = center_x + radius * np.cos(i)
        y = center_y + radius * np.sin(i)
        z = 0.1
        traj.append([x, y, z])
    return traj


def star_2d():
    """Generate a 2D star shape in the XY plane with constant Z"""
    num_points = 200
    points_per_segment = num_points // 10

    outer_radius = 0.15
    inner_radius = 0.06
    num_points_star = 5
    center_x, center_y = 0.05, 0.2
    z_height = 0.1

    traj = []

    for i in range(num_points_star):

        angle_outer = i * 2 * np.pi / num_points_star - np.pi / 2

        angle_inner = angle_outer + np.pi / num_points_star

        x_outer = center_x + outer_radius * np.cos(angle_outer)
        y_outer = center_y + outer_radius * np.sin(angle_outer)

        x_inner = center_x + inner_radius * np.cos(angle_inner)
        y_inner = center_y + inner_radius * np.sin(angle_inner)

        for j in range(points_per_segment):
            t = j / points_per_segment
            x = x_outer * (1 - t) + x_inner * t
            y = y_outer * (1 - t) + y_inner * t
            z = z_height
            traj.append([x, y, z])

        next_idx = (i + 1) % num_points_star
        angle_next_outer = next_idx * 2 * np.pi / num_points_star - np.pi / 2
        x_next_outer = center_x + outer_radius * np.cos(angle_next_outer)
        y_next_outer = center_y + outer_radius * np.sin(angle_next_outer)

        for j in range(points_per_segment):
            t = j / points_per_segment
            x = x_inner * (1 - t) + x_next_outer * t
            y = y_inner * (1 - t) + y_next_outer * t
            z = z_height
            traj.append([x, y, z])

    return traj


def cube_3d():
    """Generate a cube trajectory in 3D space"""
    num_points = 320
    traj = []

    size = 0.1
    center_x, center_y, center_z = 0.05, 0.2, 0.1

    for i in range(80):
        t = i / 80
        if t < 0.25:
            x = center_x - size / 2 + t * 4 * size
            y = center_y - size / 2
            z = center_z - size / 2
        elif t < 0.5:
            x = center_x + size / 2
            y = center_y - size / 2 + (t - 0.25) * 4 * size
            z = center_z - size / 2
        elif t < 0.75:
            x = center_x + size / 2 - (t - 0.5) * 4 * size
            y = center_y + size / 2
            z = center_z - size / 2
        else:
            x = center_x - size / 2
            y = center_y + size / 2 - (t - 0.75) * 4 * size
            z = center_z - size / 2
        traj.append([x, y, z])

    for i in range(80):
        t = i / 80
        if t < 0.25:
            x = center_x - size / 2 + t * 4 * size
            y = center_y - size / 2
            z = center_z + size / 2
        elif t < 0.5:
            x = center_x + size / 2
            y = center_y - size / 2 + (t - 0.25) * 4 * size
            z = center_z + size / 2
        elif t < 0.75:
            x = center_x + size / 2 - (t - 0.5) * 4 * size
            y = center_y + size / 2
            z = center_z + size / 2
        else:
            x = center_x - size / 2
            y = center_y + size / 2 - (t - 0.75) * 4 * size
            z = center_z + size / 2
        traj.append([x, y, z])

    for i in range(80):
        t = i / 80
        if t < 0.25:
            x = center_x - size / 2
            y = center_y - size / 2
            z = center_z - size / 2 + t * 4 * size
        elif t < 0.5:
            x = center_x + size / 2
            y = center_y - size / 2
            z = center_z - size / 2 + (t - 0.25) * 4 * size
        elif t < 0.75:
            x = center_x + size / 2
            y = center_y + size / 2
            z = center_z - size / 2 + (t - 0.5) * 4 * size
        else:
            x = center_x - size / 2
            y = center_y + size / 2
            z = center_z - size / 2 + (t - 0.75) * 4 * size
        traj.append([x, y, z])

    return traj


def skewed_ellipse_3d():
    """Generate a skewed ellipse in 3D space"""
    num_points = 200
    t = np.linspace(0, 2 * np.pi, num_points)

    a = 0.15
    b = 0.08
    center_x, center_y, center_z = 0.05, 0.2, 0.1

    skew_factor = 0.5
    theta = np.pi / 6
    phi = np.pi / 4

    traj = []
    for i in t:

        x0 = a * np.cos(i)
        y0 = b * np.sin(i)
        z0 = 0.02 * np.sin(3 * i)

        x1 = x0 + skew_factor * y0
        y1 = y0
        z1 = z0

        x2 = x1 * np.cos(theta) - y1 * np.sin(theta)
        y2 = x1 * np.sin(theta) + y1 * np.cos(theta)
        z2 = z1

        y3 = y2 * np.cos(phi) - z2 * np.sin(phi)
        z3 = y2 * np.sin(phi) + z2 * np.cos(phi)
        x3 = x2

        x = center_x + x3
        y = center_y + y3
        z = center_z + z3

        traj.append([x, y, z])

    return traj


desired_traj = star_2d()


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
ax = fig.add_subplot(111, projection="3d")
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
    ax.plot(target_x, target_y, target_z, "r--", alpha=0.5, label="Desired Path")

    ax.plot(actual_x, actual_y, actual_z, "g-", alpha=0.7, label="Actual Path")

    if ind < len(desired_traj):
        ax.scatter(
            desired_traj[ind][0],
            desired_traj[ind][1],
            desired_traj[ind][2],
            color="red",
            s=50,
            zorder=5,
        )

    axis_length = 0.3
    ax.quiver(
        0, 0, 0, axis_length, 0, 0, color="r", arrow_length_ratio=0.1, label="X axis"
    )
    ax.quiver(
        0, 0, 0, 0, axis_length, 0, color="g", arrow_length_ratio=0.1, label="Y axis"
    )
    ax.quiver(
        0, 0, 0, 0, 0, axis_length, color="b", arrow_length_ratio=0.1, label="Z axis"
    )

    ax.plot([x0, j1_x], [y0, j1_y], [z0, j1_z], "k-", linewidth=3)

    ax.plot([j1_x, x1], [j1_y, y1], [j1_z, z1], "navy", linewidth=5, label="Link 1")

    ax.plot([x1, x2], [y1, y2], [z1, z2], "blue", linewidth=5, label="Link 2")

    ax.scatter(
        [j1_x, x1, x2],
        [j1_y, y1, y2],
        [j1_z, z1, z2],
        color=["black", "navy", "blue"],
        s=[80, 60, 60],
    )

    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title(f"3DOF Robot Arm - Frame {ind+1}/{len(robot_positions)}")

    if ind == 0:
        ax.legend(loc="upper right")

    angle_info = f"Joint 1 (q1): {q1:.2f} rad\nJoint 2 (q2): {q2:.2f} rad\nJoint 3 (q3): {q3:.2f} rad"
    ax.text2D(
        0.05,
        0.05,
        angle_info,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    if ind < len(desired_traj):
        target_pos = desired_traj[ind]
        actual_pos = (x2, y2, z2)
        error = np.sqrt(
            (target_pos[0] - actual_pos[0]) ** 2
            + (target_pos[1] - actual_pos[1]) ** 2
            + (target_pos[2] - actual_pos[2]) ** 2
        )
        error_list.append(error)
        error_info = f"Position Error: {error:.4f}"
        ax.text2D(
            0.75,
            0.05,
            error_info,
            transform=ax.transAxes,
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

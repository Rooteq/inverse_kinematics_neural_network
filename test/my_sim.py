import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

class RobotVisualizer:
    def __init__(self):
        # Robot dimensions
        self.r1 = 1.0  # Length of first link
        self.r2 = 1.0  # Length of second link
        
        # Initial joint angles
        self.q1 = 0.0  # Base rotation around x-axis
        self.q2 = 0.0  # First joint angle (shoulder)
        self.q3 = 0.0  # Second joint angle (elbow)
        
        # Create the figure and 3D axis
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Add sliders
        self.fig.subplots_adjust(bottom=0.3)  # Make room for sliders
        
        # Slider for q1 (base rotation around x-axis)
        self.ax_q1 = plt.axes([0.25, 0.2, 0.65, 0.03])
        self.slider_q1 = Slider(
            ax=self.ax_q1,
            label='Base Rotation (q₁, x-axis)',
            valmin=-np.pi,
            valmax=np.pi,
            valinit=self.q1,
        )
        self.slider_q1.on_changed(self.update)
        
        # Slider for q2 (shoulder joint)
        self.ax_q2 = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.slider_q2 = Slider(
            ax=self.ax_q2,
            label='Shoulder Joint (q₂)',
            valmin=-np.pi,
            valmax=np.pi,
            valinit=self.q2,
        )
        self.slider_q2.on_changed(self.update)
        
        # Slider for q3 (elbow joint)
        self.ax_q3 = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider_q3 = Slider(
            ax=self.ax_q3,
            label='Elbow Joint (q₃)',
            valmin=-np.pi,
            valmax=np.pi,
            valinit=self.q3,
        )
        self.slider_q3.on_changed(self.update)
        
        # Initialize the robot visualization
        self.update(None)
        
    def update(self, val):
        # Get current values from sliders
        self.q1 = self.slider_q1.val
        self.q2 = self.slider_q2.val
        self.q3 = self.slider_q3.val
        
        # Clear the current plot
        self.ax.clear()
        
        # Calculate joint positions using forward kinematics with x-axis base rotation
        x0, y0, z0 = 0, 0, 0  # Base
        
        # First joint position (after base rotation)
        j1_x, j1_y, j1_z = 0, 0, 0
        
        # End of first link
        # For a robot with base rotation around x-axis:
        # - q1 rotates around x-axis (affects y and z)
        # - q2 then rotates in the new y-z plane

        q2_offset = 0.34906585
        q3_offset = 0.785398163

        x1 = -self.r1 * np.cos(self.q2 + q2_offset)
        y1 = -self.r1 * np.sin(self.q2 + q2_offset) * np.cos(self.q1)
        z1 = self.r1 * np.sin(self.q2 + q2_offset) * np.sin(self.q1)
        
        # End effector position (end of second link)
        x2 = x1 - self.r2 * np.cos(self.q2 + q2_offset + np.pi - self.q3 - q3_offset)
        y2 = y1 - self.r2 * np.sin(self.q2 + q2_offset + np.pi - self.q3 - q3_offset) * np.cos(self.q1)
        z2 = z1 + self.r2 * np.sin(self.q2 + q2_offset + np.pi - self.q3 - q3_offset) * np.sin(self.q1)

        # Draw coordinate system at origin
        axis_length = 0.3
        self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1)
        self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1)
        self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1)
        self.ax.text(axis_length*1.1, 0, 0, "X", color='red')
        self.ax.text(0, axis_length*1.1, 0, "Y", color='green')
        self.ax.text(0, 0, axis_length*1.1, "Z", color='blue')
        
        # Draw the robot links
        # Base to first joint
        self.ax.plot([x0, j1_x], [y0, j1_y], [z0, j1_z], 'k-', linewidth=3)
        
        # First link
        self.ax.plot([j1_x, x1], [j1_y, y1], [j1_z, z1], 'navy', linewidth=5)
        
        # Second link
        self.ax.plot([x1, x2], [y1, y2], [z1, z2], 'blue', linewidth=5)
        
        # Draw joints
        self.ax.scatter([j1_x, x1, x2], [j1_y, y1, y2], [j1_z, z1, z2], 
                  color=['black', 'navy', 'blue'], s=[80, 60, 60])
        
        # Visualize the x-axis rotation plane
        # Create a semi-transparent disc to show the rotation plane
        theta = np.linspace(0, 2*np.pi, 100)
        radius = 0.5
        y_circle = radius * np.cos(theta)
        z_circle = radius * np.sin(theta)
        x_circle = np.zeros_like(theta)
        self.ax.plot(x_circle, y_circle, z_circle, 'r--', alpha=0.3)
        
        # Set plot properties
        self.ax.set_xlim([-2.5, 2.5])
        self.ax.set_ylim([-2.5, 2.5])
        self.ax.set_zlim([-2.5, 2.5])
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_zlabel('Z Position')
        self.ax.set_title('3DOF Robot Arm with X-Axis Base Rotation')
        
        # Add joint angle information
        angle_info = f'Joint 1 (q₁, x-axis): {self.q1:.2f} rad\nJoint 2 (q₂): {self.q2:.2f} rad\nJoint 3 (q₃): {self.q3:.2f} rad'
        self.ax.text2D(0.05, 0.95, angle_info, transform=self.ax.transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
        
        # Calculate and display end effector position
        ee_info = f'End Effector Position:\nX: {x2:.2f}\nY: {y2:.2f}\nZ: {z2:.2f}'
        self.ax.text2D(0.75, 0.95, ee_info, transform=self.ax.transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
        
        # Draw small sphere at end effector
        self.ax.scatter([x2], [y2], [z2], color='red', s=100, alpha=0.7)
        
        # View from a good angle to see the x-axis rotation
        # self.ax.view_init(elev=20, azim=-30)
        
        # Redraw the canvas
        self.fig.canvas.draw_idle()
    
    def show(self):
        plt.show()

if __name__ == "__main__":
    # Create and display the robot visualizer
    visualizer = RobotVisualizer()
    visualizer.show()
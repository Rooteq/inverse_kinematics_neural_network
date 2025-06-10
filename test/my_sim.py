import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

class RobotVisualizer:
    def __init__(self):
        # Robot dimensions
        self.L1 = 0.06  # Length of first link
        self.L2 = 0.2  # Length of second link
        self.L3 = 0.15  # Length of third link (end effector)
        
        # Initial joint angles
        self.q1 = 0.0  # First joint angle
        self.q2 = 0.0  # Second joint angle
        self.q3 = 0.0  # Third joint angle
        
        # Create the figure and 3D axis
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Add sliders
        self.fig.subplots_adjust(bottom=0.3)  # Make room for sliders
        
        # Slider for q1
        self.ax_q1 = plt.axes([0.25, 0.2, 0.65, 0.03])
        self.slider_q1 = Slider(
            ax=self.ax_q1,
            label='Joint 1 (q₁)',
            valmin=-np.pi,
            valmax=np.pi,
            valinit=self.q1,
        )
        self.slider_q1.on_changed(self.update)
        
        # Slider for q2
        self.ax_q2 = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.slider_q2 = Slider(
            ax=self.ax_q2,
            label='Joint 2 (q₂)',
            valmin=-np.pi,
            valmax=np.pi,
            valinit=self.q2,
        )
        self.slider_q2.on_changed(self.update)
        
        # Slider for q3
        self.ax_q3 = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider_q3 = Slider(
            ax=self.ax_q3,
            label='Joint 3 (q₃)',
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
        
        # Base position
        x0, y0, z0 = 0, 0, 0
        
        # Constant offset from the matrix (157/100 = 1.57 rad ≈ 90°)
        q1_offset = 1.57  # 157/100
        
        # Calculate joint positions using the provided transformation matrices
        
        # Joint 1 position (first transformation matrix)
        x1 = 0
        y1 = self.L1 * np.cos(self.q1 + q1_offset)
        z1 = self.L1 * np.sin(self.q1 + q1_offset)
        
        # Joint 2 position (second transformation matrix)
        x2 = self.L2 * np.sin(self.q2)
        y2 = self.L1 * np.cos(self.q1 + q1_offset) - self.L2 * np.sin(self.q1 + q1_offset) * np.cos(self.q2)
        z2 = self.L1 * np.sin(self.q1 + q1_offset) + self.L2 * np.cos(self.q1 + q1_offset) * np.cos(self.q2)
        
        # End effector position (third transformation matrix)
        x3 = self.L2 * np.sin(self.q2) + self.L3 * np.cos(self.q2) * np.sin(self.q3) + self.L3 * np.cos(self.q3) * np.sin(self.q2)
        y3 = self.L1 * np.cos(self.q1 + q1_offset) - self.L2 * np.sin(self.q1 + q1_offset) * np.cos(self.q2) - self.L3 * np.sin(self.q1 + q1_offset) * np.cos(self.q2) * np.cos(self.q3) + self.L3 * np.sin(self.q1 + q1_offset) * np.sin(self.q2) * np.sin(self.q3)
        z3 = self.L1 * np.sin(self.q1 + q1_offset) + self.L2 * np.cos(self.q1 + q1_offset) * np.cos(self.q2) + self.L3 * np.cos(self.q1 + q1_offset) * np.cos(self.q2) * np.cos(self.q3) - self.L3 * np.cos(self.q1 + q1_offset) * np.sin(self.q2) * np.sin(self.q3)
        
        # Draw coordinate system at origin
        axis_length = 0.3
        self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1)
        self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1)
        self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1)
        self.ax.text(axis_length*1.1, 0, 0, "X", color='red')
        self.ax.text(0, axis_length*1.1, 0, "Y", color='green')
        self.ax.text(0, 0, axis_length*1.1, "Z", color='blue')
        
        # Draw the robot links
        # Base to joint 1
        self.ax.plot([x0, x1], [y0, y1], [z0, z1], 'k-', linewidth=3)
        
        # Joint 1 to joint 2
        self.ax.plot([x1, x2], [y1, y2], [z1, z2], 'navy', linewidth=5)
        
        # Joint 2 to end effector
        self.ax.plot([x2, x3], [y2, y3], [z2, z3], 'blue', linewidth=5)
        
        # Draw joints and end effector
        self.ax.scatter([x1, x2, x3], [y1, y2, y3], [z1, z2, z3], 
                  color=['black', 'navy', 'red'], s=[60, 60, 100])
        
        # Set plot properties
        self.ax.set_xlim([-0.4, 0.4])
        self.ax.set_ylim([-0.4, 0.4])
        self.ax.set_zlim([-0.4, 0.4])
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_zlabel('Z Position')
        self.ax.set_title('3DOF Robot Arm with DH Parameters')
        
        # Add joint angle information
        # angle_info = f'Joint 1 (q₁): {self.q1:.2f} rad\nJoint 2 (q₂): {self.q2:.2f} rad\nJoint 3 (q₃): {self.q3:.2f} rad'
        # self.ax.text2D(0.05, 0.95, angle_info, transform=self.ax.transAxes,
        #          bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
        
        # Calculate and display end effector position
        # ee_info = f'End Effector Position:\nX: {x3:.2f}\nY: {y3:.2f}\nZ: {z3:.2f}'
        # self.ax.text2D(0.75, 0.95, ee_info, transform=self.ax.transAxes,
        #          bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
        
        # View from a good angle to see the robot
        # self.ax.view_init(elev=30, azim=45)
        
        # Redraw the canvas
        self.fig.canvas.draw_idle()
    
    def show(self):
        plt.show()

if __name__ == "__main__":
    # Create and display the robot visualizer
    visualizer = RobotVisualizer()
    visualizer.show()
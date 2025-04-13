import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import math

class RobotVisualizer:
    def __init__(self):
        # Robot dimensions
        self.l1 = 0.062  # Base height
        self.l2 = 0.2    # Length of first link
        self.l3 = 0.15   # Length of second link
        
        # Joint offsets
        self.joint_offset_2 = 0.349  # Offset for joint 2
        self.joint_offset_3 = 0.785  # Offset for joint 3
        
        # Initial position
        self.x = 0.2  # Using positive value for better initial view
        self.y = 0.0
        self.z = 0.1
        
        # Initial joint angles (will be calculated by IK)
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0
        
        # Calculate initial joint angles
        self.calculate_ik(self.x, self.y, self.z)
        
        # Create the figure and 3D axis
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Add sliders
        self.fig.subplots_adjust(bottom=0.3)  # Make room for sliders
        
        # Slider for x position (both positive and negative values)
        self.ax_x = plt.axes([0.25, 0.2, 0.65, 0.03])
        self.slider_x = Slider(
            ax=self.ax_x,
            label='X Position',
            valmin=-0.3,
            valmax=0.3,
            valinit=self.x,
        )
        self.slider_x.on_changed(self.update)
        
        # Slider for y position
        self.ax_y = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.slider_y = Slider(
            ax=self.ax_y,
            label='Y Position',
            valmin=-0.3,
            valmax=0.3,
            valinit=self.y,
        )
        self.slider_y.on_changed(self.update)
        
        # Slider for z position
        self.ax_z = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider_z = Slider(
            ax=self.ax_z,
            label='Z Position',
            valmin=-0.3,
            valmax=0.3,
            valinit=self.z,
        )
        self.slider_z.on_changed(self.update)
        
        # Initialize the robot visualization
        self.update(None)
    
    def calculate_ik(self, x, y, z):
        """Inverse kinematics calculation based on the given C++ code"""

        AG = np.sqrt(y**2 + z**2 - self.l1**2)
        OA = self.l1

        q1 = np.pi - np.arctan2(AG,OA) - np.arctan2(y,-z) # TRY NO MINUS

        GC = np.abs(x)

        AC = np.sqrt(AG**2 + GC**2)

        alpha = np.arccos(-((AC**2 - self.l2**2 - self.l3**2)/(2*self.l2*self.l3)))


        q3 = np.pi-alpha

        q2 = np.arctan2(GC,AG) - np.arctan2(self.l3*np.sin(q3),self.l2+self.l3*np.cos(q3))

        self.q1 = q1
        self.q2 = q2 - self.joint_offset_2
        self.q3 = np.pi - q3 - self.joint_offset_3


        print(f"Target: ({x:.3f}, {y:.3f}, {z:.3f})")
        print(f"Joint angles: q1={self.q1:.3f}, q2={self.q2:.3f}, q3={self.q3:.3f}")
        return True
    
    def update(self, val):
        # Get current values from sliders
        self.x = self.slider_x.val
        self.y = self.slider_y.val
        self.z = self.slider_z.val
        
        # Calculate joint angles using inverse kinematics
        ik_success = self.calculate_ik(self.x, self.y, self.z)
        
        # Clear the current plot
        self.ax.clear()
        
        # Draw robot only if IK is successful
        if ik_success is not False:
            # Base position
            x0, y0, z0 = 0, 0, 0
            
            # First joint position (after base rotation)
            j1_x, j1_y, j1_z = 0, 0, 0
            
            # End of first link using forward kinematics with joint angles
            # Keep X axis direction consistent with target X direction
            x1 = - self.l2 * np.cos(self.q2 + self.joint_offset_2)
            y1 = - self.l2 * np.sin(self.q2 + self.joint_offset_2) * np.cos(self.q1)
            z1 = self.l2 * np.sin(self.q2 + self.joint_offset_2) * np.sin(self.q1)
            
            # End effector position (end of second link)
            x2 = x1 - self.l3 * np.cos(self.q2 + self.joint_offset_2 + np.pi - self.q3 - self.joint_offset_3)
            y2 = y1 - self.l3 * np.sin(self.q2 + self.joint_offset_2 + np.pi - self.q3 - self.joint_offset_3) * np.cos(self.q1)
            z2 = z1 + self.l3 * np.sin(self.q2 + self.joint_offset_2 + np.pi - self.q3 - self.joint_offset_3) * np.sin(self.q1)
            
            # Draw coordinate system at origin
            axis_length = 0.1
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
            radius = 0.1
            y_circle = radius * np.cos(theta)
            z_circle = radius * np.sin(theta)
            x_circle = np.zeros_like(theta)
            self.ax.plot(x_circle, y_circle, z_circle, 'r--', alpha=0.3)
            
            # Draw small sphere at target position
            self.ax.scatter([self.x], [self.y], [self.z], color='red', s=100, alpha=0.7)
            self.ax.text(self.x, self.y, self.z, "Target", color='red')
            
            # Joint angle information
            angle_info = f'Joint 1 (q₁): {self.q1:.3f} rad\nJoint 2 (q₂): {self.q2:.3f} rad\nJoint 3 (q₃): {self.q3:.3f} rad'
            self.ax.text2D(0.05, 0.95, angle_info, transform=self.ax.transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
            
            # End effector position (from forward kinematics)
            ee_info = f'End Effector Position:\nX: {x2:.3f}\nY: {y2:.3f}\nZ: {z2:.3f}'
            self.ax.text2D(0.75, 0.95, ee_info, transform=self.ax.transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
            
            # Target position
            target_info = f'Target Position:\nX: {self.x:.3f}\nY: {self.y:.3f}\nZ: {self.z:.3f}'
            self.ax.text2D(0.75, 0.8, target_info, transform=self.ax.transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.7))
        else:
            # Display out of reach message
            self.ax.text2D(0.5, 0.5, "Target position out of reach!", 
                     transform=self.ax.transAxes, ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc='red', alpha=0.7))
        
        # Set plot properties
        self.ax.set_xlim([-0.4, 0.4])  # Standard X axis limits
        self.ax.set_ylim([-0.4, 0.4])
        self.ax.set_zlim([-0.4, 0.4])
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_zlabel('Z Position')
        self.ax.set_title('3DOF Robot Arm with Inverse Kinematics')
        
        # View from a good angle to see the robot
        # self.ax.view_init(elev=30, azim=-60)
        
        # Redraw the canvas
        self.fig.canvas.draw_idle()
    
    def show(self):
        plt.show()

if __name__ == "__main__":
    # Create and display the robot visualizer
    visualizer = RobotVisualizer()
    visualizer.show()
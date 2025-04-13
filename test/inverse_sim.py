import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import math

class RobotVisualizer:
    def __init__(self):
        # Robot dimensions
        self.L1 = 0.062  # Base height
        self.L2 = 0.2    # Length of first link
        self.L3 = 0.15   # Length of second link
        
        # Joint offsets from DH parameters
        self.q1_offset = 1.57  # 157/100 rad
        
        # Initial position for IK
        self.target_x = 0.2
        self.target_y = 0.0
        self.target_z = 0.1
        
        # Initial joint angles
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0
        
        # Calculate initial joint angles using IK
        self.calculate_ik(self.target_x, self.target_y, self.target_z)
        
        # Create the figure and 3D axis
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Add sliders
        self.fig.subplots_adjust(bottom=0.3)  # Make room for sliders
        
        # Slider for x position
        self.ax_x = plt.axes([0.25, 0.2, 0.65, 0.03])
        self.slider_x = Slider(
            ax=self.ax_x,
            label='X Position',
            valmin=-0.3,
            valmax=0.3,
            valinit=self.target_x,
        )
        self.slider_x.on_changed(self.update)
        
        # Slider for y position
        self.ax_y = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.slider_y = Slider(
            ax=self.ax_y,
            label='Y Position',
            valmin=-0.3,
            valmax=0.3,
            valinit=self.target_y,
        )
        self.slider_y.on_changed(self.update)
        
        # Slider for z position
        self.ax_z = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider_z = Slider(
            ax=self.ax_z,
            label='Z Position',
            valmin=-0.3,
            valmax=0.3,
            valinit=self.target_z,
        )
        self.slider_z.on_changed(self.update)
        
        # Initialize the robot visualization
        self.update(None)
    
    def calculate_ik(self, x, y, z):
        """Inverse kinematics calculation"""
        try:
            # Calculate AG (perpendicular distance from base to end-effector projected on YZ plane)
            AG = np.sqrt(y**2 + z**2 - self.L1**2)
            OA = self.L1

            # Calculate q1 (base joint angle)
            q1 = np.pi - np.arctan2(AG, OA) - np.arctan2(y, -z)
            
            # Calculate GC (distance in X direction)
            GC = np.abs(x)
            
            # Calculate AC (straight-line distance from first joint to end effector)
            AC = np.sqrt(AG**2 + GC**2)
            
            # Check if target is within reach
            if AC > self.L2 + self.L3:
                print(f"Target position ({x:.3f}, {y:.3f}, {z:.3f}) out of reach!")
                return False
            
            # Calculate q3 (elbow joint angle) using law of cosines
            cos_alpha = -((AC**2 - self.L2**2 - self.L3**2)/(2*self.L2*self.L3))
            
            # Check if the value for arccos is valid
            if cos_alpha < -1 or cos_alpha > 1:
                print(f"Target position requires invalid joint angles!")
                return False
            
            alpha = np.arccos(cos_alpha)
            q3 = np.pi - alpha
            
            # Calculate q2 (shoulder joint angle)
            q2 = np.arctan2(GC, AG) - np.arctan2(self.L3*np.sin(q3), self.L2+self.L3*np.cos(q3))

            self.q1 = q1
            self.q2 = q2
            self.q3 = q3
            
            print(f"Target: ({x:.3f}, {y:.3f}, {z:.3f})")
            print(f"Joint angles: q1={self.q1:.3f}, q2={self.q2:.3f}, q3={self.q3:.3f}")
            return True
            
        except Exception as e:
            print(f"IK calculation error: {e}")
            return False
    
    def calculate_fk(self):
        """Forward kinematics calculation using the DH transformation matrices"""
        # Base position
        x0, y0, z0 = 0, 0, 0
        
        # Joint 1 position (first transformation matrix)
        x1 = 0
        y1 = self.L1 * np.cos(self.q1 + self.q1_offset)
        z1 = self.L1 * np.sin(self.q1 + self.q1_offset)
        
        # Joint 2 position (second transformation matrix)
        x2 = self.L2 * np.sin(self.q2)
        y2 = self.L1 * np.cos(self.q1 + self.q1_offset) - self.L2 * np.sin(self.q1 + self.q1_offset) * np.cos(self.q2)
        z2 = self.L1 * np.sin(self.q1 + self.q1_offset) + self.L2 * np.cos(self.q1 + self.q1_offset) * np.cos(self.q2)
        
        # End effector position (third transformation matrix)
        x3 = self.L2 * np.sin(self.q2) + self.L3 * np.cos(self.q2) * np.sin(self.q3) + self.L3 * np.cos(self.q3) * np.sin(self.q2)
        y3 = self.L1 * np.cos(self.q1 + self.q1_offset) - self.L2 * np.sin(self.q1 + self.q1_offset) * np.cos(self.q2) - self.L3 * np.sin(self.q1 + self.q1_offset) * np.cos(self.q2) * np.cos(self.q3) + self.L3 * np.sin(self.q1 + self.q1_offset) * np.sin(self.q2) * np.sin(self.q3)
        z3 = self.L1 * np.sin(self.q1 + self.q1_offset) + self.L2 * np.cos(self.q1 + self.q1_offset) * np.cos(self.q2) + self.L3 * np.cos(self.q1 + self.q1_offset) * np.cos(self.q2) * np.cos(self.q3) - self.L3 * np.cos(self.q1 + self.q1_offset) * np.sin(self.q2) * np.sin(self.q3)
        
        return {
            'base': (x0, y0, z0),
            'j1': (x1, y1, z1),
            'j2': (x2, y2, z2),
            'ee': (x3, y3, z3)
        }
    
    def update(self, val):
        # Get target position from sliders
        self.target_x = self.slider_x.val
        self.target_y = self.slider_y.val
        self.target_z = self.slider_z.val
        
        # Calculate joint angles using inverse kinematics
        ik_success = self.calculate_ik(self.target_x, self.target_y, self.target_z)
        
        # Clear the current plot
        self.ax.clear()
        
        # Calculate robot positions using forward kinematics for visualization
        joints = self.calculate_fk()
        
        if ik_success is not False:
            # Get joint positions
            x0, y0, z0 = joints['base']
            x1, y1, z1 = joints['j1']
            x2, y2, z2 = joints['j2']
            x3, y3, z3 = joints['ee']
            
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
            self.ax.plot([x0, x1], [y0, y1], [z0, z1], 'k-', linewidth=3)
            
            # First joint to second joint
            self.ax.plot([x1, x2], [y1, y2], [z1, z2], 'navy', linewidth=5)
            
            # Second joint to end effector
            self.ax.plot([x2, x3], [y2, y3], [z2, z3], 'blue', linewidth=5)
            
            # Draw joints and end effector
            self.ax.scatter([x1, x2, x3], [y1, y2, y3], [z1, z2, z3], 
                      color=['black', 'navy', 'red'], s=[60, 60, 100])
            
            # Visualize the rotation plane at the base
            theta = np.linspace(0, 2*np.pi, 100)
            radius = 0.1
            y_circle = radius * np.cos(theta)
            z_circle = radius * np.sin(theta)
            x_circle = np.zeros_like(theta)
            self.ax.plot(x_circle, y_circle, z_circle, 'r--', alpha=0.3)
            
            # Draw the target position
            self.ax.scatter([self.target_x], [self.target_y], [self.target_z], 
                      color='orange', s=120, alpha=0.7)
            self.ax.text(self.target_x, self.target_y, self.target_z, "Target", color='red')
            
            # Joint angle information
            angle_info = f'Joint 1 (q₁): {self.q1:.3f} rad\nJoint 2 (q₂): {self.q2:.3f} rad\nJoint 3 (q₃): {self.q3:.3f} rad'
            self.ax.text2D(0.05, 0.95, angle_info, transform=self.ax.transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
            
            # End effector position (from forward kinematics)
            ee_info = f'End Effector Position:\nX: {x3:.3f}\nY: {y3:.3f}\nZ: {z3:.3f}'
            self.ax.text2D(0.75, 0.95, ee_info, transform=self.ax.transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
            
            # Target position
            target_info = f'Target Position:\nX: {self.target_x:.3f}\nY: {self.target_y:.3f}\nZ: {self.target_z:.3f}'
            self.ax.text2D(0.75, 0.8, target_info, transform=self.ax.transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.7))
            
            # Calculate error between target and actual end effector position
            error_x = abs(x3 - self.target_x)
            error_y = abs(y3 - self.target_y)
            error_z = abs(z3 - self.target_z)
            error_total = np.sqrt(error_x**2 + error_y**2 + error_z**2)
            
            error_info = f'Position Error: {error_total:.5f}'
            self.ax.text2D(0.05, 0.85, error_info, transform=self.ax.transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", fc='lightgreen', alpha=0.7))
        else:
            # Display out of reach message
            self.ax.text2D(0.5, 0.5, "Target position out of reach!", 
                     transform=self.ax.transAxes, ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc='red', alpha=0.7))
        
        # Set plot properties
        self.ax.set_xlim([-0.4, 0.4])
        self.ax.set_ylim([-0.4, 0.4])
        self.ax.set_zlim([-0.4, 0.4])
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_zlabel('Z Position')
        self.ax.set_title('3DOF Robot Arm - Inverse Kinematics')
        
        # View from a good angle to see the robot
        # self.ax.view_init(elev=30, azim=30)
        
        # Redraw the canvas
        self.fig.canvas.draw_idle()
    
    def show(self):
        plt.show()

if __name__ == "__main__":
    # Create and display the robot visualizer
    visualizer = RobotVisualizer()
    visualizer.show()
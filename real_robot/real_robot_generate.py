import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

class DatasetGenerator:
    def __init__(self, no_points, no_randomsamples):
        self.l1 = 0.2
        self.l2 = 0.15
        self.l3 = 1
        self.randomsamples = no_randomsamples
        self.theta_1 = np.linspace(-np.pi, np.pi, no_points)
        self.theta_2 = np.linspace(0, np.pi/2, no_points)
        self.theta_3 = np.linspace(0, np.pi/2, no_points)
        self.data = []
    
        self.q2_offset = 0.34906585
        self.q3_offset = 0.785398163
        
    def generate_points(self):
        for q1 in self.theta_1:
            for q2 in self.theta_2:
                for q3 in self.theta_3:
                    # New equations
                    x1 = self.l1 * np.cos(q2 + self.q2_offset)
                    y1 = self.l1 * np.sin(q2 + self.q2_offset) * np.cos(q1)
                    z1 = self.l1 * np.sin(q2 + self.q2_offset) * np.sin(q1)
        
                    # End effector position (end of second link)
                    x = np.round(x1 + self.l2 * np.cos(q2 + self.q2_offset + np.pi - q3 - self.q3_offset), 2)
                    y = np.round(y1 + self.l2 * np.sin(q2 + self.q2_offset + np.pi - q3 - self.q3_offset) * np.cos(q1), 2)
                    z = np.round(z1 + self.l2 * np.sin(q2 + self.q2_offset + np.pi - q3 - self.q3_offset) * np.sin(q1), 2)

                    # z = np.round(self.l1*np.cos(q1) + self.l2*np.cos(q2+np.pi-q3), 2)
                    # y = np.round(self.l1*np.sin(q1) + self.l2*np.cos(q2+np.pi-q3), 2)
                    # z = np.round((self.l1*np.sin(q2) + self.l2*np.sin(q2+np.pi-q3))*np.sin(q1), 2)
                    
                    input_position = (x, y, z)  # Added z coordinate
                    output_joints = (np.round(q1, 2), np.round(q2, 2), np.round(q3, 2))
                    
                    for sample in range(self.randomsamples):
                        # This is teaching robot about local solution space - dont do sudden movements, it also fills up space with more solutions
                        # It teaches the neural net the relationship between small changes in joint angles and resulting end-effector positions
                        input_joints = (np.round(q1 + 0.2*(np.random.random()-0.5), 2),
                                       np.round(q2 + 0.2*(np.random.random()-0.5), 2), 
                                       np.round(q3 + 0.2*(np.random.random()-0.5), 2))
                        self.data.append([input_position, input_joints, output_joints])
    
    def plot_points(self):
        # Plot the generated points in 3D
        if not self.data:
            print("No data to plot. Generate points first.")
            return
            
        # Extract positions from the data
        positions = [data_point[0] for data_point in self.data]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        z_coords = [pos[2] for pos in positions]
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the points
        scatter = ax.scatter(x_coords, y_coords, z_coords, 
                           c=z_coords, cmap='viridis', alpha=0.6, s=20)
        
        # Add labels and title
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('3D Workspace of Robot End Effector')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Z coordinate (m)')
        
        # Set equal aspect ratio for better visualization
        max_range = max(max(x_coords) - min(x_coords),
                       max(y_coords) - min(y_coords),
                       max(z_coords) - min(z_coords)) / 2.0
        
        mid_x = (max(x_coords) + min(x_coords)) * 0.5
        mid_y = (max(y_coords) + min(y_coords)) * 0.5
        mid_z = (max(z_coords) + min(z_coords)) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print(f"Plotted {len(self.data)} data points")
        print(f"X range: {min(x_coords):.3f} to {max(x_coords):.3f}")
        print(f"Y range: {min(y_coords):.3f} to {max(y_coords):.3f}")
        print(f"Z range: {min(z_coords):.3f} to {max(z_coords):.3f}")
    
    def save(self):
        mydata = csv.writer(open("real_robot_dataset.csv", "w"))
        mydata.writerow(['input position', 'input joint', 'output joint'])
        for data_i in self.data:
            mydata.writerow([data_i[0], data_i[1], data_i[2]])

def handle_dataset(path):
    my_file = Path(path)
    # if my_file.is_file():
    #     print("Dataset already exists")
    # else:
    generator = DatasetGenerator(30, 5)
    generator.generate_points()
    generator.save()
    generator.plot_points()
    print("Created dataset")
    # file exists

# handle_dataset("./real_robot_dataset.csv")
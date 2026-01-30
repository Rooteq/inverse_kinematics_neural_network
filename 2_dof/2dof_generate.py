import numpy as np
import csv
from pathlib import Path


DATASET_PATH = "2dof_dataset.csv"
NUM_POINTS = 40
NUM_SAMPLES = 20


class DatasetGenerator:
    def __init__(self, num_points, num_samples):
        self.l1 = 1.0
        self.l2 = 1.0
        self.num_samples = num_samples
        self.theta_1 = np.linspace(-np.pi, np.pi, num_points)
        self.theta_2 = np.linspace(-np.pi, np.pi, num_points)
        self.data = []

    def generate_points(self):
        for q1 in self.theta_1:
            for q2 in self.theta_2:
                x = np.round(self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2), 2)
                y = np.round(self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2), 2)

                input_position = (x, y)
                output_joints = (np.round(q1, 2), np.round(q2, 2))
                
                # Input joints are previous joints with random noise added - it simulates previous joint configuration
                # of the robot at high control frequency.
                for _ in range(self.num_samples):
                    input_joints = (
                        np.round(q1 + 0.2 * (np.random.random() - 0.5), 2),
                        np.round(q2 + 0.2 * (np.random.random() - 0.5), 2),
                    )
                    self.data.append([input_position, input_joints, output_joints])

    def save(self, path):
        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["input position", "input joint", "output joint"])
            for data_point in self.data:
                writer.writerow(data_point)


def main():
    dataset_file = Path(DATASET_PATH)
    
    if dataset_file.exists():
        print(f"Dataset exists: {DATASET_PATH}")
        return
    
    generator = DatasetGenerator(NUM_POINTS, NUM_SAMPLES)
    generator.generate_points()
    generator.save(DATASET_PATH)
    print(f"Generated dataset: {DATASET_PATH} ({len(generator.data)} samples)")


if __name__ == "__main__":
    main()

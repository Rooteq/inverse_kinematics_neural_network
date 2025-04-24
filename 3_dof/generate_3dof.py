import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

class DatasetGenerator:
    def __init__(self, no_points, no_randomsamples):
        self.l1 = 1
        self.l2 = 1
        self.l3 = 1
        self.randomsamples = no_randomsamples

        self.theta_1 = np.linspace(-np.pi, np.pi, no_points)
        self.theta_2 = np.linspace(-np.pi, np.pi, no_points)
        self.theta_3 = np.linspace(-np.pi, np.pi, no_points)

        self.data = []


    def generate_points(self):
        
        for q1 in self.theta_1:
            for q2 in self.theta_2:
                for q3 in self.theta_3:
                    x = np.round(self.l1*np.cos(q1) + self.l2*np.cos(q1+q2) + self.l3*np.cos(q1+q2+q3),2)
                    y = np.round(self.l1*np.sin(q1) + self.l2*np.sin(q1+q2) + self.l3*np.sin(q1+q2+q3),2)

                    input_position = (x,y)
                    output_joints = (np.round(q1,2), np.round(q2,2), np.round(q3,2))
                    for sample in range(self.randomsamples):

                        # This is teaching robot about local solution space - dont do sudden movements, it also fills up space with more solutions
                        # It teaches the neural net the relationship between small changes in joint angles and resulting end-effector positions
                        input_joints = (np.round(q1 + 0.2*(np.random.random()-0.5) ,2) ,np.round(q2 + 0.2*(np.random.random()-0.5) ,2), np.round(q3 + 0.2*(np.random.random()-0.5), 2)) 
                        self.data.append( [ input_position , input_joints , output_joints ] )

    def plot_points(self):
        # pass
        # print(self.data)
        s = np.array(self.data)
        print(s)

        # fig1 = plt.figure()
        # plt.xlim([-np.pi,np.pi])
        # plt.ylim([-np.pi,np.pi])
        # plt.scatter( s[:,1,0], s[:,1,1], color = 'red')
        # plt.scatter( s[:,2,0], s[:,2,1], color = 'blue')

        # fig2 = plt.figure()
        # plt.xlim([-0.35,0.35])
        # plt.ylim([-0.35,0.35])
        # plt.scatter( s[:,0,0], s[:,0,1], color = 'blue')
        # plt.show()
    
    def save(self):
        # print(self.data)
        mydata = csv.writer(open("3dof_dataset.csv", "w"))

        mydata.writerow([ 'input position', 'input joint', 'output joint' ])

        for data_i in self.data:
            mydata.writerow([ data_i[0], data_i[1], data_i[2] ])    

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

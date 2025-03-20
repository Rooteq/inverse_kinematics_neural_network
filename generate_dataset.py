import numpy as np
import matplotlib.pyplot as plt

class DatasetGenerator:
    def __init__(self, no_points, no_randomsamples):
        self.l1 = 0.2
        self.l2 = 0.15
        self.randomsamples = no_randomsamples

        self.theta_1 = np.linspace(-np.pi, np.pi, no_points)
        self.theta_2 = np.linspace(-np.pi, np.pi, no_points)

        self.data = []


    def generate_point(self):
        
        for q1 in self.theta_1:
            for q2 in self.theta_2:
                x = np.round(self.l1*np.cos(q1) + self.l2*np.cos(q1+q2),2)
                y = np.round(self.l1*np.sin(q1) + self.l2*np.sin(q1+q2),2)

                input_position = (x,y)
                output_joints = (np.round(q1,2), np.round(q2,2))
                for sample in range(self.randomsamples):
                    input_joints = (np.round(q1 + 0.2*(np.random.random()-0.5) ,2) ,np.round(q2 + 0.2*(np.random.random()-0.5) ,2))
                    self.data.append( [ input_position , input_joints , output_joints ] )

    def plot_points(self):
        # print(self.data)
        s = np.array(self.data)

        print(s)

        fig = plt.figure()
        plt.xlim([-np.pi,np.pi])
        plt.ylim([-np.pi,np.pi])
        plt.scatter( s[:,1,0], s[:,1,1], color = 'red')
        plt.scatter( s[:,2,0], s[:,2,1], color = 'blue')
        plt.show()

    

generator = DatasetGenerator(20, 20)
generator.plot_points()


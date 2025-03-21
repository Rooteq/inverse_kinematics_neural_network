import numpy as np
import matplotlib.pyplot as plt


def solve_ik(x,y):
    

angle = np.linspace(0,np.pi/2,100)
traj = []
for i in angle:
    traj.append([2*np.cos(i),2*np.sin(i)])   

# print(np.array(traj))

tr = np.array(traj)



# fig = plt.figure()
# plt.xlim([-2.5, 2.5])
# plt.ylim([-2.5,2.5])
# plt.plot( tr[:,0], tr[:,1],'-')
# plt.show()
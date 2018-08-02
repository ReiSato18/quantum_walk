import numpy as np
import matplotlib.pyplot as plt
import math
import itertools

n=50
step= 500
l = 0
#flip flop walk(Szegedy walk)
theta = 3*(math.pi)/12
H = np.array([[np.cos(theta),np.sin(theta)],[np.sin(theta),-np.cos(theta)]])

P = np.zeros((2,2)); P[1,:] = H[1,:]
Q = np.zeros((2,2)); Q[0,:] = H[0,:]
###å›ºå®šç«¯
p = [[0,0],[1,0]]
q = [[0,1],[0,0]]
#######
x_list=[ i for i in range(0,2*n+1)]
t_list=[ i for i in range(0,step+1)]

p_list= np.zeros(2*n+1)
phi_list = np.zeros((2*n+1,2),dtype="complex")
next_phi_list = np.zeros((2*n+1,2),dtype="complex")

phi_list[n] = [1 , 1j]/np.sqrt(2)
p_list[n] = 1.0
###########
for t in range(0,step+1):
    if t == 0:
        pass
    else:
        for x in x_list:
            x1 = (x - 1)
            x2 = (x + 1)
            if x == 0:
                next_phi_list[x] = np.array([np.dot(P,phi_list[x2])])
            elif x == 1:
                next_phi_list[x] = np.array([np.dot(P,phi_list[x2]) + np.dot(q,phi_list[x1])])
            elif x == 2*n:
                next_phi_list[x] = np.array([np.dot(Q,phi_list[x1])])
            elif x == 2*n-1:
                next_phi_list[x] = np.array([np.dot(Q,phi_list[x1]) + np.dot(p,phi_list[x2])])
            else:
                next_phi_list[x] = np.array([np.dot(P,phi_list[x2]) + np.dot(Q,phi_list[x1])])

            p_list[x] = np.real(np.vdot(next_phi_list[x],next_phi_list[x]))
        phi_list = np.copy(next_phi_list)
    print(t,p_list.sum())
    plt.xlabel("x",fontsize="24")
    plt.ylabel("probability",fontsize="24")
    plt.ylim([0,0.25])
    plt.plot(x_list,p_list,color="blue",label="quantum walk",linewidth="1")
    plt.legend(title="flip flop walk",loc="best",fontsize=10)
    plt.tight_layout()
    plt.pause(0.01)
    plt.cla()



########
plt.xlabel("x",fontsize="24")
plt.ylabel("probability",fontsize="24")
#plt.ylim([0,0.25])
plt.plot(x_list,p_list,color="red",label="quantum walk",linewidth="1")
plt.legend(title="flip flop walk",loc="best",fontsize=10)
plt.tight_layout()
#plt.show()

import numpy as np
import matplotlib.pyplot as plt
import itertools
##################
#background; i read reference document written by Takuya Machida
#time evolution descreate quantum walk equation is below
# psi[t+1,x,y] = P*psi[x-1,y,t] + Q*psi[x+1,y,t] + R*psi[x,y+1,t] + S*psi[x,y-1,t]
##################
####################
n = 10
#direction
d = 4
#all vertex i use. stage is N*N.
N = (2*n+1)**2
step = 50
#for plotting
x_list = [i for i in range(0,2*n+1)]
y_list = [i for i in range(0,2*n+1)]
t_list = [i for i in range(1,step+1)]
#to store a marked node
mark_list = []
#to store a non-marked node
p_list=[]
####################
#Grover diffusion
D = np.array([[-1,1,1,1],[1,-1,1,1],[1,1,-1,1],[1,1,1,-1]]) / 2
#Grover diffusion is diveded for four directions,including shift operator (maybe a little unsure)
P = np.zeros((4,4)); P[0,:] = D[0,:] #left
Q = np.zeros((4,4)); Q[1,:] = D[1,:] #right
S = np.zeros((4,4)); S[2,:] = D[2,:] #up
R = np.zeros((4,4)); R[3,:] = D[3,:] #down
###initial state
#initial state to start Grover algorithm
phi_map = np.ones((2*n+1, 2*n+1,d),dtype="complex")
phi_map /= np.sqrt(d*N)
#to store t+1 state of phi_map
next_phi_map = np.zeros((2*n+1, 2*n+1,d),dtype="complex")
###initial probability
p_map = np.zeros([2*n+1,2*n+1])
for i in itertools.product(x_list,y_list):
    p_map[i] = np.real(np.vdot(next_phi_map[i], next_phi_map[i]))
###main calculation
for t in t_list:
    for i in itertools.product(x_list,y_list):
            x = i[0]
            y = i[1]
            #for Boundary conditions
            x1 = (x-1 + 2*n+1) % (2*n+1)
            x2 = (x+1) % (2*n+1)
            y1 = (y-1 + 2*n+1) % (2*n+1)
            y2 = (y+1) % (2*n+1)

            if i == (18,18):#marked node
                next_phi_map[i] = -np.array([np.dot(P, phi_map[x2,y]) + np.dot(Q, phi_map[x1,y]) + np.dot(R, phi_map[x,y2]) + np.dot(S, phi_map[x,y1])])
            else:#the others
                next_phi_map[i] = np.array([np.dot(P, phi_map[x2,y]) + np.dot(Q, phi_map[x1,y]) + np.dot(R, phi_map[x,y2]) + np.dot(S, phi_map[x,y1])])
            p_map[i] = np.real(np.vdot(next_phi_map[i], next_phi_map[i]))
    phi_map = np.copy(next_phi_map)
    #######
    print(t, p_map.sum())
    mark_list.append(np.real(np.vdot(phi_map[18,18] , phi_map[18,18])))
    p_list.append(np.real(np.vdot(phi_map[10,10],phi_map[10,10])))
##########
plt.xlabel("t",fontsize="24")
plt.ylabel("probability",fontsize="24")
plt.plot(t_list,mark_list,label="mark",linewidth="2.0",color="red")
plt.plot(t_list,p_list,label="no mark",linewidth ="2.0",color="black")
plt.legend(title="markedvertexprobablity",loc="best",fontsize=10)
plt.tight_layout()
plt.show()
#########

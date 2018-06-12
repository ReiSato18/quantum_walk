import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

#環境設定
n=100 #x軸
m=100 #t
theta = 3*(math.pi)/12

P = [[np.cos(theta),np.sin(theta)],[0,0]]
Q = [[0,0],[np.sin(theta),-np.cos(theta)]]
x_list=[]#xline
t_list=[]#time
p_list=[]#probability
s_list=[]#state
a = 1/math.sqrt(2)
b = 1j/math.sqrt(2)

#ランダムウォーク
R=1/2
L=1/2
X_list=[]
P_list=[]

#quantumwalk
for j in range(0,2*n+1):
    if j  == n:
        phai = [a ,b]
        pro = 1
    else:
        phai = [0,0]
        pro =0
    p = np.dot(phai,np.conj(phai))

    x_list.append(j)
    X_list.append(j)
    s_list.append(phai)
    p_list.append(p)
    P_list.append(pro)



for t in range(0,m+1):
    t_list.append(t)
    if t ==0:
        s_list
        p_list
        P_list
    else:
        next_s_list = [0]*len(s_list)
        next_P_list = [0]*len(P_list) 
        for i in range(0,2*n+1):
            if i == 0:
                next_s_list[i] = np.dot(P, s_list[i+1])
                next_P_list[i] = P_list[i+1]*L
            elif i == 2*n:
                next_s_list[i] = np.dot(Q, s_list[i-1])
                next_P_list[i] = P_list[i-1]*R
            else:
                next_s_list[i] = np.dot(P, s_list[i+1]) + np.dot(Q, s_list[i-1])
                next_P_list[i] = P_list[i+1]*L + P_list[i-1]*R

            p_list[i] = np.dot(next_s_list[i],np.conj(next_s_list[i]))
        s_list = next_s_list
        P_list = next_P_list



    print(p_list,P_list)

    plt.xlabel("x")
    plt.ylabel("probability")
    plt.ylim([0,0.1])
    plt.xlim([-n,3*n])
    plt.plot( x_list,np.real(p_list),color="red",linewidth=1.0,label="quantum walk")
    plt.plot(X_list, P_list,color="blue",linewidth=1.0,label="random walk")
    plt.legend(loc="best")
    plt.pause(0.01)
    plt.cla()

#plt.ylim([0,0.1])
#plt.xlim([-n,3*n])
#plt.xlabel("x",fontsize=24)
#plt.ylabel("probability",fontsize=24)
#plt.plot(x_list,np.real(p_list),color="red",label="quantum walk",linewidth=0.7)
#plt.plot(X_list,P_list,color="blue",label="random walk",linewidth=0.7)
#plt.legend(title="t=100",loc="best",fontsize=10)
#plt.show()

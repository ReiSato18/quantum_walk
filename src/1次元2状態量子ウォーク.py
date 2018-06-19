import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

#環境設定
n=400
theta = 3*(math.pi)/12

P = [[np.cos(theta),np.sin(theta)],[0,0]]  #Shift Operator
Q = [[0,0],[np.sin(theta),-np.cos(theta)]] #Shift Operator
x_list=[]#xline
t_list=[]#time
p_list=[]#probability
s_list=[]#state
a = 1/math.sqrt(2)   #you can choose any number that is normalized with b
b = 1j/math.sqrt(2)

for x in range(0,2*n+1):
    if x == n:
        phai = [a ,b]
    else:
        phai = [0,0]
    p = np.dot(phai,np.conj(phai))

    x_list.append(x)
    s_list.append(phai)
    p_list.append(p)


for t in range(0,501):
    t_list.append(t)
    if t ==0:
        s_list
        p_list
    else:
        next_s_list = [0]*len(s_list)  ）
        for x in range(0,2*n+1):
            if x == 0:
                next_s_list[0] = np.inner(P, s_list[1])

            elif x == 2*n:
                next_s_list[2*n] = np.inner(Q, s_list[2*n-1])

            else:
                next_s_list[x] = np.inner(P, s_list[x+1]) + np.inner(Q, s_list[x-1])
            p_list[x] = np.dot(next_s_list[x],np.conj(next_s_list[x]))
        s_list = next_s_list


    #if you want to watch the real time move of probability
    #print(p_list)
    #plt.xlabel("x")
    #plt.ylabel("probability")
    #plt.ylim([0,0.1])
    #plt.xlim([-n,3*n])
    #plt.plot(x_list,np.real(p_list),color="red",linewidth=0.3)
    #plt.pause(0.01)
    #plt.cla()

#t=500 only as a figure
plt.xlabel("x")
plt.ylabel("probability")
plt.ylim([0,0.1])
plt.xlim([-n,3*n])
plt.plot(x_list,np.real(p_list),color="red",linewidth=0.3)
plt.show()

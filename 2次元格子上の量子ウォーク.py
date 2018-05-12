import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.animation as animation

#環境設定
n=2  #tの範囲
m=4  #xの範囲

P = [[-1/2, 1/2, 1/2, 1/2],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
Q = [[0,0,0,0],[1/2, -1/2, 1/2, 1/2],[0,0,0,0],[0,0,0,0]]
R = [[0,0,0,0],[0,0,0,0],[1/2, 1/2, -1/2, 1/2],[0,0,0,0]]
S = [[0,0,0,0],[0,0,0,0],[0, 0, 0, 0],[1/2, 1/2, 1/2, -1/2]]

x_list=[]#xline
y_list=[]
t_list=[]#time
p_list=[]#probability
s_list=[]#state
a = 1#1/math.sqrt(2)
b = 0#1j/math.sqrt(2)
c = 0
d = 0

p = np.zeros((2*m+1,2*m+1))

def probability(x,y):
    return p[x,y]

for i in range(0,2*m+1):
    if i == m:
        phi = [a ,b, c, d]
    else:
        phi = [0, 0, 0, 0]
    p1 = np.dot(phi,np.conj(phi))

    x_list.append(i)
    y_list.append(i)
    s_list.append(phi)
    p_list.append(p1)

def probability(x,y):
    return p[x,y]

for t in range(0, n+1):
    t_list.append(t)
    if t == 0:
        s_list
        p_list
        #probability(m,m)
    else:
        next_s_list = [0]*len(s_list)
        for x in range(0,2*m+1):
            if x == 0:
                for y in range(0,2*m+1):   #x=0, 0<= y =<2*m
                    if y == 0:
                        next_s_list[y] = np.inner(P, s_list[x+1]) + np.inner(R, s_list[y+1])
                    elif y == 2*m:
                        next_s_list[y] = np.inner(P, s_list[x+1]) + np.inner(S, s_list[y-1])
                    else:
                        next_s_list[y] = np.inner(P, s_list[x+1]) + np.inner(R,s_list[y+1]) +np.inner(S, s_list[y-1])
                #probability(x,y) = np.dot(next_s_list[y], np.conj(next_s_list[y]))
            elif x == 2*m:                  #x=2*m, 0<=y<=2*m
                for y in range(0, 2*m+1):
                    if y == 0:
                        next_s_list[y] = np.inner(Q, s_list[x-1]) + np.inner(R, s_list[y+1])
                    elif y == 2*m:
                        next_s_list[y] = np.inner(Q, s_list[x-1]) + np.inner(S, s_list[y-1])
                    else:
                        next_s_list[y]  = np.inner(Q, s_list[x-1]) + np.inner(R,s_list[y+1]) +np.inner(S, s_list[y-1])
                #probability(x,y) = np.dot(next_s_list[y], np.conj(next_s_list[y]))
            else:
                if y == 0:
                    next_s_list[y] = np.inner(Q,s_list[x-1]) + np.inner(P,s_list[x+1])
                elif y == 2*m:
                    next_s_list[y] = np.inner(Q,s_list[x-1]) + np.inner(P,s_list[x+1]) + np.inner(S,s_list[y-1])
                else:
                    next_s_list[y]  = np.inner(P, s_list[x+1]) + np.inner(Q, s_list[x-1]) + np.inner(R, s_list[y+1]) + np.inner(S,s_list[y-1])
                 #probability(x,y)= np.dot(next_s_list[y], np.conj(next_s_list[y]))
            p_list[x] = np.dot(next_s_list[y], np.conj(next_s_list[y]))
        s_list = next_s_list

    print(t,x_list,p_list)



#3Dplot

fig = plt.figure()
ax = Axes3D(fig)
#X,Y = np.meshgrid(x_list,y_list)
#print(X,p_list)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("probability")

ax.set_xlim(2*m,0)
ax.set_ylim(0,2*m)
ax.set_zlim(0,1)
ax.plot(x_list, y_list, p_list, color ="red", linewidth=1)
plt.show()

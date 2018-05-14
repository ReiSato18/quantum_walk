import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#import matplotlib.animation as animation

#環境設定
n=10  #tの範囲
m=15  #偶数

P = [[-1/2, 1/2, 1/2, 1/2],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
Q = [[0,0,0,0],[1/2, -1/2, 1/2, 1/2],[0,0,0,0],[0,0,0,0]]
R = [[0,0,0,0],[0,0,0,0],[1/2, 1/2, -1/2, 1/2],[0,0,0,0]]
S = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[1/2, 1/2, 1/2, -1/2]]

t_list = []
x_list = []
y_list = []

phi_map = np.zeros((2*m+1, 2*m+1,4)) #np.zeros((行,列,[]の中身の数))
phi_map[m,m]= np.array([1,0,0,0])

p_map=np.zeros([2*m+1,2*m+1])

for i in range(0,2*m+1):
    p = np.dot(phi_map[i,i], np.conj(phi_map[i,i]))
    p_map[i,i]=p
    x_list.append(i)
    y_list.append(i)
#print(p_map)

for t in range(0,n+1):
    t_list.append(t)
    if t == 0:
        phi_map
        p_map
    else:
        next_phi_map = np.zeros((2*m+1,2*m+1, 4))
        for x in range(0,2*m+1):
            if x == 0:
                for y in range(0,2*m+1):
                    if y == 0:
                        next_phi_map[x,y] = np.array([np.inner(P, phi_map[x+1,y]) + np.inner(R, phi_map[x,y+1])])
                    elif y == 2*m:
                        next_phi_map[x,y] = np.array([np.inner(P, phi_map[x+1,y]) + np.inner(S, phi_map[x,y-1])])
                    else:
                        next_phi_map[x,y] = np.array([np.inner(P, phi_map[x+1,y]) + np.inner(S, phi_map[x,y-1]) + np.inner(R, phi_map[x,y+1])])
            elif x == 2*m:
                for y in range(0,2*m+1):
                    if y == 0:
                        next_phi_map[x,y] = np.array([np.inner(Q, phi_map[x-1,y]) + np.inner(R, phi_map[x,y+1])])
                    elif y == 2*m:
                        next_phi_map[x,y] = np.array([np.inner(Q, phi_map[x-1,y]) + np.inner(S, phi_map[x,y-1])])
                    else:
                        next_phi_map[x,y] = np.array([np.inner(Q, phi_map[x-1,y]) + np.inner(S, phi_map[x,y-1]) + np.inner(R, phi_map[x,y+1])])
            else:
                for y in range(0,2*m+1):
                    if y == 0:
                        next_phi_map[x,y] = np.array([np.inner(P, phi_map[x+1,y]) + np.inner(Q, phi_map[x-1,y]) + np.inner(R, phi_map[x,y+1])])
                    elif y == 2*m:
                        next_phi_map[x,y] = np.array([np.inner(P, phi_map[x+1,y]) + np.inner(Q, phi_map[x-1,y]) + np.inner(S, phi_map[x,y-1])])
                    else:
                        next_phi_map[x,y] = np.array([np.inner(P, phi_map[x+1,y]) + np.inner(Q, phi_map[x-1,y]) + np.inner(R, phi_map[x,y+1]) + np.inner(S, phi_map[x,y-1])])
                    p_map[x,y] = np.dot(next_phi_map[x,y], np.conj(next_phi_map[x,y]))
        phi_map = next_phi_map

    print(t,p_map)
　　
　　#real-time(no finished)
    
    #fig = plt.figure()
    #ax = Axes3D(fig)
    #X,Y = np.meshgrid(x_list,y_list)

    #ax.set_xlabel("x")
    #ax.set_ylabel("y")
    #ax.set_zlabel("probability")

    #ax.set_xlim(2*m,0)
    #ax.set_ylim(0,2*m)
    #ax.set_zlim(0,1)
    #ax.plot_wireframe(X, Y, p_map, color ="red", linewidth=1)
    #plt.pause(0.01)
    #plt.cla()





fig = plt.figure()
ax = Axes3D(fig)
X,Y = np.meshgrid(x_list, y_list)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("probability")

ax.set_xlim(2*m,0)
ax.set_ylim(0,2*m)
ax.set_zlim(0,0.35)
#ax = fig.add_subplot((2*m,2*m), projection="3d")
#surf = ax.plot_surface(X, Y, p_map, cmap =cm.coolwarm , linewidth=0)

#ax.bar3d(x_list,y_list,p_map,dx,dy,dz)
#fig.colorbar(surf)
ax.bar3d(X.ravel(), Y.ravel(), p_map.ravel() ,0.1, 0.1, -p_map.ravel(),color ="red")#,cmap=cm.hot)
#ax.w_xaxis.set_pane_color((0,0,0,0))
#ax1.w_yaxis.set_pane_color((0., 0., 0., 0.))
#ax1.w_zaxis.set_pane_color((0., 0., 0., 0.))
plt.show()

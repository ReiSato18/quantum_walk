import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import matplotlib.colors as colors
#############################
#quantumwalk on cubic scaled n*n*n
n=3
p_map = np.zeros((2*n+1)*(2*n+1)*(2*n+1)).reshape(2*n+1,2*n+1,2*n+1) 
phi_map = np.zeros(((2*n+1),(2*n+1),(2*n+1),6),dtype="complex")
phi_map[n,n,n]=np.array([1,0,0,0,0,0])
p_map[n,n,n]=np.real(np.inner(phi_map[n,n,n],np.conj(phi_map[n,n,n])))
r_list=[]#x,y,zの長さ収納用
t_list=[]
x_list=[i for i in range(2*n+1)]
y_list=[i for i in range(2*n+1)]
z_list=[i for i in range(2*n+1)]
#t_list=[i for i in range(n+1)]
#quantumcoin
L=[[-2/3,1/3,1/3,1/3,1/3,1/3],[0,0,0,0,0,0],[0,0,0,0,0,0], #Left
  [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
R=[[0,0,0,0,0,0],[1/3,-2/3,1/3,1/3,1/3,1/3],[0,0,0,0,0,0], #Right
  [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
F=[[0,0,0,0,0,0],[0,0,0,0,0,0],[1/3,1/3,-2/3,1/3,1/3,1/3], #Forward
  [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
B=[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
  [1/3,1/3,1/3,-2/3,1/3,1/3],[0,0,0,0,0,0],[0,0,0,0,0,0]] #Back
U=[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
  [0,0,0,0,0,0],[1/3,1/3,1/3,1/3,-2/3,1/3],[0,0,0,0,0,0]] #Up
D=[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
  [0,0,0,0,0,0],[0,0,0,0,0,0],[1/3,1/3,1/3,1/3,1/3,-2/3]] #Down

for t in range(0,n+1):
    t_list.append(t)
    if t==0:
        p_map
        phi_map
    else:
        next_phi_map = np.zeros(((2*n+1),(2*n+1),(2*n+1),6),dtype="complex")
        for x in range(0,2*n+1):
            if x == 0:
                for y in range(0,2*n+1):
                    if y == 0:
                        for z in range(0,2*n+1):
                            if z ==0:
                                next_phi_map[x,y,z]=np.array([np.dot(L,phi_map[x+1,y,z])+np.dot(B,phi_map[x,y+1,z])+np.dot(D,phi_map[x,y,z+1])])
                            elif z == 2*n:
                                next_phi_map[x,y,z]=np.array([np.dot(L,phi_map[x+1,y,z])+np.dot(B,phi_map[x,y+1,z])+np.dot(U,phi_map[x,y,z-1])])
                            else:
                                next_phi_map[x,y,z]=np.array([np.dot(L,phi_map[x+1,y,z])+np.dot(B,phi_map[x,y+1,z])+
                                                              np.dot(U,phi_map[x,y,z-1])+np.dot(D,phi_map[x,y,z+1])])
                    elif y == 2*n:
                        for z in range(0,2*n+1):
                            if z ==0:
                                next_phi_map[x,y,z]=np.array([np.dot(L,phi_map[x+1,y,z])+np.dot(F,phi_map[x,y-1,z])+np.dot(D,phi_map[x,y,z+1])])
                            elif z == 2*n:
                                next_phi_map[x,y,z]=np.array([np.dot(L,phi_map[x+1,y,z])+np.dot(F,phi_map[x,y-1,z])+np.dot(U,phi_map[x,y,z-1])])
                            else:
                                next_phi_map[x,y,z]=np.array([np.dot(L,phi_map[x+1,y,z])+np.dot(F,phi_map[x,y-1,z])+
                                                              np.dot(U,phi_map[x,y,z-1])+np.dot(D,phi_map[x,y,z+1])])
                    else:
                        for z in range(0,2*n+1):
                            if z ==0:
                                next_phi_map[x,y,z]=np.array([np.dot(L,phi_map[x+1,y,z])+np.dot(F,phi_map[x,y-1,z])+
                                                              np.dot(B,phi_map[x,y+1,z])+np.dot(D,phi_map[x,y,z+1])])
                            elif z == 2*n:
                                next_phi_map[x,y,z]=np.array([np.dot(L,phi_map[x+1,y,z])+np.dot(F,phi_map[x,y-1,z])+
                                                              np.dot(B,phi_map[x,y+1,z])+np.dot(U,phi_map[x,y,z-1])])
                            else:
                                next_phi_map[x,y,z]=np.array([np.dot(L,phi_map[x+1,y,z])+np.dot(F,phi_map[x,y-1,z])+np.dot(B,phi_map[x,y+1,z])+
                                                              np.dot(U,phi_map[x,y,z-1])+np.dot(D,phi_map[x,y,z+1])])
            elif x == 2*n:
                for y in range(0,2*n+1):
                    if y == 0:
                        for z in range(0,2*n+1):
                            if z ==0:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(B,phi_map[x,y+1,z])+np.dot(D,phi_map[x,y,z+1])])
                            elif z == 2*n:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(B,phi_map[x,y+1,z])+np.dot(U,phi_map[x,y,z-1])])
                            else:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(B,phi_map[x,y+1,z])+
                                                              np.dot(U,phi_map[x,y,z-1])+np.dot(D,phi_map[x,y,z+1])])
                    elif y == 2*n:
                        for z in range(0,2*n+1):
                            if z ==0:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(F,phi_map[x,y-1,z])+np.dot(D,phi_map[x,y,z+1])])
                            elif z == 2*n:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(F,phi_map[x,y-1,z])+np.dot(U,phi_map[x,y,z-1])])
                            else:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(F,phi_map[x,y-1,z])+
                                                              np.dot(U,phi_map[x,y,z-1])+np.dot(D,phi_map[x,y,z+1])])
                    else:
                        for z in range(0,2*n+1):
                            if z ==0:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(F,phi_map[x,y-1,z])+
                                                              np.dot(B,phi_map[x,y+1,z])+np.dot(D,phi_map[x,y,z+1])])
                            elif z == 2*n:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(F,phi_map[x,y-1,z])+
                                                              np.dot(B,phi_map[x,y+1,z])+np.dot(U,phi_map[x,y,z-1])])
                            else:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(B,phi_map[x,y+1,z])+
                                                              np.dot(F,phi_map[x,y-1,z])+np.dot(U,phi_map[x,y,z-1])+np.dot(D,phi_map[x,y,z+1])])
            else:#xz平面
                for y in range(0,2*n+1):
                    if y == 0:
                        for z in range(0,2*n+1):
                            if z == 0:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(L,phi_map[x+1,y-1,z])+
                                                              np.dot(B,phi_map[x,y+1,z])+np.dot(D,phi_map[x,y,z+1])])
                            elif z == 2*n:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(L,phi_map[x+1,y,z])+
                                                              np.dot(B,phi_map[x,y+1,z])+np.dot(U,phi_map[x,y,z-1])])
                            else:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(L,phi_map[x+1,y,z])+np.dot(B,phi_map[x,y+1,z])+
                                                              np.dot(U,phi_map[x,y,z-1])+np.dot(D,phi_map[x,y,z+1])])
                    elif y == 2*n:
                        for z in range(0,2*n+1):
                            if z == 0:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(L,phi_map[x+1,y-1,z])+
                                                              np.dot(F,phi_map[x,y-1,z])+np.dot(D,phi_map[x,y,z+1])])
                            elif z == 2*n:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(L,phi_map[x+1,y,z])+
                                                              np.dot(F,phi_map[x,y-1,z])+np.dot(U,phi_map[x,y,z-1])])
                            else:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(L,phi_map[x+1,y,z])+np.dot(F,phi_map[x,y-1,z])+
                                                              np.dot(U,phi_map[x,y,z-1])+np.dot(D,phi_map[x,y,z+1])])
                    else:
                        for z in range(0,2*n+1):
                            if z == 0:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(L,phi_map[x+1,y-1,z])+
                                                              np.dot(F,phi_map[x,y-1,z])+np.dot(B,phi_map[x,y+1,z])+np.dot(D,phi_map[x,y,z+1])])
                            elif z == 2*n:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(L,phi_map[x+1,y,z])+
                                                              np.dot(F,phi_map[x,y-1,z])+np.dot(B,phi_map[x,y+1,z])+np.dot(U,phi_map[x,y,z-1])])
                            else:
                                next_phi_map[x,y,z]=np.array([np.dot(R,phi_map[x-1,y,z])+np.dot(L,phi_map[x+1,y,z])+np.dot(F,phi_map[x,y-1,z])+
                                                              np.dot(B,phi_map[x,y+1,z])+np.dot(U,phi_map[x,y,z-1])+np.dot(D,phi_map[x,y,z+1])])
                            p_map[x,y,z]=np.real(np.inner(next_phi_map[x,y,z],np.conj(next_phi_map[x,y,z])))
        phi_map = next_phi_map
    print(t,p_map)
  
fig = plt.figure()
ax = Axes3D(fig, rect=(0.1,0.1,0.8,0.8))
X,Y,Z=np.meshgrid(x_list,y_list,z_list)
ax.set_xlabel("x",labelpad=10,fontsize=24)
ax.set_ylabel("y",labelpad=10,fontsize=24)
ax.set_zlabel("z",labelpad=10,fontsize=24)
ax.set_xlim(2*n,0)
ax.set_ylim(0,2*n)
ax.set_zlim(0,2*n)
ax.w_xaxis.set_pane_color((0, 0, 0, 0))
ax.w_yaxis.set_pane_color((0, 0, 0, 0))
ax.w_zaxis.set_pane_color((0, 0, 0, 1))
ax.grid(color="white")
ax.grid(False)
mask =p_map>0.0
#####################
mask =p_map>0.0
offset = p_map[mask].ravel() + np.abs(p_map[mask].min())
fracs = offset.astype(float)/offset.max()
norm = colors.Normalize(fracs.min(), fracs.max())
clrs = cm.flag(norm(fracs))
####################
ax.scatter(X[mask].ravel(),Y[mask].ravel(),Z[mask].ravel(), p_map[mask].ravel(), color="red")
plt.show()

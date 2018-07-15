import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
#######できてない###########
stage1=np.array([[1,1,1],[1,0,1],[1,1,1]],dtype="float")
b=np.zeros([3,3])
#######
c=np.vstack([stage1,stage1,stage1])#横に結合
e=np.hstack([stage1,b,stage1])#縦に結合
#######
stage2=np.hstack([c,e.T,c])
#######
h = np.zeros([9,9])
f = np.vstack([stage2,stage2,stage2])
g = np.hstack([stage2,h,stage2])
stage3=np.hstack([f,g.T,f])
#######
H = np.zeros([27,27])
J = np.vstack([stage3,stage3,stage3])
K = np.hstack([stage3,H,stage3])
stage4 = np.hstack([J,K.T,J])
#######
n = 26
m = 4
######
x_list=[i for i in range(3*n+3)]
y_list=[i for i in range(3*n+3)]
t_list=[i for i in range(m+1)]
p_list=[]
#######
P = [[-1/2, 1/2, 1/2, 1/2],[0,0,0,0],[0,0,0,0],[0,0,0,0]] #right
Q = [[0,0,0,0],[1/2, -1/2, 1/2, 1/2],[0,0,0,0],[0,0,0,0]] #left
R = [[0,0,0,0],[0,0,0,0],[1/2, 1/2, -1/2, 1/2],[0,0,0,0]] #down
S = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[1/2, 1/2, 1/2, -1/2]] #up
#######
p_map = np.zeros([3*n+3,3*n+3])
phi_map = np.zeros((3*n+3,3*n+3,4),dtype="complex")
next_phi_map = np.zeros((3*n+3,3*n+3, 4),dtype="complex")
phi_map[0,0]=np.array([1,0,0,0])
p_map[0,0]=np.real(np.inner(phi_map[0,0],np.conj(phi_map[0,0])))
##########
for t in range(0,m+1):
    if t == 0:
        p_map
        phi_map
    else:
        for x in range(0,3*n+3):
            if x == 0:
                for y in range(0,3*n+3):
                    if y == 0:
                        next_phi_map[x,y]=np.array([np.dot(R,phi_map[x,y+1])+np.dot(Q,phi_map[x+1,y])])
                    elif y == 3*n+2:
                        next_phi_map[x,y]=np.array([np.dot(S,phi_map[x,y-1])+np.dot(Q,phi_map[x+1,y])])
                    else:
                        if stage4[x+1,y] == 0:
                            next_phi_map[x,y]=np.array([np.dot(R,phi_map[x,y+1])+np.dot(S,phi_map[x,y-1])])
                        else:
                            next_phi_map[x,y]=np.array([np.dot(Q,phi_map[x+1,y])+np.dot(R,phi_map[x,y+1])+np.dot(S,phi_map[x,y-1])])
            elif x == 3*n+2:
                for y in range(0,3*n+3):
                    if y == 0:
                        next_phi_map[x,y]=np.array([np.dot(R,phi_map[x,y+1])+np.dot(P,phi_map[x-1,y])])
                    elif y == 3*n+2:
                        next_phi_map[x,y]=np.array([np.dot(S,phi_map[x,y-1])+np.dot(P,phi_map[x-1,y])])
                    else:
                        if stage4[x-1,y]==0:
                            next_phi_map[x,y]=np.array([np.dot(R,phi_map[x,y+1])+np.dot(S,phi_map[x,y-1])])
                        else:
                            next_phi_map[x,y]=np.array([np.dot(P,phi_map[x-1,y])+np.dot(R,phi_map[x,y+1])+np.dot(S,phi_map[x,y-1])])
            else:
                for y in range(0,3*n+3):
                    if y == 0:
                        if stage4[x,y+1] == 0:
                            next_phi_map[x,y]=np.array([np.dot(P,phi_map[x-1,y])+np.dot(Q,phi_map[x+1,y])])
                        else:
                            next_phi_map[x,y]=np.array([np.dot(Q,phi_map[x+1,y])+np.dot(R,phi_map[x,y+1])+np.dot(P,phi_map[x-1,y])])
                    elif y == 3*n+2:
                        if stage4[x,y-1] == 0:
                            next_phi_map[x,y]=np.array([np.dot(P,phi_map[x-1,y])+np.dot(Q,phi_map[x+1,y])])
                        else:
                            next_phi_map[x,y]=np.array([np.dot(Q,phi_map[x+1,y])+np.dot(Q,phi_map[x,y-1])+np.dot(P,phi_map[x-1,y])])
                    else:
                        #if stage4[x,y] == 0:
                            #next_phi_map[x,y] == np.array([0,0,0,0])
                        #if:
                        if stage4[x,y-1] == 0:
                            next_phi_map[x,y]=np.array([np.dot(Q,phi_map[x+1,y])+np.dot(R,phi_map[x,y+1])+np.dot(P,phi_map[x-1,y])])
                        elif stage4[x,y+1] == 0:
                            next_phi_map[x,y]=np.array([np.dot(Q,phi_map[x+1,y])+np.dot(S,phi_map[x,y-1])+np.dot(P,phi_map[x-1,y])])
                            ###########
                        elif stage4[x-1,y] == 0 and stage4[x,y-1] == 0:
                            next_phi_map[x,y]==np.array([0,0,0,0])
                        elif stage4[x-1,y]==0 and stage4[x,y+1]==0:
                            next_phi_map[x,y]==np.array([0,0,0,0])
                        elif stage4[x+1,y]==0 and stage4[x,y+1]==0:
                            next_phi_map[x,y]==np.array([0,0,0,0])
                        elif stage4[x+1,y]==0 and stage4[x,y-1]==0:
                            next_phi_map[x,y]==np.array([0,0,0,0])
                            ############
                        elif stage4[x+1,y]==0 and stage4[x-1,y]==0 and stage4[x,y+1]==0:
                            next_phi_map[x,y]==np.array([0,0,0,0])
                        elif stage4[x+1,y]==0 and stage4[x-1,y]==0 and stage4[x,y-1]==0:
                            next_phi_map[x,y]==np.array([0,0,0,0])
                        elif stage4[x,y+1]==0 and stage4[x,y-1]==0 and stage4[x-1,y]==0:
                            next_phi_map[x,y]==np.array([0,0,0,0])
                        elif stage4[x,y+1]==0 and stage4[x,y-1]==0 and stage4[x+1,y]==0:
                            next_phi_map[x,y]==np.array([0,0,0,0])
                            #############
                        elif  stage4[x+1,y]==0 and stage4[x-1,y]==0 and stage4[x,y+1]==0 and stage4[x,y-1]==0:
                            next_phi_map[x,y] == np.array([0,0,0,0])
                            #############
                        elif stage4[x-1,y] == 0:
                            next_phi_map[x,y]=np.array([np.dot(Q,phi_map[x+1,y])+np.dot(R,phi_map[x,y+1])+np.dot(S,phi_map[x,y-1])])
                        elif stage4[x+1,y] == 0:
                            next_phi_map[x,y]=np.array([np.dot(P,phi_map[x-1,y])+np.dot(R,phi_map[x,y+1])+np.dot(S,phi_map[x,y-1])])
                        elif stage4[x,y-1] == 0 and stage4[x,y+1] == 0:
                            next_phi_map[x,y]=np.array([np.dot(P,phi_map[x-1,y])+np.dot(Q,phi_map[x+1,y])])
                        elif stage4[x-1,y] == 0 and stage4[x+1,y] == 0:
                            next_phi_map[x,y]=np.array([np.dot(R,phi_map[x,y+1])+np.dot(S,phi_map[x,y-1])])
                        else:
                            next_phi_map[x,y]=np.array([np.dot(P,phi_map[x-1,y])+np.dot(Q,phi_map[x+1,y])+np.dot(R,phi_map[x,y+1])+np.dot(S,phi_map[x,y-1])])
                                #phi_map = next_phi_map
                            p_map[x,y]=np.real(np.inner(next_phi_map[x,y], np.conj(next_phi_map[x,y])))
    #p_list.append(np.real(np.inner(phi_map[26,26], np.conj(phi_map[26,26]))))
            phi_map = next_phi_map
    print(t,p_map)
#for t in range(0,m+1):
    #if t == 0:
        #p_map
        #phi_map
    #else:
        #for i in range(0,3*n+3):
            #p_map[i,i]=np.real(np.inner(phi_map[i,i], np.conj(phi_map[i,i])))
    #print(t,p_map)
#plt.xlabel("t",fontsize="24")
#plt.ylabel("probability",fontsize="24")
#plt.ylim([0,0.0000001])
#plt.plot(t_list,np.real(p_list),color="red",label="quantum walk",linewidth="1")
#plt.legend(title="x=26,y=26",loc="best",fontsize=10)
#plt.tight_layout()
#plt.show()
##############
fig = plt.figure()
ax = Axes3D(fig, rect=(0.1,0.1,0.8,0.8))
X,Y = np.meshgrid(x_list, y_list)
ax.set_xlabel("x",labelpad=10,fontsize=24)
ax.set_ylabel("y",labelpad=20,fontsize=24)
ax.set_zlabel("$|\psi|^2$",labelpad=10,fontsize=24)
ax.set_xlim(3*n+2,0)
ax.set_ylim(0,3*n+2)
ax.set_zlim(0,1)
#############
mask= p_map > 0.0
offset = p_map[mask].ravel() + np.abs(p_map[mask].min())
fracs = offset.astype(float)/offset.max()
norm = colors.Normalize(fracs.min(), fracs.max())
clrs = cm.cool(norm(fracs))
###########
ax.bar3d(X[mask].ravel(), Y[mask].ravel(), p_map[mask].ravel() ,1, 1, -p_map[mask].ravel(),color =clrs)
ax.w_xaxis.set_pane_color((0, 0, 0, 0))
ax.w_yaxis.set_pane_color((0, 0, 0, 0))
ax.w_zaxis.set_pane_color((0, 0, 0, 1))
ax.grid(color="white")
ax.grid(False)
############
plt.show()

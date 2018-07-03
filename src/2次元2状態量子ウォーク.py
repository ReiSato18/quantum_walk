import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
#まだちょっとおかしい

#環境設定
n=50  #tの範囲
m=50  #xy
r=1

theta=3*(math.pi)/12
#いろいろ準備
t_list=[]
x_list=[]
y_list=[]
p_spot=[]

phi_map =np.zeros((2*m+1,2*m+1,2),dtype="complex")
phi_map[m,m]=[1,0]  #初期状態
p_map=np.zeros([2*m+1,2*m+1])

#量子コイン
P = [[math.cos(theta),-1j*math.sin(theta)],
    [0,0]]                                  #+xに進む
Q = [[0,0],
    [-1j*math.sin(theta), math.cos(theta)]] #-xに進む
#量子コイン
R = [[1/math.sqrt(r), math.sqrt((r-1)/r)],
    [0,0]]                                 #+yに進む
S = [[0,0],
    [math.sqrt((r-1)/r), -1/math.sqrt(r)]] #留まる

for i in range(0,2*m+1):
    p=np.inner(phi_map[i,i],np.conj(phi_map[i,i]))
    p_map[i,i]=np.real(p) #もしかして、2状態じゃおかしい？
    x_list.append(i)
    y_list.append(i)

for t in range(0,n+1):
    t_list.append(t)
    if t == 0:
        phi_map
        p_map
    else:
        next_phi_map = np.zeros((2*m+1,2*m+1,2),dtype="complex") #x軸の変化
        final_phi_map= np.zeros((2*m+1,2*m+1,2),dtype="complex") #最終
        for x in range(0,2*m+1):
            if x == 0:
                for y in range(m,2*m+1):
                    if y == m:
                        next_phi_map[x,y] = np.array([np.dot(Q, phi_map[x+1,y])])
                        #final_phi_map[x,y+1] = np.array([np.dot(R, next_phi_map[x,y])]) #この計算はelseの場所に含まれる
                        final_phi_map[x,y] = np.array([np.dot(S, next_phi_map[x,y])])#+np.array([np.dot(R, next_phi_map[x,y-1])])
                    elif y ==2*m:
                        next_phi_map[x,y] = np.array([np.dot(Q,phi_map[x+1,y])])
                        final_phi_map[x,y] = np.array([np.dot(S,next_phi_map[x,y])+np.dot(R,next_phi_map[x,y-1])])
                        #final_phi_map[x,y+1]=np.array([np.dot(S,next_phi_map[x,y])])
                    else:
                        next_phi_map[x,y] = np.array([np.dot(Q, phi_map[x+1,y])])
                        final_phi_map[x,y] = np.array([np.dot(R,next_phi_map[x,y-1])+np.dot(S,next_phi_map[x,y])])
            elif x == 2*m:
                for y in range(m,2*m+1):
                    if y == m:
                        next_phi_map[x,y] = np.array([np.dot(P, phi_map[x-1,y])])
                        #final_phi_map[x,y+1] = np.array([np.dot(R,next_phi_map[x,y])])
                        final_phi_map[x,y] = np.array([np.dot(S,next_phi_map[x,y])])
                    elif y ==2*m:
                        next_phi_map[x,y] = np.array([np.dot(P,phi_map[x-1,y])])
                        final_phi_map[x,y] = np.array([np.dot(S,next_phi_map[x,y])+np.dot(R,next_phi_map[x,y-1])])
                        #final_phi_map[x,y+1] = np.array([np.dot(S,next_phi_map[x,y])])
                    else:
                        next_phi_map[x,y] = np.array([np.dot(P, phi_map[x-1,y])])
                        final_phi_map[x,y] = np.array([np.dot(R,next_phi_map[x,y-1])+np.dot(S,next_phi_map[x,y])])
                        #final_phi_map[x,y] =np.array([np.dot(S,next_phi_map[x,y])])
            else:
                for y in range(m,2*m+1):
                    if y == m:
                        next_phi_map[x,y] = np.array([np.dot(P,phi_map[x-1,y])+np.dot(Q,phi_map[x+1,y])])
                        #final_phi_map[x,y+1] = np.array([np.dot(R,next_phi_map[x,y])])
                        final_phi_map[x,y] = np.array([np.dot(S,next_phi_map[x,y])])#+np.array([np.dot(R,next_phi_map[x,y-1])])
                    elif y == 2*m:
                        next_phi_map[x,y] = np.array([np.dot(P,phi_map[x-1,y])+np.dot(Q,phi_map[x+1,y])])
                        final_phi_map[x,y]= np.array([np.dot(S,next_phi_map[x,y])+np.dot(R,next_phi_map[x,y-1])])
                    else:
                        next_phi_map[x,y] = np.array([np.dot(P,phi_map[x-1,y])+np.dot(Q,phi_map[x+1,y])])
                        final_phi_map[x,y] = np.array([np.dot(R,next_phi_map[x,y-1])+np.dot(S,next_phi_map[x,y])])
                        #final_phi_map[x,y] = np.array([np.dot(S,next_phi_map[x,y])])


                    p_map[x,y]= np.real(np.inner(final_phi_map[x,y], np.conj(final_phi_map[x,y])))
        phi_map=final_phi_map
    #p_spot.append(p_map[80,80])
    #print(t,p_spot)

#plt.xlabel("t",fontsize="24")
#plt.ylabel("probability",fontsize="24")
#plt.ylim([0,0.0001])
#plt.xlim([-2,n])
#plt.plot(t_list,np.real(p_spot),color="red",linewidth=0.7)
#plt.tight_layout()
#plt.show()



#3次元プロット
fig = plt.figure()
ax = Axes3D(fig, rect=(0.1,0.1,0.8,0.8)) #rect=(x0,y0,width,height)
X,Y = np.meshgrid(x_list, y_list)
ax.set_xlabel("x",labelpad=10,fontsize=24)
ax.set_ylabel("y",labelpad=10,fontsize=24)
ax.set_zlabel("$|\psi|^2$",labelpad=10,fontsize=18)
ax.set_xlim(2*m,0)
ax.set_ylim(0,3*m)
ax.set_zlim(0,0.01)
#
mask = p_map > 0.0
offset = p_map[mask].ravel() + np.abs(p_map[mask].min())
fracs = offset.astype(float)/offset.max()
norm = colors.Normalize(fracs.min(), fracs.max())
clrs = cm.cool(norm(fracs))
ax.w_xaxis.set_pane_color((0, 0, 0, 0))
ax.w_yaxis.set_pane_color((0, 0, 0, 0))
ax.w_zaxis.set_pane_color((0, 0, 0, 1))
ax.grid(color="white")
ax.grid(False)
#p_map[np.abs(p_map)<=0.0]=np.nan
#np.nanmin(p_map,axis=None,out=None)

ax.bar3d(Y[mask].ravel(), X[mask].ravel(), p_map[mask].ravel() ,1, 1, -p_map[mask].ravel(),color=clrs)
plt.show()

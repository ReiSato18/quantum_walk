import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib import cm

#環境設定
n=50 #x軸
m=50 #t
theta = 3*(math.pi)/12
p_spot=[]
P_spot=[]

P = [[np.cos(theta),np.sin(theta)],[0,0]]
Q = [[0,0],[np.sin(theta),-np.cos(theta)]]
x_list=[]#xline
t_list=[]#time
p_list=[]#probability
s_list=[]#state
a = 1/math.sqrt(2)
b = 1j/math.sqrt(2)
p_map=[]
pp_map =np.zeros([2*m+1,2*m+1])#,dtype="complex")
z=np.zeros([2*m+1,2*m+1])
#ランダムウォーク
R=1/2
L=1/2
X_list=[]
P_list=[]
#step_list=[]

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



for t in range(0,2*m+1):
    t_list.append(t)
    if t ==0:
        s_list
        p_list
        P_list
    else:
        next_s_list = [0]*len(s_list)
        next_P_list = [0]*len(P_list) #listと同じ要素の数ですべて0を用意（初期化）
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
            #pp_map[t]=p_list


        s_list = next_s_list
        P_list = next_P_list
    pp_map[t]=np.real(p_list)
    #print(t,np.real(pp_map),np.real(p_list))

    #原点での確率変化
    #p_map.append(p_list)
    #p_spot.append(p_list[50]) #ひとつの座標での確率の動きを見たい場合
    #P_spot.append(P_list[50])
    #print(t,p_list[0])
#plt.xlabel("t",fontsize="24")
#plt.ylabel("probability",fontsize="24")
#plt.plot(t_list,np.real(p_spot),color="red",label="quantum walk",linewidth="1")
#plt.plot(t_list,P_spot,color="blue",label="random walk",linewidth="1.8")
#plt.legend(title="x=50",loc="best",fontsize=10)
#plt.tight_layout()
#plt.show()
    #plt.pause(0.1)
    #plt.cla()

#p_map=np.array(p_map)
#print(p_map)

#for i in range(0,m+1):
    #pp_map[i,i]=p_map[i,i]

    #plt.xlabel("x")
    #plt.ylabel("probability")
    #plt.ylim([0,0.1])
    #plt.xlim([-n,3*n])
    #plt.plot( x_list,np.real(p_list),color="red",linewidth=1.0,label="quantum walk")
    #plt.plot(X_list, P_list,color="blue",linewidth=1.0,label="random walk")
    #plt.legend(loc="best")
    #plt.pause(0.01)
    #plt.cla()

#plt.ylim([0,0.1])
#plt.xlim([-n,3*n])
#plt.xlabel("x",fontsize=24)
#plt.ylabel("probability",fontsize=24)
#plt.plot(x_list,np.real(p_list),color="red",label="quantum walk",linewidth=0.7)
#plt.plot(X_list,P_list,color="blue",label="random walk",linewidth=0.7)
#plt.legend(title="t=100",loc="best",fontsize=10)
#plt.show()

#plt.xlabel("t")
#plt.ylabel("probability")
#plt.xlim([0,2*n])
#plt.ylim([0,1])
#plt.plot(t_list,np.real(p_list),color="red",label="quantumwalk")
#plt.plot(t_list,P_list,color="blue",label="random walk")
#plt.legend(loc="best")
#plt.show()


#3次元
#3次元barプロット
fig = plt.figure()
ax = Axes3D(fig, rect=(0.1,0.1,0.8,0.8)) #rect=(x0,y0,width,height)
X,Y = np.meshgrid(x_list, t_list)
ax.set_xlabel("$\it{position}$",labelpad=10,fontsize=20)
ax.set_ylabel("time",labelpad=20,fontsize=20)
ax.set_zlabel("$|\psi|^2$",labelpad=10,fontsize=20)
ax.set_xlim(2*n,0)
ax.set_ylim(2*n,0)
ax.set_zlim(0,1)
#
mask = pp_map > 0.0
offset = pp_map[mask].ravel() + np.abs(pp_map[mask].min())
fracs = offset.astype(float)/offset.max()
norm = colors.Normalize(fracs.min(), fracs.max())
clrs = cm.cool(norm(fracs))
ax.bar3d(X[mask].ravel(), Y[mask].ravel(), pp_map[mask].ravel() ,0.5, 0.5, -pp_map[mask].ravel(),color=clrs)
ax.w_xaxis.set_pane_color((0, 0, 0, 0))
ax.w_yaxis.set_pane_color((0, 0, 0, 0))
ax.w_zaxis.set_pane_color((0, 0, 0, 1))
#grid線を消す
ax.grid(color="white")
ax.grid(False)
plt.show()

#1次元
#plt.ylim([0,0.1])
#plt.xlim([0,m])
#plt.xlabel("t",fontsize=24)
#plt.ylabel("probability",fontsize=24)
#plt.plot(t_list,np.real(p_list),color="red",label="quantum walk",linewidth=0.7)
#plt.plot(t_list,P_list,color="blue",label="random walk",linewidth=0.7)
#plt.legend(title="t=100",loc="best",fontsize=10)
#plt.show()

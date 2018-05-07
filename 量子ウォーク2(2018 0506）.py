import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

#環境設定
n=300
theta= 10*(math.pi)/12

P = [[np.cos(theta),np.sin(theta)],[0,0]]
Q = [[0,0],[np.sin(theta),-np.cos(theta)]]
x_list=[]#xline
t_list=[]#time
p_list=[]#probability
s_list=[]#state

for x in range(0,2*n+1):
    if x == n:
        phai = [1, 0]
    else:
        phai = [0,0]
    p = np.dot(phai,phai)

    x_list.append(x)
    s_list.append(phai)
    p_list.append(p)

#def p(x_list):
        #p=np.inner(x_list[x],x_list[x])
        #return p

#p_list.append(p(x_list))
    #print(x,end='')
#t =0
#t=0で原点の値(python上では5番目の配列)は[1 0]

for t in range(0,2*n):
    t_list.append(t)
    if t ==0:
        s_list
        p_list
        #x = 2*n/2
    else:
        for x in range(0,2*n+1):
            if x == 0:
                s_list[0] = np.inner(P, s_list[1])
                #p_list[0] = np.inner(x_list[0],x_list[0])
            if x == 2*n:
                s_list[2*n] = np.inner(Q, s_list[2*n-1])
                #p_list[2*n] = np.inner(x_list[2*n],x_list[2*n])
            else:
                s_list[x] = np.inner(P, s_list[x+1]) + np.inner(Q, s_list[x-1])
                p_list[x] = np.dot(s_list[x],s_list[x])
                #p_list[x]=p
                #print(t,x,p)
                #plt.plot(t,p)
                #plt.pause(0.1)

    #2Dplot
    #fig = plt.figure()
    print(x_list,p_list)
    #plt.bar(x_list,p_list,width=0.1,color='black')
    plt.xlabel("x")
    plt.ylabel("probability")
    plt.ylim([0,0.1])
    plt.xlim([-n,3*n])
    plt.plot(x_list,p_list,color="red",linewidth=0.5)
    plt.pause(0.01)
    plt.cla()

    #ims=[]
    #im = plt.plot(x_list,p_list)
    #ims.append(im)
    #ani = animation.ArtistAnimation(fig,ims, interbal=1)
    #plt.show()



    #3Dplot
    #fig=plt.figure()
    #ax =Axes3D(fig)
    #fig.add_subplot(111, projection="3d")
    #ax.bar(x_list, t_list, p_list, color="black")
    #ax.set_xlabel("x")
    #ax.set_ylabel("t")
    #ax.set_zlabel("probability")

    #plt.show()

import numpy as np
import matplotlib.pyplot as plt
import math

#set the environment
n=300
theta= (math.pi)/4  #you can change any theta you want

P = [[np.cos(theta),np.sin(theta)],[0,0]] #Shift Operator 
Q = [[0,0],[np.sin(theta),-np.cos(theta)]] #Shift Operator
x_list=[]#xline  
t_list=[]#all time
p_list=[]#all probability depend on x
s_list=[]#all state depend on x

for x in range(0,2*n+1): #you can arange the range you want
    if x == n:           #the start 
        phai = [1, 0]
    else:
        phai = [0,0]
    p = np.dot(phai,phai)

    x_list.append(x)    #[0,1,2,3,4,.......,n,n+1,.....2*n+1]
    s_list.append(phai) #[[0,0],[0,0],..........,[1,0],[0,0],....[0,0]]
    p_list.append(p)    #[0,0,0,0,0,0...........1,......0,0,0,0...]


for t in range(0,2*n): #calculate the probability depend on time.
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
    print(x_list,p_list)
    #plt.bar(x_list,p_list,width=0.1,color='black')
    plt.xlabel("x")
    plt.ylabel("probability")
    plt.ylim([0,0.1])
    plt.xlim([-n,3*n])
    plt.plot(x_list,p_list,color="red",linewidth=0.5)
    plt.pause(0.01)
    plt.cla()

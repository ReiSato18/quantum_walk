import numpy as np
import matplotlib.pyplot as plt
import math


#まだ完成していない！！！！！！！！！！！！！！

n=4
m=3
theta=3*(math.pi)/12

x_list=[]
t_list=[]
phi_list=[]
init_phi=[]
new_phi=[]
p_list=[]

for i in range(0,2*n+1):
    a=[0,0]
    new_phi.append(a)
    phi_list.append(a)
    #p_list.append(i)

def xline():
    for i in range(0,2*n+1):
        x_list.append(i)
    return x_list
#print(xline()[n])

def quantumcoin(theta):
    C = [[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]
    return C
#print(quantumcoin(3*(math.pi)/12))

#t=0の時のphi状態
def initPositionPhi():
    phi = [1,0]
    othersphi=[0,0]
    for i in range(0,2*n+1):
        if i == n:
            init_phi.append(phi)
        else:
            init_phi.append(othersphi)
    return init_phi
#print(initPositionPhi())

def CoinOperator(x,new_phi):
    new_phi[x]=np.dot(quantumcoin(theta),new_phi[x])
    return new_phi[x]
print(CoinOperator(n,initPositionPhi()))

#a=(CoinOperator(1,initPositionPhi()))[n][0]
#print(a)

#def ShiftOperator(steps,phi):
for t in range(1,m+1):
    #if t==0:
        #initPositionPhi()
    if t==1:
        new_phi[n-t][1] = (CoinOperator(n,initPositionPhi()))[1]
        new_phi[n+t][0] = (CoinOperator(n,initPositionPhi()))[0]
        print(t,new_phi)
    else:#t=2~3
        new_phi[n-t] = (CoinOperator(n-(t-1),new_phi))[1] + (CoinOperator(n-(t+1),new_phi))[0]
        new_phi[n+t] = (CoinOperator(n+(t-1),new_phi))[0] + (CoinOperator(n+(t+1),new_phi))[1]
    print(t,new_phi)


        #if x==0:
            #new_phi[x+1][0]=(CoinOperator(x,new_phi[x]))[x][0]
        #if x==2*n:
            #new_phi[x-1][1]=(CoinOperator(x,new_phi[x]))[x][1]
        #else:
            #new_phi[x+1][0]=(CoinOperator(x,new_phi[x]))[x][0]
            #new_phi[x-1][1]=(CoinOperator(x,new_phi[x]))[x][1]


            #elif x==2*n:
                #new_phi[n+t][0]=(CoinOperator(1,new_phi[n+(t-1)]))[n+(t-1)][0]
                #new_phi[n-t][1]=(CoinOperator(1,new_phi[n-(t-1)]))[n-(t-1)][1]
            #else:
    #print(t,new_phi)
    #print(t,final_phi)
    #for x in range(0,2*n+1):
        #final_phi[x]=np.dot(quantumcoin(theta),final_phi[x])
    #return new_phi
#print(final_phi)

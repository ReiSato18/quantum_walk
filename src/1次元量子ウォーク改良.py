import numpy as np
import matplotlib.pyplot as plt
import math


#まだ完成していない！！！！！！！！！！！！！！

N=4  #step数
n=4  #座標の修得
m=3  #時間
theta=3*(math.pi)/12

x_list=[]
t_list=[]
phi_list=[]
shift_phi=[]
Shiftphi=[]
init_phi=[]
p_list=[]

for i in range(0,2*N+1):
    shift_phi.append([0,0])
    phi_list.append([0,0])
    Shiftphi.append([0,0])
    p_list.append(0)

#量子コイン
def quantumcoin(theta):
    C = [[np.cos(theta),-np.sin(theta)],
        [np.sin(theta),np.cos(theta)]]
    return C


#初期のphi状態の定義
def initPositionPhi():
    for i in range(0,2*n+1):
        if i == n:
            init_phi.append([1,0])
        else:
            init_phi.append([0,0])
    return init_phi

#量子コインの動作定義
def Coin_phi(x, step, phi):
    Coinphi=[]
    for i in range(0,2*n+1):
        Coinphi.append([0,0])
    for s in range(0,step):
        Coinphi[x] = np.dot(quantumcoin(theta), phi_list[x])
    return Coinphi[x]
#print(Coin_phi(1,1,initPositionPhi()))

#Shiftoperatorの定義
def Shift_phi(x):
    for x in range(0,2*n+1):
        if x == 0:
            Shiftphi[x][1]=Coin_phi(x+1,step,shift_phi[x+1])[1]
        elif x == 2*n:
            Shiftphi[x][0]=Coin_phi(x-1,step,shift_phi[x-1])[0]
        else:
            Shiftphi[x][1]=Coin_phi(x+1,step,shift_phi[x+1])[1]
            Shiftphi[x][0]=Coin_phi(x-1,step,shift_phi[x-1])[0]
        return Shiftphi[x]

#量子ウォークのメイン動作
for step in range(0,N): #状態計算
    if step == 0:
        phi_list=initPositionPhi()
    else:
        for x in range(0,2*n+1):
            Coin_phi(x,step,phi_list)
            phi_list[x]=Shift_phi(x)
    #print(phi_list)
    print(Coin_phi())    


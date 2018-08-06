import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.quantum.grover import OracleGate
####
n = 2
itr = 2
##stage##
f1 = np.zeros([2*n+1,2*n+1])
#0~8の内のどれか
#欲しい場所に１をおく
f1[1,2] = 1
t_list=[]
p1_list=[]
p2_list=[]
######
psi = np.ones([2*n+1,2*n+1,2])
psi /= np.linalg.norm(psi)
#print(psi)
######
#np.matmul(A,B)でA,Bの行列積
#np.eye(3) 3*3の単位行列
def Uf(v):#oracle
    new = np.matmul(np.eye(2*n+1),v) -2*f1 *np.dot(f1, v)
    return new/np.linalg.norm(new)
print(Uf(psi))

def Us(v):#D
    new = 2*np.matmul(np.ones((2*n+1,2*n+1))/(2*n+1), v) - np.matmul(np.eye(2*n+1), v)
    return new / np.linalg.norm(new)
print(Us(Uf(psi)))


for t in range(itr):
    psi = Us(Uf(psi))
    #p1_list.append(psi[50,50]*psi[50,50])
    p2_list.append(psi[1,2]*psi[1,2])
    t_list.append(t)
    #print(t,psi[10]*psi[10])

#plt.plot(t_list,p1_list, color="red",label="50")
plt.plot(t_list,p2_list, color="purple",label="40")
plt.xlabel("t",fontsize="24")
plt.ylabel("probability",fontsize="24")
#plt.ylim([-0.2,1.2])
plt.tight_layout()
plt.legend(loc="best")
plt.show()

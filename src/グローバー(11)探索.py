import itertools
from sympy.physics.quantum.qubit import Qubit, measure_all
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.gate import HadamardGate, IdentityGate
from math import sqrt
import matplotlib.pyplot as plt
import functools
import operator
import random

def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)

#行列に書き直そう!
n = 10
itr = 8

f_ = [0]*n #[0,0,...0,0]の箱を用意
f_[1] = 1 #f1=1
#f_[random.randint(0, n-1)] = 1
fa = Qubit(*f_)  #|010>

basis = []
for psi_ in itertools.product([0,1], repeat=n):
    basis.append(Qubit(*psi_))
psi0 = sum(basis)/sqrt(2**n)
psi = sum(basis)/sqrt(2**n)

Hs = prod([HadamardGate(i) for i in range(n)])
Is = prod([IdentityGate(i) for i in range(n)])

'''
p_ = [0]*n
p = Qubit(*p_)
psi0 = qapply(Hs*p).doit()
psi = qapply(Hs*p)
'''

Uf = lambda q: qapply(Is*q - 2*fa*Dagger(fa)*q)     #lambda 引数:処理内容
Us = lambda q: qapply(2*psi0*Dagger(psi0)*q - Is*q)

for i in range(itr):
    psi = Us(Uf(psi))
    y = [v[1] for v in measure_all(psi)]
    x = [''.join(map(str, v[0].qubit_values)) for v in measure_all(psi)]
    plt.plot(x, y, marker='.', label=i)
    plt.xlabel("number of boxes")
    plt.ylabel("probability")
    print(y)

plt.grid()
#plt.yscale('log')
plt.legend()
plt.show()

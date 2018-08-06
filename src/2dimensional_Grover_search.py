import numpy as np
import matplotlib.pyplot as plt
import itertools

n = 10
N = 4*(n**2)
step = 100
x_list = [i for i in range(0, 2*n+1)]
y_list = [i for i in range(0, 2*n+1)]
t_list = [i for i in range(0, step+1)]
p_list = []
####################
G = np.array([[-1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1]]) / 2
P = np.zeros((4, 4))
P[0, :] = G[0, :]
Q = np.zeros((4, 4))
Q[1, :] = G[1, :]
R = np.zeros((4, 4))
R[2, :] = G[2, :]
S = np.zeros((4, 4))
S[3, :] = G[3, :]
# 33
phi_map = np.ones((2*n+1, 2*n+1, 4), dtype="complex")
#phi_map[0,0]= np.array([1,0,0,0])/(np.sqrt(4*N))
phi_map /= np.sqrt(4*N)
next_phi_map = np.zeros((2*n+1, 2*n+1, 4), dtype="complex")
p_map = np.zeros([2*n+1, 2*n+1])
p_map[0, 0] = 1.0
######################
for t in range(0, step+1):
    if t == 0:
        pass
    else:
        for i in itertools.product(x_list, y_list):
            x = i[0]
            y = i[1]

            x1 = (x-1 + 2*n+1) % (2*n+1)
            x2 = (x+1) % (2*n+1)
            y1 = (y-1 + 2*n+1) % (2*n+1)
            y2 = (y+1) % (2*n+1)
            if i == (8, 8):  # mark
                next_phi_map[i] = np.array([np.dot(-P, phi_map[x1, y]) + np.dot(-Q, phi_map[x2, y])
                                            + np.dot(-R, phi_map[x, y2]) + np.dot(-S, phi_map[x, y1])])
            else:
                next_phi_map[i] = np.array([np.dot(P, phi_map[x1, y]) + np.dot(Q, phi_map[x2, y])
                                            + np.dot(R, phi_map[x, y2]) + np.dot(S, phi_map[x, y1])])

            p_map[8, 8] = np.real(
                np.inner(next_phi_map[8, 8], np.conj(next_phi_map[8, 8])))
        phi_map = np.copy(next_phi_map)
    # print(t,p_map.sum())

    p_list.append(np.real(np.vdot(phi_map[8, 8], phi_map[8, 8])))

plt.xlabel("t", fontsize="24")
plt.ylabel("probability", fontsize="24")
plt.plot(t_list, p_list, color="red", label="quantum walk", linewidth="1")
plt.legend(title="markedvertexprobablity", loc="best", fontsize=10)
plt.tight_layout()
plt.show()

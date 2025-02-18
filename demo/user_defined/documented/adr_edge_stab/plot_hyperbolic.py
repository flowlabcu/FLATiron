import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os
import matplotlib
matplotlib.rcParams.update({'font.size': 14, 'font.weight': 'bold'})

# ------------------------------------------------------- #

from flatiron_tk.io import h5_mod
import fenics as fe

def read_data(ne, gamma):
    fname = 'output_%d_%.5f_supg/c.h5'%(ne, gamma)
    (mesh, _, _) = h5_mod.h5_read_mesh(fname)
    V = fe.FunctionSpace(mesh, 'CG', 1)
    c = h5_mod.h5_read(fname, 'c', 'function', mesh=mesh, function_space=V)
    L = 1
    ymid = L/10/2
    x = np.linspace(0, 1, ne+1)
    cmid = np.zeros(ne+1)
    for i in range(ne+1):
        cmid[i] = c(fe.Point((x[i], ymid)))
    return x, cmid

xx = np.linspace(0, 1, 1000)
H = np.heaviside(0.5-xx, 1)

# i = 0
# for gamma in [0, 1e-3, 1e-2, 1e-1]:
#     i+=1
#     plt.figure()
#     plt.plot(xx, H, 'k--', label='H(0.5-x)')
#     x, c = read_data(400, gamma)
#     if gamma == 0:
#         label = "$\gamma=0$"
#     else:
#         exponent = int(np.floor(np.log10(abs(gamma))))
#         base = gamma * 10**(-1*exponent)
#         label = "$\gamma$=%dE%d"%(base, exponent)
#     plt.plot(x, c, 'C1', label=label)
#     plt.legend(ncols=1)
#     plt.grid(True)
#     plt.xlim([0.4, 0.6])
#     plt.ylim([-0.1, 1.2])
#     plt.grid(True)
#     plt.gca().set_xticks([0.4, 0.45, 0.5, 0.55, 0.6])
#     plt.xlabel('x')
#     plt.ylabel('$c_{mid}$')
#     plt.tight_layout()
#     plt.savefig('hyperbolic_flatiron_%d.png'%i, dpi=200)


plt.plot(xx, H, 'k--', label='H(0.5-x)')
for ne in [100, 200, 400, 800]:
    mod = ne//100
    if mod == 1:
        label = "$h=10^{-2}$"
    else:
        label = "$h=\\frac{1}{%d}10^{-2}$"%mod
    x, c = read_data(ne, 0)
    plt.plot(x, c, '-', label=label)
    # x, c = read_data(ne, 1e-2)
    # plt.plot(x, c, '-', label=label)
plt.xlim([0.4, 0.6])
plt.legend(ncols=1)
plt.grid(True)
plt.xlim([0.4, 0.6])
plt.gca().set_xticks([0.4, 0.45, 0.5, 0.55, 0.6])
plt.xlabel('x')
plt.ylabel('$c_{mid}$')
plt.tight_layout()
# plt.savefig('hyperbolic_href_edge.png', dpi=200)
plt.show()


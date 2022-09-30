'''
This script is used to generate the one-dimensional image
magnitude vectors, both, theoretically and numerically.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def magnitude_theoretical(curve):
    curve = curve.flatten()
    curve_pad = np.pad(curve,1,mode='symmetric')

    boundary_x_1 = np.zeros_like(curve)
    boundary_x_1[0] = boundary_x_1[-1] = 1

    pref_x_1_plus = 1-np.exp(-np.abs(curve-curve_pad[0:-2]))
    delta_x_1_plus = pref_x_1_plus != 0

    pref_x_1_minus = 1-np.exp(-np.abs(curve-curve_pad[2:]))
    delta_x_1_minus = pref_x_1_minus != 0

    mag_vec_x_1 = 0.5*(0.001+boundary_x_1+pref_x_1_plus*delta_x_1_plus+pref_x_1_minus*delta_x_1_minus)

    return mag_vec_x_1

x = np.arange(0,1.0001,0.001).reshape(-1,1)

y_1 = np.zeros_like(x)
y_1[333:] = 3
y_1[666:] = 8

y_2 = np.zeros_like(x)
y_2[333:] = 3
y_2[666:] = 5

X = np.hstack([x,y_1,y_2])

sim_matrix = np.exp(-cdist(X,X,metric='cityblock'))
mag_vec = np.linalg.solve(sim_matrix,np.ones_like(x))

np.unique(mag_vec.round(10))

print(sorted(mag_vec,reverse=True)[:10])

mag_vec_theoretical = magnitude_theoretical(y_1)
print(sorted(mag_vec_theoretical,reverse=True)[:10])


cmap = plt.get_cmap('tab10')
norm = plt.Normalize(0,100)

fig,ax1 = plt.subplots()
ax1.plot(x,y_1,color=cmap(norm(22)),linewidth=2)
ax1.plot(x,y_2,color=cmap(norm(33)),linewidth=2)
ax1.set_ylabel('Step function',fontsize=12)
ax2 = ax1.twinx()
ax2.plot(x,mag_vec,linewidth=2,color=cmap(norm(0)),linestyle='dashed')
ax2.plot(x,mag_vec_theoretical,color=cmap(norm(11)),linestyle='dotted',linewidth=2)
ax2.set_ylabel('Magnitude measure',fontsize=12)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax1.set_xlabel('Domain',fontsize=12)
plt.tight_layout()

plt.savefig('two_steps.pdf')

plt.show()

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 15:21:01 2025

@author: Daniel
"""

import numpy as np
from matplotlib import pyplot as plt

use_gpu = False
compare = True
streamplot = False
real_pressure = True

# ======== TOGGLE BETWEEN GPU AND CPU ========
if use_gpu:
    try:
        import cupy as cp
        print("Using GPU acceleration with CuPy")
        xp = cp
    except ImportError:
        print("CuPy not found, falling back to CPU")
        use_gpu = False
        xp = np
else:
    xp = np
    print("Using CPU mode with NumPy")

#================== DATA LOADING SEQUENCE ==================
p_ = [] #nu, rho, nt, dt, dx, dy, Lx, Ly
time = []

U = []
V = []
P = []

with open('data\parameters.txt', 'r') as f:
    for line in f:
        p_.append(float(line))

with open('data\\time.txt', 'r') as f:
    for line in f:
        time.append(xp.round(float(line), 10))
    time = time[:-1]

grid_data = xp.load('data\grid.npz')

nu = p_[0]
rho = p_[1]
nt = int(p_[2])
dt = p_[3]
dx = p_[4]
dy = p_[5]
Lx = p_[6]
Ly = p_[7]

X = grid_data['arr_0']
Y = grid_data['arr_1']

for i in range(nt-1):
    data = xp.load(f'data\data{i+1}.npz')
    U.append(data['arr_0'][1:-1, 1:-1])
    V.append(data['arr_1'][1:-1, 1:-1])
    P.append(data['arr_2'][1:-1, 1:-1])

#================== FUNCTIONS ==================
def tgv(X, Y, nu, rho, t):
    u = xp.sin(X)*xp.cos(Y)*xp.exp(-2*nu*t)
    v = -xp.cos(X)*xp.sin(Y)*xp.exp(-2*nu*t)
    p = (rho/4)*(xp.cos(2*X)+xp.cos(2*Y))*xp.exp(-2*nu*t)*xp.exp(-2*nu*t)
    
    return u, v, p

a_U = []    
a_V = []
a_P = []

a_vel_mean = []
vel_mean = []
a_U_mean = []
a_V_mean = []
U_mean = []
V_mean = []

U_mae = []
V_mae = []
P_mae = []

total_tmae = xp.empty_like(P[0])
U_tmae = xp.empty_like(P[0])
V_tmae = xp.empty_like(P[0])
P_tmae = xp.empty_like(P[0])
total_tmae[:, :] = 0
U_tmae[:, :] = 0
V_tmae[:, :] = 0
P_tmae[:, :] = 0


U_relmae = []
V_relmae = []
P_relmae = []

if compare:  
    for i in range(len(time)):
        a_u, a_v, a_p = tgv(X, Y, nu, rho, time[i])
        
        U_mae.append(xp.mean(xp.abs(U[i] - a_u)))
        V_mae.append(xp.mean(xp.abs(V[i] - a_v)))
        P_mae.append(xp.mean(xp.abs(P[i] - a_p)))
        
        U_tmae += xp.abs(U[i] - a_u)
        V_tmae += xp.abs(V[i] - a_v)
        P_tmae += xp.abs(P[i] - a_p)
        total_tmae += xp.sqrt((U[i] - a_u)**2 + (V[i] - a_v)**2)
        
        U_relmae.append(xp.mean(xp.abs(U[i] - a_u))/xp.mean(xp.abs(a_u))*100)
        V_relmae.append(xp.mean(xp.abs(V[i] - a_v))/xp.mean(xp.abs(a_v))*100)
        P_relmae.append(xp.mean(xp.abs(P[i] - a_p))/xp.mean(xp.abs(a_p))*100)
        
        a_vel_mean.append(xp.mean(a_u**2 + a_v**2))
        vel_mean.append(xp.mean(U[i]**2 + V[i]**2))
        
        a_U_mean.append(xp.mean(xp.abs(a_u)))
        a_V_mean.append(xp.mean(xp.abs(a_v)))
        U_mean.append(xp.mean(xp.abs(U[i])))
        V_mean.append(xp.mean(xp.abs(V[i])))
        
        a_U.append(a_u)
        a_V.append(a_v)
        a_P.append(a_p)

def a_fix(a):
    vis_a = xp.zeros((xp.shape(a)[0]+1, xp.shape(a)[1]+1))
    vis_a[:-1, :-1] = a
    vis_a[-1, :-1] = a[0] # bottom a -> top padded_a
    vis_a[:-1, -1] = a[:, 0] # left a -> right padded_a
    vis_a[-1, -1] = a[0, 0] #corner
    
    return vis_a

#================== VISUAL PARAMETERS ==================
vmax = xp.max([xp.max(P), abs(xp.min(P))])
q_vis = 4
qscale = xp.max([np.max(U), abs(xp.min(U)), xp.max(V), abs(xp.min(V))])*len(X)/q_vis
levels = 32
cut = 0
vis_X, vis_Y = np.meshgrid(xp.linspace(0, Lx, num=int(Lx/dx)+1), xp.linspace(0, Ly, num=int(Ly/dy)+1))
#================== PLOT DATA ==================
for i in range(nt-1):
    if (i+1)%100==0 or i+1==5:
        plt.gca().set_aspect('equal', adjustable='box')
        plt.contourf(vis_X, vis_Y, a_fix(P[i]), alpha=0.6, cmap='jet',
                     vmax=vmax, vmin=-vmax, levels=levels)        
        plt.colorbar(label='Pressure')
        
        '''
        if compare:
            if not streamplot:
                plt.quiver(X[::q_vis, ::q_vis], Y[::q_vis, ::q_vis],
                           a_U[i][::q_vis, ::q_vis], a_V[i][::q_vis, ::q_vis],
                           scale=qscale, color='red')
                
            if streamplot:
                plt.streamplot(X, Y, a_U[i], a_V[i])
        '''
        
        if not streamplot:
            plt.quiver(vis_X[::q_vis, ::q_vis], vis_Y[::q_vis, ::q_vis],
                       a_fix(U[i])[::q_vis, ::q_vis], a_fix(V[i])[::q_vis, ::q_vis],
                       scale=qscale, linewidth=1)
        if streamplot:
            plt.streamplot(X, Y, U[i], V[i])
        plt.xlim(0, Lx)
        plt.ylim(0, Ly)
        plt.title(f'Plot of velocity and pressure at t={xp.round(time[i], 3)}', fontsize=11)
        plt.xlabel('$x\in[0, 2\pi$]')
        plt.ylabel('$y\in[0, 2\pi$]')
        plt.tight_layout()
        plt.show()

if compare:
    plt.title('Mean Velocity Magnitude over time')
    plt.plot(time[cut:], a_vel_mean[cut:], label='Analytical Solution', color='green')
    plt.plot(time[cut:], vel_mean[cut:], label='Simulation Data', color='red')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.legend()
    plt.show()
    plt.plot(time[cut:], U_mae, label='MAE in u', color='red')
    plt.plot(time[cut:], V_mae, label='MAE in v', color='red')
    plt.plot(time[cut:], P_mae, label='MAE in p', color='blue')
    plt.title('Mean Absolute Error over time in pressure and velocity')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.legend()
    plt.show()
    plt.plot(time[cut:], U_relmae, label='%MAE in u', color='orange')
    plt.plot(time[cut:], V_relmae, label='%MAE in v', color='orange')
    plt.plot(time[cut:], P_relmae, label='%MAE in p', color='purple')
    plt.title('Percentage Mean Absolute Error over time in pressure and velocity')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.legend()
    plt.show()
    plt.contourf(vis_X, vis_Y, a_fix(total_tmae), alpha=0.6, cmap='Reds',
                 vmax=float(xp.max(total_tmae)), vmin=0, levels=levels)
    plt.colorbar(label='Error')
    plt.title('Sum of error in velocity magnitude')
    plt.xlabel('$x\in[0, 2\pi$]')
    plt.ylabel('$y\in[0, 2\pi$]')
    plt.show()
    plt.contourf(vis_X, vis_Y, a_fix(U_tmae), alpha=0.6, cmap='Reds',
                 vmax=float(xp.max(U_tmae)), vmin=0, levels=levels)
    plt.colorbar(label='Error')
    plt.title('Sum of error in u')
    plt.xlabel('$x\in[0, 2\pi$]')
    plt.ylabel('$y\in[0, 2\pi$]')
    plt.show()
    plt.contourf(vis_X, vis_Y, a_fix(V_tmae), alpha=0.6, cmap='Reds',
                 vmax=float(xp.max(V_tmae)), vmin=0, levels=levels)
    plt.colorbar(label='Error')
    plt.title('Sum of error in v')
    plt.xlabel('$x\in[0, 2\pi$]')
    plt.ylabel('$y\in[0, 2\pi$]')
    plt.show()
    plt.contourf(vis_X, vis_Y, a_fix(P_tmae), alpha=0.6, cmap='Reds',
                 vmax=float(xp.max(P_tmae)), vmin=0, levels=levels)
    plt.colorbar(label='Error')
    plt.title('Sum of error in p')
    plt.xlabel('$x\in[0, 2\pi$]')
    plt.ylabel('$y\in[0, 2\pi$]')
    plt.show()
    plt.close()
    
plt.close()
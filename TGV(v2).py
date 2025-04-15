import numpy as np
import matplotlib.pyplot as plt

use_gpu = True # Set to True for GPU acceleration
debug = False # Set to True to turn off main loop and run debug section

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
    
# ================== PARAMETERS ==================
Lx = 2*xp.pi # Physical domain size in x
Ly = Lx # Physical domain size in y
dx = 0.1*xp.pi # Spatial resolution in x
dy = dx # Spatial reolution in Y

Lx -= dx # Removes extra index for symmetric IC
Ly -= dy

xlength = int(Lx/dx) + 1 # Number of grid points in x
ylength = int(Ly/dy) + 1 # Add 1 to account for the 0 index

nt = 1200 # Time steps
dt = 0.05 # Time step size
CFL_limit = 0.5
nit = 500 # Pressure solver iterations

# Physical properties
rho = 0.01 # Density
nu = 0.4 # Kinematic viscosity

q_vis = 1 # Visualisation downsampling
vmax = None # Colourmap range
vmin = None
levels = 24 # Colourmap levels

x = xp.linspace(0, Lx, num=xlength) # Create grid
y = xp.linspace(0, Ly, num=ylength)
X, Y = xp.meshgrid(x, y)

# ================== HELPER FUNCTIONS ==================
def get_array(a):
    """Convert to numpy array if using GPU"""
    return cp.asnumpy(a) if use_gpu else a

def pad_array(a):
    """Applies periodic BC ghost cells to a 2D array"""
    padded_a = xp.zeros((xlength+2, ylength+2))
    
    padded_a[1:-1, 1:-1] = a
    padded_a[0, 1:-1] = a[-1] # top a -> bottom padded_a
    padded_a[-1, 1:-1] = a[0] # bottom a -> top padded_a
    padded_a[1:-1, 0] = a[:, -1] # right a -> left padded_a
    padded_a[1:-1, -1] = a[:, 0] # left a -> right padded_a
    
    return padded_a

def build_up_b(b, u, v, dt, dx, dy, rho):
    """Build RHS of Poisson equation"""
    
    dudx = (u[1:-1, 2:] - u[1:-1, :-2])/(2*dx)
    dudy = (u[2:, 1:-1] - u[:-2, 1:-1])/(2*dy)
    dvdx = (v[1:-1, 2:] - v[1:-1, :-2])/(2*dx)
    dvdy = (v[2:, 1:-1] - v[:-2, 1:-1])/(2*dy)
    
    b[1:-1, 1:-1] = rho*(1/dt*(dudx + dvdy) - dudx**2 - 2*(dudy*dvdx) - dvdy**2)
    b = pad_array(b[1:-1, 1:-1])
    
    return b

def pressure_poisson(p, b, dx, dy, nit):
    """Solve pressure Poisson equation in nit iterations"""
    
    for _ in range(nit):
        pn = p.copy()
        
        p[1:-1, 1:-1] = (pn[1:-1, 2:] - pn[1:-1, :-2])*(dy**2)+(pn[2:, 1:-1] - pn[:-2, 1:-1])*(dx**2)
        p[1:-1, 1:-1] -= b[1:-1, 1:-1]*(dx**2)*(dy**2)
        p[1:-1, 1:-1] /= 2*(dx**2 + dy**2)
        p = pad_array(p[1:-1, 1:-1])
    
    return p


if not debug:
    # ================== MAIN INITIALIZATION ==================
    u = xp.zeros((xlength, ylength))
    v = xp.zeros((xlength, ylength))
    u = xp.sin(X)*xp.cos(Y)
    v = -xp.cos(X)*xp.sin(Y)
    
    u = pad_array(u)
    v = pad_array(v)
    
    un = xp.empty_like(u)
    vn = xp.empty_like(v)
    
    b = xp.zeros((xlength, ylength))
    b = pad_array(b)
    
    p = xp.zeros((xlength, ylength))
    p = pad_array(p)
    
    # ================== MAIN SIMULATION LOOP ==================
    stepcount = 0
    
    for i in range(nt):  
        
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, u, v, dt, dx, dy, rho)
        p = pressure_poisson(p, b, dx, dy, nit)

        u_conv = (un[1:-1, 1:-1]*(dt/(2*dx))*(un[1:-1, 2:] - un[1:-1, :-2]) + 
                  vn[1:-1, 1:-1]*(dt/(2*dy))*(un[2:, 1:-1] - un[:-2, 1:-1]))
        
        u_diff = (nu*((dt/dx**2)*(un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2]) + 
                      (dt/dy**2)*(un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1])))
        
        u_pressure = dt/(2*rho*dx)*(p[1:-1, 2:] - p[1:-1, :-2])
        
        u[1:-1, 1:-1] = un[1:-1, 1:-1] - u_conv - u_pressure + u_diff
    
        v_conv = (un[1:-1, 1:-1]*(dt/(2*dx))*(vn[1:-1, 2:] - vn[1:-1, :-2]) + 
                  vn[1:-1, 1:-1]*(dt/(2*dy))*(vn[2:, 1:-1] - vn[:-2, 1:-1]))
        
        v_diff = (nu*((dt/dx**2)*(vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2]) + 
                      (dt/dy**2)*(vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1])))
        
        v_pressure = dt/(2*rho*dy)*(p[2:, 1:-1] - p[:-2, 1:-1])
        
        v[1:-1, 1:-1] = vn[1:-1, 1:-1] - v_conv - v_pressure + v_diff
        
        u = pad_array(u[1:-1, 1:-1])
        v = pad_array(v[1:-1, 1:-1])
        
        stepcount += 1

        plt.contourf(get_array(X), get_array(Y), get_array(p[1:-1, 1:-1]), alpha=0.6, cmap='jet',
                     vmax=vmax, vmin=vmin, levels=levels)
        plt.colorbar()
        plt.title('t='+str(xp.round(stepcount*dt, 3)))
        plt.quiver(get_array(X)[::q_vis, ::q_vis], get_array(Y)[::q_vis, ::q_vis],
                   get_array(u[1:-1, 1:-1])[::q_vis, ::q_vis], get_array(v[1:-1, 1:-1])[::q_vis, ::q_vis])
        plt.show()

# ================== DEBUG FUNCTIONS ==================
def check_pad(a):
    if a[1:-1, 1].all()!=a[1:-1, -1].all():
        print("False!")
    elif a[1:-1, -2].all()!=a[1:-1, 0].all():
        print("False!")
    elif a[1, 1:-1].all()!=a[-1, 1:-1].all():
        print("False!")
    elif a[-2, 1:-1].all()!=a[0, 1:-1].all():
        print("False!")
    else:
        print("True!")
    
    return 0

def divergence(u, v):
    div_u = (u[1:-1, 2:] - u[1:-1, :-2])/dx + (v[2:, 1:-1] - v[:-2, 1:-1])/dy
    return div_u

# ================== TESTING ==================
if debug:
    pass
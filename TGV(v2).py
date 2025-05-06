import numpy as np
import matplotlib.pyplot as plt

use_gpu = True # Set to True for GPU acceleration

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
dx = (2/64)*xp.pi # Spatial resolution in x
dy = dx # Spatial reolution in Y

u_noise = 0.1
v_noise = u_noise

xlength = int(Lx/dx) # Number of grid points in x
ylength = int(Ly/dy) # Add 1 to account for the 0 index


nt = 400+1 # Time steps
dt = 0.0025 # Time step size
CFL_limit = 0.24 #nu = 0.4
nit = 100  # Pressure solver iterations

# Physical properties
rho = 0.01 # Density
nu = 0.4 # Kinematic viscosity

q_vis = 4 # Visualisation downsampling
vmax = None # Colourmap range
vmin = None
levels = 32 # Colourmap levels

x = xp.linspace(0, Lx-dx, num=xlength) # Create grid
y = xp.linspace(0, Ly-dy, num=ylength)
X, Y = xp.meshgrid(x, y)

#awkward fix
vis_X, vis_Y = np.meshgrid(xp.linspace(0, Lx, num=xlength+1), xp.linspace(0, Ly, num=ylength+1))

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
    
    padded_a[-1, -1] = a[0, 0]
    padded_a[0, 0] = a[-1, -1]
    padded_a[-1, 0] = a[0, -1]
    padded_a[0, -1] = a[-1, 0]
    
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

def a_fix(a):
    vis_a = xp.zeros((xlength+1, ylength+1))
    vis_a[:-1, :-1] = a
    vis_a[-1, :-1] = a[0] # bottom a -> top padded_a
    vis_a[:-1, -1] = a[:, 0] # left a -> right padded_a
    vis_a[-1, -1] = a[0, 0] #corner
    
    return vis_a


if __name__=="__main__":
    # ================== MAIN INIT. ==================
    u = xp.zeros((xlength, ylength))
    v = xp.zeros((xlength, ylength))
    u = xp.sin(X)*xp.cos(Y) 
    u += (u_noise*xp.random.randn(xlength, ylength))
    v = -xp.cos(X)*xp.sin(Y) 
    v += (v_noise*xp.random.randn(xlength, ylength))
    
    u = pad_array(u)
    v = pad_array(v)
    
    un = xp.empty_like(u)
    vn = xp.empty_like(v)
    
    b = xp.zeros((xlength, ylength))
    b = pad_array(b)
    
    p = xp.zeros((xlength, ylength))
    p = pad_array(p)
    
    data = []
    time = [0]
            
    xp.savez('data\grid.npz', X, Y)
    
    plt.contourf(get_array(vis_X), get_array(vis_Y), get_array(a_fix(p[1:-1, 1:-1])), alpha=0.6, cmap='jet',
                 vmax=vmax, vmin=vmin, levels=levels)
    plt.colorbar()
    plt.title(str(xlength)+'x'+str(ylength)+' mesh, t=0, dt='+str(dt))
    plt.xlim(0, Lx)
    plt.ylim(0, Ly)
    plt.quiver(get_array(vis_X)[::q_vis, ::q_vis], get_array(vis_Y)[::q_vis, ::q_vis],
               get_array(a_fix(u[1:-1, 1:-1]))[::q_vis, ::q_vis], get_array(a_fix(v[1:-1, 1:-1]))[::q_vis, ::q_vis])
    plt.show()
    
    # ================== MAIN SIMULATION LOOP ==================   
    for i in range(nt):  
        
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, u, v, dt, dx, dy, rho)
        p = pressure_poisson(p, b, dx, dy, nit)

    
        # Calculate adaptive time step based on CFL
        #CFL = xp.max(xp.abs(u)) * dt / dx + xp.max(xp.abs(v)) * dt / dy
        #print(CFL)
        #if CFL > CFL_limit:
            #new_dt = CFL_limit * dt / CFL
            #print(f"Adjusting dt from {dt:.6e} to {new_dt:.6e} due to CFL violation")
            #dt = new_dt

        # ================== CALCULATION ================== 
        u_conv = (un[1:-1, 1:-1]*(dt/(2*dx))*(un[1:-1, 2:] - un[1:-1, :-2]) + 
                  vn[1:-1, 1:-1]*(dt/(2*dy))*(un[2:, 1:-1] - un[:-2, 1:-1]))
        
        #u_conv = (un[1:-1, 1:-1]*(dt/(2*dx))*(un[1:-1, 2:] - un[1:-1, 1:-1]) + 
                  #vn[1:-1, 1:-1]*(dt/(2*dy))*(un[2:, 1:-1] - un[1:-1, 1:-1]))
        
        u_diff = (nu*((dt/dx**2)*(un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2]) + 
                      (dt/dy**2)*(un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1])))
        
        u_pressure = dt/(2*rho*dx)*(p[1:-1, 2:] - p[1:-1, :-2])
        
        u[1:-1, 1:-1] = un[1:-1, 1:-1] - u_conv - u_pressure + u_diff
    
        v_conv = (un[1:-1, 1:-1]*(dt/(2*dx))*(vn[1:-1, 2:] - vn[1:-1, :-2]) + 
                  vn[1:-1, 1:-1]*(dt/(2*dy))*(vn[2:, 1:-1] - vn[:-2, 1:-1]))
        
        #v_conv = (un[1:-1, 1:-1]*(dt/(2*dx))*(vn[1:-1, 2:] - vn[1:-1, 1:-1]) + 
                  #vn[1:-1, 1:-1]*(dt/(2*dy))*(vn[2:, 1:-1] - vn[1:-1, 1:-1]))
        
        v_diff = (nu*((dt/dx**2)*(vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2]) + 
                      (dt/dy**2)*(vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1])))
        
        v_pressure = dt/(2*rho*dy)*(p[2:, 1:-1] - p[:-2, 1:-1])
        
        v[1:-1, 1:-1] = vn[1:-1, 1:-1] - v_conv - v_pressure + v_diff
        
        u = pad_array(u[1:-1, 1:-1])
        v = pad_array(v[1:-1, 1:-1]) 
        
        # ================== DATA MANAGEMENT AND VISUALISATION ==================
        
        time.append((time[-1] + dt))
        data.append([u, v, p])
        
        if (i+1)%10==0 or i+1==5:
            plt.contourf(get_array(vis_X), get_array(vis_Y), get_array(a_fix(p[1:-1, 1:-1])), alpha=0.6, cmap='jet',
                         vmax=vmax, vmin=vmin, levels=levels)
            plt.colorbar()
            plt.title(str(xlength)+'x'+str(ylength)+' mesh, t='+str(xp.round((i+1)*dt, 3))+', dt='+str(dt))
            plt.xlim(0, Lx)
            plt.ylim(0, Ly)
            plt.quiver(get_array(vis_X)[::q_vis, ::q_vis], get_array(vis_Y)[::q_vis, ::q_vis],
                       get_array(a_fix(u[1:-1, 1:-1]))[::q_vis, ::q_vis], get_array(a_fix(v[1:-1, 1:-1]))[::q_vis, ::q_vis])
            plt.show()

    for i in range(nt):
        xp.savez('data\data'+str(i+1)+'.npz', data[i][0], data[i][1], data[i][2])

    with open('data\parameters.txt', 'w') as f:
        for i in [nu, rho, nt, dt, dx, dy, Lx, Ly]:
            f.write(f'{i}\n')
        f.close()
            
    with open('data\\time.txt', 'w') as f:
        for i in time[1:]:
            f.write(f'{i}\n')
        f.close()
    
    plt.close()
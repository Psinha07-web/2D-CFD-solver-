"""
2D Incompressible Navier-Stokes Solver
Projection method (Chorin's) on a staggered MAC grid.
Outputs: velocity (u, v), pressure (p), vorticity (w).
"""

import numpy as np
from scipy.linalg import solve
import h5py
import os


class CFDSolver2D:
    """
    Solves 2D incompressible Navier-Stokes on a uniform Cartesian grid.

    Grid layout (staggered MAC):
      - u[i,j] : x-velocity at (i+0.5, j)   shape (Nx+1, Ny)
      - v[i,j] : y-velocity at (i, j+0.5)   shape (Nx, Ny+1)
      - p[i,j] : pressure   at (i, j)        shape (Nx, Ny)
    """

    def __init__(self, Nx=64, Ny=64, Lx=1.0, Ly=1.0, Re=100.0, dt=0.001):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.Re = Re
        self.dt = dt
        self.nu = 1.0 / Re  # kinematic viscosity

        self.dx = Lx / Nx
        self.dy = Ly / Ny

        # Staggered grid fields
        self.u = np.zeros((Nx + 1, Ny))      # x-velocity
        self.v = np.zeros((Nx, Ny + 1))      # y-velocity
        self.p = np.zeros((Nx, Ny))          # pressure

       
        self._build_poisson_matrix()


    # Poisson matrix (Laplacian with Neumann BCs on all walls)

    def _build_poisson_matrix(self):
        N = self.Nx * self.Ny
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        A = np.zeros((N, N))

        def idx(i, j):
            return i * self.Ny + j

        for i in range(self.Nx):
            for j in range(self.Ny):
                k = idx(i, j)
                # Center
                coeff = 0.0
                # East
                if i + 1 < self.Nx:
                    A[k, idx(i + 1, j)] = 1.0 / dx2
                    coeff -= 1.0 / dx2
                else:
                    coeff -= 1.0 / dx2  # Neumann: ghost = interior
                # West
                if i - 1 >= 0:
                    A[k, idx(i - 1, j)] = 1.0 / dx2
                    coeff -= 1.0 / dx2
                else:
                    coeff -= 1.0 / dx2
                # North
                if j + 1 < self.Ny:
                    A[k, idx(i, j + 1)] = 1.0 / dy2
                    coeff -= 1.0 / dy2
                else:
                    coeff -= 1.0 / dy2
                # South
                if j - 1 >= 0:
                    A[k, idx(i, j - 1)] = 1.0 / dy2
                    coeff -= 1.0 / dy2
                else:
                    coeff -= 1.0 / dy2
                A[k, k] = coeff

        # Fix pressure gauge: pin p[0,0] = 0
        A[0, :] = 0.0
        A[0, 0] = 1.0
        self.A_poisson = A

    # Boundary conditions (lid-driven cavity default)
 
    def apply_boundary_conditions(self, u_lid=1.0):
        """Lid-driven cavity: top wall moves at u_lid, all others no-slip."""
        Nx, Ny = self.Nx, self.Ny
        self.u[0, :] = 0.0
        self.u[Nx, :] = 0.0
        self.v[:, 0] = 0.0
        self.v[:, Ny] = 0.0
        self.u[:, Ny - 1] = 2 * u_lid - self.u[:, Ny - 2] 

    def _interp_u(self, x, y):
        """Bilinear interpolation of u at arbitrary (x, y)."""
        i = np.clip(x / self.dx - 0.5, 0, self.Nx - 1)
        j = np.clip(y / self.dy, 0, self.Ny - 1)
        i0, j0 = int(i), int(j)
        i1 = min(i0 + 1, self.Nx)
        j1 = min(j0 + 1, self.Ny - 1)
        fi, fj = i - i0, j - j0
        return ((1 - fi) * (1 - fj) * self.u[i0, j0] +
                fi * (1 - fj) * self.u[i1, j0] +
                (1 - fi) * fj * self.u[i0, j1] +
                fi * fj * self.u[i1, j1])

    def _interp_v(self, x, y):
        """Bilinear interpolation of v at arbitrary (x, y)."""
        i = np.clip(x / self.dx, 0, self.Nx - 1)
        j = np.clip(y / self.dy - 0.5, 0, self.Ny - 1)
        i0, j0 = int(i), int(j)
        i1 = min(i0 + 1, self.Nx - 1)
        j1 = min(j0 + 1, self.Ny)
        fi, fj = i - i0, j - j0
        return ((1 - fi) * (1 - fj) * self.v[i0, j0] +
                fi * (1 - fj) * self.v[i1, j0] +
                (1 - fi) * fj * self.v[i0, j1] +
                fi * fj * self.v[i1, j1])

    def advect(self):
        """Semi-Lagrangian advection for u and v."""
        Nx, Ny, dx, dy, dt = self.Nx, self.Ny, self.dx, self.dy, self.dt
        u_new = np.copy(self.u)
        v_new = np.copy(self.v)

        # Advect u
        for i in range(1, Nx):
            for j in range(Ny):
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                uu = self.u[i, j]
                vv = 0.25 * (self.v[i - 1, j] + self.v[i, j] +
                             self.v[i - 1, j + 1] + self.v[i, j + 1])
                xp = np.clip(x - dt * uu, 0.5 * dx, Nx * dx - 0.5 * dx)
                yp = np.clip(y - dt * vv, 0.5 * dy, Ny * dy - 0.5 * dy)
                u_new[i, j] = self._interp_u(xp, yp)

        # Advect v
        for i in range(Nx):
            for j in range(1, Ny):
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                uu = 0.25 * (self.u[i, j - 1] + self.u[i + 1, j - 1] +
                             self.u[i, j] + self.u[i + 1, j])
                vv = self.v[i, j]
                xp = np.clip(x - dt * uu, 0.5 * dx, Nx * dx - 0.5 * dx)
                yp = np.clip(y - dt * vv, 0.5 * dy, Ny * dy - 0.5 * dy)
                v_new[i, j] = self._interp_v(xp, yp)

        self.u = u_new
        self.v = v_new

    # Diffusion (explicit central differences

    def diffuse(self):
        nu, dt, dx, dy = self.nu, self.dt, self.dx, self.dy
        Nx, Ny = self.Nx, self.Ny

        u_new = np.copy(self.u)
        v_new = np.copy(self.v)

        # Laplacian of u (interior)
        u_new[1:Nx, 1:Ny - 1] += nu * dt * (
            (self.u[2:Nx + 1, 1:Ny - 1] - 2 * self.u[1:Nx, 1:Ny - 1] + self.u[0:Nx - 1, 1:Ny - 1]) / dx ** 2 +
            (self.u[1:Nx, 2:Ny] - 2 * self.u[1:Nx, 1:Ny - 1] + self.u[1:Nx, 0:Ny - 2]) / dy ** 2
        )

        # Laplacian of v (interior)
        v_new[1:Nx - 1, 1:Ny] += nu * dt * (
            (self.v[2:Nx, 1:Ny] - 2 * self.v[1:Nx - 1, 1:Ny] + self.v[0:Nx - 2, 1:Ny]) / dx ** 2 +
            (self.v[1:Nx - 1, 2:Ny + 1] - 2 * self.v[1:Nx - 1, 1:Ny] + self.v[1:Nx - 1, 0:Ny - 1]) / dy ** 2
        )

        self.u = u_new
        self.v = v_new

    # Pressure projection
    def pressure_project(self):
        Nx, Ny, dx, dy, dt = self.Nx, self.Ny, self.dx, self.dy, self.dt

        # Divergence of u*
        div = np.zeros((Nx, Ny))
        div += (self.u[1:Nx + 1, :] - self.u[0:Nx, :]) / dx
        div += (self.v[:, 1:Ny + 1] - self.v[:, 0:Ny]) / dy

        rhs = div.flatten() / dt
        rhs[0] = 0.0  # pressure gauge fix

        p_flat = np.linalg.solve(self.A_poisson, rhs)
        self.p = p_flat.reshape(Nx, Ny)

        # Correct velocity
        self.u[1:Nx, :] -= dt * (self.p[1:Nx, :] - self.p[0:Nx - 1, :]) / dx
        self.v[:, 1:Ny] -= dt * (self.p[:, 1:Ny] - self.p[:, 0:Ny - 1]) / dy 
        
    # Derived quantities

    def vorticity(self):
        """Vorticity w = dv/dx - du/dy at cell centers."""
        Nx, Ny = self.Nx, self.Ny
        dx, dy = self.dx, self.dy
        # Interpolate to cell centers
        dvdx = (self.v[1:Nx, 1:Ny] - self.v[0:Nx - 1, 1:Ny]) / dx
        dudy = (self.u[1:Nx, 1:Ny] - self.u[1:Nx, 0:Ny - 1]) / dy
        w = np.zeros((Nx, Ny))
        w[1:Nx, 1:Ny] = dvdx - dudy
        return w

    def velocity_magnitude(self):
        """Speed at cell centers via interpolation."""
        Nx, Ny = self.Nx, self.Ny
        uc = 0.5 * (self.u[0:Nx, :] + self.u[1:Nx + 1, :])
        vc = 0.5 * (self.v[:, 0:Ny] + self.v[:, 1:Ny + 1])
        return np.sqrt(uc ** 2 + vc ** 2)

    def cell_center_velocity(self):
        Nx, Ny = self.Nx, self.Ny
        uc = 0.5 * (self.u[0:Nx, :] + self.u[1:Nx + 1, :])
        vc = 0.5 * (self.v[:, 0:Ny] + self.v[:, 1:Ny + 1])
        return uc, vc

    # Single time step

    def step(self, u_lid=1.0):
        self.apply_boundary_conditions(u_lid)
        self.advect()
        self.diffuse()
        self.apply_boundary_conditions(u_lid)
        self.pressure_project()
        self.apply_boundary_conditions(u_lid)

    # Run simulation and collect snapshots
  

    def run(self, n_steps=500, snapshot_every=10, u_lid=1.0):
        """
        Run the solver for n_steps and return snapshots.

        Returns dict with arrays of shape (n_snapshots, Nx, Ny).
        """
        snapshots = {"u": [], "v": [], "p": [], "vorticity": [], "speed": []}

        for step_i in range(n_steps):
            self.step(u_lid)
            if step_i % snapshot_every == 0:
                uc, vc = self.cell_center_velocity()
                snapshots["u"].append(uc.copy())
                snapshots["v"].append(vc.copy())
                snapshots["p"].append(self.p.copy())
                snapshots["vorticity"].append(self.vorticity().copy())
                snapshots["speed"].append(self.velocity_magnitude().copy())
                print(f"  Step {step_i:4d}/{n_steps}  |  "
                      f"max|u|={np.max(np.abs(self.u)):.4f}  "
                      f"max|p|={np.max(np.abs(self.p)):.4f}")

        for k in snapshots:
            snapshots[k] = np.array(snapshots[k])  # (T, Nx, Ny)
        return snapshots

# Save / load helpers

def save_snapshots_hdf5(snapshots, path, metadata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in snapshots.items():
            f.create_dataset(k, data=v, compression="gzip")
        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v
    print(f"Saved snapshots → {path}")


def load_snapshots_hdf5(path):
    with h5py.File(path, "r") as f:
        data = {k: f[k][:] for k in f.keys()}
        meta = dict(f.attrs)
    return data, meta


def save_snapshots_npy(snapshots, directory):
    os.makedirs(directory, exist_ok=True)
    for k, v in snapshots.items():
        np.save(os.path.join(directory, f"{k}.npy"), v)
    print(f"Saved snapshots → {directory}")

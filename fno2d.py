"""
Fourier Neural Operator (FNO) for 2D fluid flow.

Architecture:
  Input:  (batch, in_channels, Nx, Ny)   — stacked [u, v, p] at time t
  Output: (batch, out_channels, Nx, Ny)  — predicted [u, v, p] at time t+1

Reference: Li et al. "Fourier Neural Operator for Parametric PDEs" (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Spectral convolution layer

class SpectralConv2d(nn.Module):
    """
    2D Fourier layer.
    Performs convolution in Fourier space using the first
    `modes1` x `modes2` Fourier modes (low-frequency).
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  
        self.modes2 = modes2  

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def _complex_mul2d(self, x, weights):
        """Batched complex matrix-vector multiplication in Fourier space."""
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x):
        B, _, Nx, Ny = x.shape

        # FFT
        x_ft = torch.fft.rfft2(x)

       
        out_ft = torch.zeros(B, self.out_channels, Nx, Ny // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self._complex_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self._complex_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Inverse FFT
        return torch.fft.irfft2(out_ft, s=(Nx, Ny))

class FNOBlock2d(nn.Module):
    """
    One FNO layer:  x → SpectralConv(x) + W·x  → activation
    The bypass W·x is a standard 1x1 conv (pointwise linear).
    """

    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))

class FNO2D(nn.Module):
    """
    Full FNO for 2D Navier-Stokes prediction.

    Args:
        in_channels  : number of input fields (e.g. 3 for u, v, p)
        out_channels : number of output fields
        modes1       : Fourier modes in x
        modes2       : Fourier modes in y
        width        : latent channel width
        n_layers     : number of FNO blocks
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 modes1=12,
                 modes2=12,
                 width=32,
                 n_layers=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width

      
        self.lift = nn.Conv2d(in_channels + 2, width, kernel_size=1) 
        self.blocks = nn.ModuleList([
            FNOBlock2d(width, modes1, modes2) for _ in range(n_layers)
        ])

        self.proj = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, kernel_size=1)
        )

        self._grid_cache = {}

    def _get_grid(self, Nx, Ny, device):
        key = (Nx, Ny, device)
        if key not in self._grid_cache:
            x = torch.linspace(0, 1, Nx, device=device)
            y = torch.linspace(0, 1, Ny, device=device)
            gx, gy = torch.meshgrid(x, y, indexing="ij")
            self._grid_cache[key] = torch.stack([gx, gy], dim=0)  # (2, Nx, Ny)
        return self._grid_cache[key]

    def forward(self, x):
        """
        x: (batch, in_channels, Nx, Ny)
        returns: (batch, out_channels, Nx, Ny)
        """
        B, C, Nx, Ny = x.shape
        grid = self._get_grid(Nx, Ny, x.device).unsqueeze(0).expand(B, -1, -1, -1)

        x = torch.cat([x, grid], dim=1)  # (B, C+2, Nx, Ny)
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)

        # Project
        return self.proj(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class FNO2DRollout(FNO2D):
    """
    FNO2D that can auto-regressively roll out T future timesteps.
    """

    def rollout(self, x0, T):
        """
        x0: (batch, in_channels, Nx, Ny) — initial condition
        T:  number of future steps to predict
        Returns: (batch, T, in_channels, Nx, Ny)
        """
        preds = []
        x = x0
        for _ in range(T):
            x = self.forward(x)
            preds.append(x.unsqueeze(1))
        return torch.cat(preds, dim=1)

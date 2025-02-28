import numpy as np

def concentration(n, r0, s):
    '''Calculate the concentration of particles in the confocal volume.'''
    return n / (np.pi**1.5 * s * r0**3)

def diffusion_coefficient(tauD, r0):
    return r0**2 / (4 * tauD)

def twod(tau, tauD):
    """Unnormalized 2D diffusion model"""
    return 1/(1 + tau/tauD)

def threed(tau, tauD, s):
    """Unnormalized 3D diffusion model"""
    return 1/((1 + tau/tauD) * np.sqrt(1 + tau/(s**2 * tauD)))

def threedd(tau, D, r0, s):
    """Unnormalized 3D diffusion model in terms of the diffusion coefficient"""
    tauD = r0**2 / (4 * D)
    return 1/((1 + tau/tauD) * np.sqrt(1 + tau/(s**2 * tauD)))

def trip(tau, tautr, T):
    '''Unnormalized triplet state model'''
    if tautr == 0 or T == 0:
        UTC = 0
    else:
        UTC = 1 + T/(1-T) * np.exp(-tau / tautr)
    return UTC

def CF_2d_gauss(taus, n, tauD, offset):
    """2D diffusion model with a gaussian confocal volume."""
    UDC = twod(taus, tauD)
    G = offset + UDC / n
    return G

def CF_3d_gauss(taus, n, tauD, s, offset):
    """3D diffusion model with a gaussian confocal volume."""
    UDC = threed(taus, tauD, s)
    G = offset + UDC / n
    return G

def CF_3d_gauss_fixD(taus, n, D, r0, s, offset):
    """3D diffusion model with a Gaussian confocal volume.
       Addition r0 parameter allows for fixing of the diffusion coefficient."""
    UDC = threedd(taus, D, r0, s)
    G = offset + UDC / n
    return G

def CF_3d_gauss_1T(taus, n, tauD, s, offset, tautr1, T1):
    """3D diffusion model with a gaussian confocal volume and a single triplet state."""
    UDC = threed(taus, tauD, s)
    UTC = trip(taus, tautr1, T1)
    G = offset + UTC * UDC / n
    return G

def CF_3d_gauss_1T_fixD(taus, n, D, r0, s, offset, tautr1, T1):
    """3D diffusion model with a Gaussian confocal volume.
       Addition r0 parameter allows for fixing of the diffusion coefficient."""
    UDC = threedd(taus, D, r0, s)
    UTC = trip(taus, tautr1, T1)
    G = offset + UTC * UDC / n
    return G

def CF_3d_gauss_2T(taus, n, tauD, s, offset, tautr1, T1, tautr2, T2):
    """3D diffusion model with a gaussian confocal volume and a single triplet state."""
    UDC = threed(taus, tauD, s)
    UTC1 = trip(taus, tautr1, T1)
    UTC2 = trip(taus, tautr2, T2)
    G = offset + UTC1 * UTC2 * UDC / n
    return G

def CF_3d_gauss_2T(taus, n, D, r0, s, offset, tautr1, T1, tautr2, T2):
    """3D diffusion model with a gaussian confocal volume and a single triplet state."""
    UDC = threedd(taus, D, r0, s)
    UTC1 = trip(taus, tautr1, T1)
    UTC2 = trip(taus, tautr2, T2)
    G = offset + UTC1 * UTC2 * UDC / n
    return G

def weight_function(errors, n = 1):
    """Weight function for fitting."""
    return 1/errors**n
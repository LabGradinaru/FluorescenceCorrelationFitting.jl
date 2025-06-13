import numpy as np

def concentration(n, r0, s):
    """Calculate the concentration of particles in the confocal volume."""
    return n / (np.pi**1.5 * s * r0**3)

def diffusion_coefficient(tauD, r0):
    return r0**2 / (4 * tauD)

def G0(n,T):
    return (1-T)/n

def twod(tau, tauD):
    """Unnormalized 2D diffusion model"""
    return 1/(1 + tau/tauD)

def twoda(tau, tauD,a):
    """Unnormalized 2D diffusion model with anomalous diffusion"""
    return 1/(1 + (tau/tauD)**a)

def threed(tau, tauD, s):
    """Unnormalized 3D diffusion model"""
    return 1/((1 + tau/tauD) * np.sqrt(1 + tau/(s**2 * tauD)))

def threedd(tau, D, r0, s):
    """Unnormalized 3D diffusion model in terms of the diffusion coefficient"""
    tauD = r0**2 / (4 * D)
    return 1/((1 + tau/tauD) * np.sqrt(1 + tau/(s**2 * tauD)))

def trip(tau, tautr, T):
    """Unnormalized triplet state model."""
    if tautr == 0 or T == 0:
        UTC = 0
    else:
        UTC = 1 + T/(1-T) * np.exp(-tau / tautr)
    return UTC

def conf(tau, amps, tcs):
    """Unnormalized contribution due to conformational dynamics.
       Krichevsky O, Bonnet G. (2002). Fluorescence correlation spectroscopy: the technique and its applications"""
    if np.sum(amps) == 0 or np.sum(tcs) == 0:
        UCC = 0
    else:
        UCC = 1 + np.sum(amps * np.exp(-tau/tcs))
    return UCC

def CF_2d_gauss(taus, n, tauD, offset):
    """2D diffusion model with a gaussian confocal volume."""
    UDC = twod(taus, tauD)
    G = offset + UDC / n
    return G

def CF_2d_2c(taus, n, tauD1, tauD2, f1, offset):
    """2D diffusion model with a gaussian confocal volume."""
    UDC1 = twod(taus, tauD1)
    UDC2 = twod(taus, tauD2)
    G = offset + (f1 * UDC1 + (1 - f1) * UDC2) / n
    return G

def CF_2d_3c(taus, n, tauD1, tauD2, tauD3, f1,f2, offset):
    """2D diffusion model with a gaussian confocal volume."""
    UDC1 = twod(taus, tauD1)
    UDC2 = twod(taus, tauD2)
    UDC3 = twod(taus, tauD3)
    G = offset + (f1 * UDC1 + f2 * UDC2+ (1-f1-f2) * UDC3) / n
    return G

def CF_2d_1T(taus, n, tauD, tautr, T, offset):
    """2D diffusion model with a gaussian confocal volume."""
    UDC = twod(taus, tauD)
    UTC = trip(taus, tautr, T)
    G = offset + UDC * UTC / n
    return G

def CF_2da_1T(taus, n, tauD, a1, tautr, T, offset):
    """2D diffusion model with a gaussian confocal volume."""
    UDC = twoda(taus, tauD, a1)
    UTC = trip(taus, tautr, T)
    G = offset + UDC * UTC / n
    return G

def CF_2da_2c(taus, n, tauD1, a1, tauD2, a2, f1, offset):
    """2D diffusion model with a gaussian confocal volume."""
    UDC1 = twoda(taus, tauD1,a1)
    UDC2 = twoda(taus, tauD2,a2)
    G = offset + (f1 * UDC1 + (1 - f1) * UDC2) / n
    return G

def CF_2da_2c_1T(taus, n, tauD1, a1, tauD2, a2, f1, tautr, T, offset):
    """2D diffusion model with a gaussian confocal volume."""
    UDC1 = twoda(taus, tauD1,a1)
    UDC2 = twoda(taus, tauD2,a2)
    UTC = trip(taus, tautr, T)
    G = offset + (f1 * UDC1 + (1 - f1) * UDC2) * UTC / n
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
    """3D diffusion model with a Gaussian confocal volume and a single triplet state.
       Addition r0 parameter allows for fixing of the diffusion coefficient."""
    UDC = threedd(taus, D, r0, s)
    UTC = trip(taus, tautr1, T1)
    G = offset + UTC * UDC / n
    return G

def CF_3d_gauss_2T(taus, n, tauD, s, offset, tautr1, T1, tautr2, T2):
    """3D diffusion model with a gaussian confocal volume and two triplet states."""
    UDC = threed(taus, tauD, s)
    UTC1 = trip(taus, tautr1, T1)
    UTC2 = trip(taus, tautr2, T2)
    G = offset + UTC1 * UTC2 * UDC / n
    return G

def CF_3d_gauss_2T_fixD(taus, n, D, r0, s, offset, tautr1, T1, tautr2, T2):
    """3D diffusion model with a Gaussian confocal volume and two triplet states.
       Addition r0 parameter allows for fixing of the diffusion coefficient."""
    UDC = threedd(taus, D, r0, s)
    UTC1 = trip(taus, tautr1, T1)
    UTC2 = trip(taus, tautr2, T2)
    G = offset + UTC1 * UTC2 * UDC / n
    return G

def CF_3d_gauss_3T(taus, n, tauD, s, offset, tautr1, T1, tautr2, T2, tautr3, T3):
    """3D diffusion model with a gaussian confocal volume and three triplet states."""
    UDC = threed(taus, tauD, s)
    UTC1 = trip(taus, tautr1, T1)
    UTC2 = trip(taus, tautr2, T2)
    UTC3 = trip(taus, tautr3, T3)
    G = offset + UTC1 * UTC2 * UTC3 * UDC / n
    return G

def CF_3d_gauss_2c(taus, n, tauD1, tauD2, f1, s, offset):
    """Two-component 3D diffusion model with a gaussian confocal volume.
       This model assumes that each component has equal brightness.
       Corrections can be performed after to account for this following
       https://www.fcsxpert.com/classroom/theory/autocorrelation-diffusion-multiple.html"""
    UDC1 = threed(taus, tauD1, s)
    UDC2 = threed(taus, tauD2, s)
    G = offset + (f1 * UDC1 + (1 - f1) * UDC2) / n
    return G

def CF_3d_gauss_3c(taus, n, tauD1, tauD2, tauD3, f1, f2, s, offset):
    """Three-component 3D diffusion model with a gaussian confocal volume.
       This model assumes that each component has equal brightness."""
    UDC1 = threed(taus, tauD1, s)
    UDC2 = threed(taus, tauD2, s)
    UDC3 = threed(taus, tauD3, s)
    G = offset + (f1 * UDC1 + f2 * UDC2 + (1 - f1 - f2) * UDC3) / n
    return G

def weight_function(errors, n = 1):
    """Weight function for fitting."""
    return 1/errors**n
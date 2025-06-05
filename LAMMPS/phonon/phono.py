import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ── User parameters ────────────────────────────────────────────
files = [
    (r"$\Gamma = 0.245$", "vel_crystal.txt", "blue"),
    (r"$\Gamma = 1.627$", "vel_9400.txt", "cyan"),
    (r"$\Gamma = 3.655$", "vel_9750.txt", "lime"),
    (r"$\Gamma = 3.826$", "vel_hex.txt", "green"),
    (r"$\Gamma = 4.058$",  "vel_9900.txt", "orange"), 
    (r"$\Gamma = 4.414$",  "vel_10000.txt", "orangered"),
    (r"$\Gamma = 4.910$",  "vel_liquid.txt", "red"),
]
n_atoms  = 11040        # number of atoms per frame
dt       = 1e-15  # sampling interval (here 10 steps × 0.1 fs = 1 fs)
T        = 300.0        # temperature (K)
# ────────────────────────────────────────────────────────────────

# Physical constants
m   = 12.011 * 1.66054e-27  # carbon mass (kg)
kB  = 1.380649e-23          # Boltzmann constant (J/K)

def compute_dos(filename):
    # 1) Load velocities (drop the id column)
    data = np.loadtxt(filename)
    vel  = data[:,1:4]  # (n_frames*n_atoms, 3)

    # 2) Reshape into (n_frames, n_atoms, 3)
    n_frames = vel.shape[0] // n_atoms
    vel = vel.reshape(n_frames, n_atoms, 3)

    # 3) FFT‐based VACF
    mom      = vel * m
    flat     = mom.reshape(n_frames, -1)
    n_pad    = 1 << (2*n_frames - 1).bit_length()
    F        = rfft(flat, n=n_pad, axis=0)
    acf_full = irfft(F * np.conjugate(F), axis=0)[:n_frames]
    vacf     = acf_full.sum(axis=1)
    vacf    /= vacf[0]
    # 4) FFT VACF → raw spectrum
    spec = rfft(vacf)
    spec *= (2 * dt / np.pi)
    # 5) Classical DOS normalisation
    dos = spec.real * (m / (3 * n_atoms * kB * T))
    # 6) Frequency axis in THz
    freq_Hz = rfftfreq(n_frames, dt)
    freq_THz = freq_Hz * 1e-12
    return freq_THz, dos

plt.figure(figsize=(8,6))
offsets = np.linspace(0, 20, len(files))

# Plot each DOS curve with its offset
for (label, fname, clr), off in zip(files, offsets):
    freq_THz, dos = compute_dos(fname)
    freq_cm = freq_THz * 1e12 / 2.99792458e10
    sm = gaussian_filter1d(dos * 1e24, sigma=0) + off
    plt.plot(freq_cm, sm, lw=1.5, label=label, zorder=1, c=clr)
    # label the curve at its rightmost end
    plt.text(2100, 1.01*off, label,va='center', ha='left',fontsize=14, fontweight='bold', color='k', zorder=3)

# Final tweaks
plt.xlim(0, 2000)
plt.yticks([])
plt.xlabel('Frequency (cm$^{-1}$)', fontsize=16)
plt.ylabel('Phonon DOS (arb. units)', fontsize=16)
plt.tight_layout()
plt.savefig('PhDOS.png', dpi=1200)
plt.show()

import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ── User parameters ────────────────────────────────────────────
files = [
    ("Crystal", "vel_crystal.txt"),
    ("Hexatic", "vel_hexatic.txt"),
    ("Liquid",  "vel_liquid.txt"),
]
n_atoms  = 11040        # number of atoms per frame
dt       = 10 * 0.1e-15  # sampling interval (here 10 steps × 0.1 fs = 1 fs)
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
ax = plt.gca()

offsets = np.linspace(0, 8.0, len(files)) 

# Key mode positions in cm⁻¹
lines_cm = {
    'ZA/ZO @ M (≈300)': 300,
    'ZO @ M (≈550)':   550,
    'E₂g @ Γ (≈1580)': 1580,
}

# Plot each DOS curve with its offset
for (label, fname), off in zip(files, offsets):
    freq_THz, dos = compute_dos(fname)
    freq_cm = freq_THz * 1e12 / 2.99792458e10
    sm = gaussian_filter1d(dos * 1e24, sigma=2) + off
    ax.plot(freq_cm, sm, lw=1.5, label=label, zorder=1)
    # label the curve at its rightmost end
    ax.text(2100, off, label,
            va='center', ha='left',
            fontsize=12, fontweight='bold', color='k', zorder=3)

# Draw vertical guide lines and horizontal labels
ymax = offsets[-1] + 0.5
for idx, (text, x) in enumerate(lines_cm.items()):
    ax.axvline(x, color='k', ls='--', lw=1, zorder=0)
    # stagger the labels vertically to avoid overlap
    y_loc = ymax - 0.2*idx
    ax.text(x, y_loc, text,
            va='bottom', ha='center',
            fontsize=11, fontweight='bold', backgroundcolor='white',
            zorder=4)

# Final tweaks
ax.set_xlim(0, 2100)
ax.set_ylim(-0.2, ymax + 0.2)
ax.set_xlabel('Frequency (cm$^{-1}$)', fontsize=14)
ax.set_ylabel('g(ω) + offset (arb. units)', fontsize=14)
ax.set_yticks([])   # no y‐ticks when using offsets
plt.tight_layout()
plt.savefig('PhDOS_offsets_styled.png', dpi=300)
plt.show()
from scipy.signal import find_peaks
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

def minimum_image_vec(dxy, Lx, Ly):
    dxy[:, 0] -= np.round(dxy[:, 0] / Lx) * Lx
    dxy[:, 1] -= np.round(dxy[:, 1] / Ly) * Ly
    return dxy

def compute_psi6(tree, pos, Lx, Ly, r_cut):
    """ψ6(i) with the *same* cKDTree used later for pairs."""
    nbr_lists = tree.query_ball_point(pos, r_cut)
    psi6 = np.empty(len(pos), dtype=np.complex128)

    for i, nbrs in enumerate(nbr_lists):
        if i in nbrs:
            nbrs.remove(i)
        if not nbrs:
            psi6[i] = 0.0
            continue
        vecs = pos[nbrs] - pos[i]
        vecs = minimum_image_vec(vecs, Lx, Ly)
        theta = np.arctan2(vecs[:, 1], vecs[:, 0])
        psi6[i] = np.exp(1j * 6.0 * theta).mean()
    return psi6

def g6(pos, dr, r_cut=1.2):
    """
    Bond‑orientational correlation g6(r) without any bin‑threshold masking.
    """
    # ----- box size ----------------------------------------------------
    Lx, Ly = np.ptp(pos[:, 0])+50, np.ptp(pos[:, 1])+50
    r_max = 0.5*min(Lx, Ly)

    # ----- neighbour tree ---------------------------------------------
    tree = cKDTree(pos)
    psi6 = compute_psi6(tree, pos, Lx, Ly, r_cut)
    psi6_mag2 = (np.abs(psi6) ** 2).mean()

    # ----- all pairs up to r_max --------------------------------------
    pairs = np.array(list(tree.query_pairs(r_max)))
    dxy = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    dxy = minimum_image_vec(dxy, Lx, Ly)
    r = np.hypot(dxy[:, 0], dxy[:, 1])

    # ----- histogram ---------------------------------------------------
    nbins = int(np.ceil(r_max / dr))
    bin_idx = np.floor(r / dr).astype(np.int64)

    num = np.zeros(nbins, dtype=np.complex128)
    counts = np.bincount(bin_idx, minlength=nbins)
    weights = psi6[pairs[:, 0]] * np.conj(psi6[pairs[:, 1]])
    np.add.at(num, bin_idx, weights)

    # ----- g6(r) -------------------------------------------------------
    r_cent = (np.arange(nbins) + 0.5) * dr
    with np.errstate(divide="ignore", invalid="ignore"):
        g6_r = (num / counts).real / psi6_mag2   
    g6_r[counts == 0] = np.nan                  # bins with no statistics

    return r_cent, g6_r, counts                


a = 1.42
hexatic_labels = {
    r'$\Gamma = 1.12$', r'$\Gamma = 1.00$',
    r'$\Gamma = 0.96$', r'$\Gamma = 0.90$'
}

plt.figure(figsize=(10,6))
for label, fname, color, rcut in [
    (r'$\Gamma = 1.12$', 'data/7000K.txt',  "cyan",   1.3),
    (r'$\Gamma = 1.00$', 'data/7750K.txt',  "navy",   1.3),
    (r'$\Gamma = 0.96$', 'data/8000K.txt',  "lime",   1.3),
    (r'$\Gamma = 0.90$', 'data/8250K.txt',  "green",  1.3),
    (r'$\Gamma = 0.85$', 'data/8500K.txt',  "orange", 1.3),
    (r'$\Gamma = 0.83$', 'data/9000K.txt',  "red",    1.3)
]:
    # --- same loop you have ---
    pos = np.loadtxt(fname)[:,2:4] / a
    Lx, Ly = np.ptp(pos[:,0]) + 50, np.ptp(pos[:,1]) + 50
    r_max = 0.5 * min(Lx, Ly)
    r, g6_r, _ = g6(pos, dr=0.075, r_cut=min(rcut, r_max))
    g6_smooth = gaussian_filter1d(np.nan_to_num(g6_r, nan=0.0), sigma=1.5)

    plt.loglog(r, g6_smooth, c=color, label=label, lw=2)
    i0 = np.argmin(np.abs(r - 1.0))
    g0 = g6_smooth[i0]
    peaks, _ = find_peaks(g6_smooth, distance=int(1.0/0.075))
    r_peaks, g_peaks = r[peaks], g6_smooth[peaks]

    if label in hexatic_labels:
        rmin, rmax = 0.0, 10.0
        mask = (r_peaks >= rmin) & (r_peaks <= rmax)
        ln_rp = np.log(r_peaks[mask])
        ln_gp = np.log(g_peaks[mask])
        slope, _ = np.polyfit(ln_rp, ln_gp, 1)
        chi = -slope
        fit = g0 * r**(-chi)
        fit_r = r
    else:
        rmin2, rmax2 = 0.0, 5.0
        mask2 = (r_peaks >= rmin2) & (r_peaks <= rmax2)
        rp2 = r_peaks[mask2]
        ln_gp2 = np.log(g_peaks[mask2])
        slope2, _ = np.polyfit(rp2, ln_gp2, 1)
        xi = -1.0 / slope2
        A2 = g0 * np.exp(1.0/xi)
        fit_r = r[(r>=rmin2)&(r<=rmax2)]
        fit   = A2 * np.exp(-fit_r/xi)

    plt.loglog(fit_r, fit, ls='--', c=color, lw=2)

plt.xlabel(r'$r/a$', fontsize=18)
plt.ylabel(r'$g_6(r)$', fontsize=18)
plt.xlim(0.8, 55)
plt.ylim(1e-2, 1)
plt.grid(True, which="both", ls="--")
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.savefig('hexatic.png', dpi=150)

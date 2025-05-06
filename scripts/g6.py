import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

# ──────────────────────────────────────────────────────────────────────
def minimum_image_vec(dxy, Lx, Ly):
    dxy[:, 0] -= np.round(dxy[:, 0] / Lx) * Lx
    dxy[:, 1] -= np.round(dxy[:, 1] / Ly) * Ly
    return dxy

# ──────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────
def g6(pos, dr=0.1, r_max=None, r_cut=1.2):
    """
    Bond‑orientational correlation g6(r) without any bin‑threshold masking.
    Bins with zero pairs are returned as NaN so you can decide later whether
    to plot them, interpolate, etc.
    """
    # ----- box size ----------------------------------------------------
    Lx, Ly = np.ptp(pos[:, 0]), np.ptp(pos[:, 1])
    if r_max is None:
        r_max = 0.5 * min(Lx, Ly)

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
        g6_r = (num / counts).real / psi6_mag2   # keep the sign, drop abs()
    g6_r[counts == 0] = np.nan                  # bins with no statistics

    return r_cent, g6_r, counts                 # counts returned for reference

a = 1.42
plt.figure()

for lbl, fn, c, cut in [('crystal','data/300K.txt',"b", 1.2),
                        ('"hexatic"', 'data/hexatic.txt', "lime", 1.2*a),
                        ('liquid', 'data/liquid.txt', "r", 1.3)]:
    pos = np.loadtxt(fn)[:,2:4]
    r, g, _ = g6(pos/a, dr=0.05, r_cut=cut)
    # you can smooth if you like, just beware of NaNs
    g6_smooth = gaussian_filter1d(np.nan_to_num(g, nan=0.0), sigma=2)
    plt.plot(r, g6_smooth, label=lbl, c=c)
        
plt.loglog(r, 0.9*r**(-1/4), c='k', label=r"$\sim r^{-1/4}$")
plt.xlabel(r'$r/a$')
plt.ylabel(r'$g_6(r)$')
plt.xlim(1, 80/a)
plt.ylim(1e-2, 1)
plt.legend()
plt.tight_layout()
plt.savefig("output.pdf")
plt.show()
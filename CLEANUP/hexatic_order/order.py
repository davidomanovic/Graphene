import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional

def g6(
    positions: np.ndarray,
    dr: float,
    r_cut: float = 1.2,
    box: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-particle ψ₆, its global <|ψ₆|²>, and the bond-orientational
    correlation function g₆(r).

    Parameters
    ----------
    positions : (N,2) float ndarray
        x,y coordinates of N particles.
    dr : float
        Bin width for g₆(r).
    r_cut : float, optional
        Nearest-neighbour cutoff for computing ψ₆ (default 1.2).
    box : tuple (Lx, Ly), optional
        Periodic box dimensions.  If None, the Lx,Ly are estimated as
        ptp(positions)+100 (not recommended for production).

    Returns
    -------
    r_centers : (M,) ndarray
        Radial bin centers for g₆(r).
    g6_r : (M,) ndarray
        Bond-orientational correlation vs r.
    counts : (M,) int ndarray
        Number of pairs contributing to each bin.
    """
    N = positions.shape[0]
    if box is None:
        Lx = np.ptp(positions[:,0]) + 100
        Ly = np.ptp(positions[:,1]) + 100
    else:
        Lx, Ly = box

    # Build KD-tree for neighbour lists
    tree = cKDTree(positions)
    # compute ψ₆ for each particle
    sm = tree.sparse_distance_matrix(tree, r_cut, output_type="coo_matrix")
    i, j = sm.row, sm.col
    mask = i != j
    i, j = i[mask], j[mask]

    # Apply minimum-image convention
    dxy = positions[j] - positions[i]
    dxy[:,0] -= np.round(dxy[:,0]/Lx) * Lx
    dxy[:,1] -= np.round(dxy[:,1]/Ly) * Ly

    theta = np.arctan2(dxy[:,1], dxy[:,0])
    w = np.exp(1j*6*theta)

    sums = np.zeros(N, complex)
    counts = np.zeros(N, int)
    np.add.at(sums,     i, w)
    np.add.at(counts,   i, 1)

    psi6 = np.zeros(N, complex)
    nonzero = counts>0
    psi6[nonzero] = sums[nonzero]/counts[nonzero]

    # g6(r)
    r_max = 0.5 * min(Lx, Ly)
    sm2 = tree.sparse_distance_matrix(tree, r_max, output_type="coo_matrix")
    i2, j2 = sm2.row, sm2.col
    mask2 = i2<j2
    i2, j2 = i2[mask2], j2[mask2]

    dxy2 = positions[j2] - positions[i2]
    dxy2[:,0] -= np.round(dxy2[:,0]/Lx) * Lx
    dxy2[:,1] -= np.round(dxy2[:,1]/Ly) * Ly
    r = np.hypot(dxy2[:,0], dxy2[:,1])

    nbins = int(np.ceil(r_max/dr))
    bin_idx = np.floor(r/dr).astype(int)
    weights = psi6[i2] * np.conj(psi6[j2])

    num    = np.bincount(bin_idx, minlength=nbins, 
                        weights=weights.real) \
           + 1j*np.bincount(bin_idx, minlength=nbins,
                            weights=weights.imag)
    cnts   = np.bincount(bin_idx, minlength=nbins)

    r_centers = (np.arange(nbins) + 0.5)*dr
    with np.errstate(divide='ignore', invalid='ignore'):
        g6_r = (num/cnts).real
    g6_r[cnts==0] = np.nan

    return r_centers, g6_r, cnts
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from collections import defaultdict
frames = np.load("9690.npy")

n_frames, n_atoms, _ = frames.shape

k_b     = 469.0    # kcal/(mol·Å²)
b0      = 1.42     # Å
k_theta = 63.0     # kcal/(mol·rad²)
theta0  = 120.0 * np.pi/180.0  # radians
k_B     = 0.0019872041         # kcal/(mol·K)


def find_dynamic_cutoff(positions, factor=1.2):
    dists, _ = cKDTree(positions).query(positions, k=4)
    bond_len = np.median(dists[:, 3])
    return factor * bond_len

def build_neighbor_list(positions):
    tree = cKDTree(positions)
    R    = find_dynamic_cutoff(positions)
    raw  = tree.query_ball_point(positions, r=R)
    N     = len(positions)
    neigh = [set() for _ in range(N)]
    for i, cand in enumerate(raw):
        ds = np.linalg.norm(positions[cand] - positions[i], axis=1)
        picks = [idx for idx, _ in sorted(zip(cand, ds), key=lambda x: x[1])
                 if idx != i][:3]
        for j in picks:
            neigh[i].add(j)
    for i in range(N):
        for j in neigh[i]:
            neigh[j].add(i)
    return [list(s) for s in neigh]

def compute_forces_and_counts(coords):
    """
    Given coords (n_atoms,2), build neighbor_list, then
    - compute forces array (n_atoms,2)
    - return forces, n_bonds, n_angles
    """
    neighbor_list = build_neighbor_list(coords)

    # build bond_list (i<j)
    bonds = []
    for i, nbrs in enumerate(neighbor_list):
        for j in nbrs:
            if j > i:
                bonds.append((i, j))
    bond_list = np.array(bonds, dtype=int)
    n_bonds = len(bond_list)

    # build angles_idx
    angles_idx = []
    for c, nbrs in enumerate(neighbor_list):
        if len(nbrs) < 2:
            continue
        for ii in range(len(nbrs)):
            for jj in range(ii+1, len(nbrs)):
                angles_idx.append((nbrs[ii], c, nbrs[jj]))
    angles_idx = np.array(angles_idx, dtype=int)
    n_angles = len(angles_idx)

    # init forces
    forces = np.zeros((n_atoms, 2))

    # bond forces
    for i, j in bond_list:
        rij  = coords[i] - coords[j]
        b    = np.linalg.norm(rij)
        dUdb = 2 * k_b * (b - b0)
        fij  = -dUdb * (rij / b)
        forces[i] +=  fij
        forces[j] += -fij

    # angle forces
    for i, c, j in angles_idx:
        vci = coords[i] - coords[c]
        vcj = coords[j] - coords[c]
        ri, rj = np.linalg.norm(vci), np.linalg.norm(vcj)
        cosθ = np.dot(vci, vcj) / (ri * rj)
        cosθ = np.clip(cosθ, -1.0, 1.0)
        θ    = np.arccos(cosθ)
        dUdθ = 2 * k_theta * (θ - theta0)
        sinθ = np.sqrt(1.0 - cosθ*cosθ)
        if sinθ < 1e-8:
            continue
        fi = -dUdθ/(ri*sinθ) * ((vcj/rj) - cosθ*(vci/ri))
        fj = -dUdθ/(rj*sinθ) * ((vci/ri) - cosθ*(vcj/rj))
        fc = -(fi + fj)
        forces[i] += fi
        forces[j] += fj
        forces[c] += fc

    return forces, n_bonds, n_angles

sum_F2     = 0.0
sum_bonds  = 0
sum_angles = 0

for frame in frames:
    forces, nb, na = compute_forces_and_counts(frame)
    sum_F2     += np.sum(np.linalg.norm(forces, axis=1)**2)
    sum_bonds  += nb
    sum_angles += na

trace_bonds  = 4.0 * k_b
trace_angles = (14.0/3.0) * k_theta

sum_trace = trace_bonds*sum_bonds + trace_angles*sum_angles

T_conf = sum_F2 / (k_B * sum_trace)

print(f"Configurational temperature = {T_conf:.1f} K")


# 2) Write to a text file
with open("T_conf.txt", "w") as f:
    f.write(f"{T_conf:.6f}\n")

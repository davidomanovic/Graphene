import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── Load trajectories ─────────────────────────────────────────────────────────
crystal = np.load("crystal.npy")   # shape = (n_frames, n_atoms, 2)
hexatic = np.load("hexatic.npy")
liquid  = np.load("liquid.npy")

# ── Force‐field constants ───────────────────────────────────────────────────────
k_b     = 469.0    # kcal/(mol·Å²)
b0      = 1.42     # Å
k_theta = 63.0     # kcal/(mol·rad²)
theta0  = 120.0 * np.pi/180.0  # radians

# ── Helpers ────────────────────────────────────────────────────────────────────
def find_dynamic_cutoff(positions, factor=1.2):
    dists, _ = cKDTree(positions).query(positions, k=4)
    return factor * np.median(dists[:, 3])

def build_neighbor_list(positions):
    tree = cKDTree(positions)
    R    = find_dynamic_cutoff(positions)
    raw  = tree.query_ball_point(positions, r=R)
    N    = len(positions)
    neigh= [set() for _ in range(N)]
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

def compute_total_energy(coords):
    nbrs = build_neighbor_list(coords)
    E_bond = 0.0
    for i, neigh in enumerate(nbrs):
        for j in neigh:
            if j>i:
                b = np.linalg.norm(coords[i]-coords[j])
                E_bond += k_b*(b - b0)**2
    E_ang = 0.0
    for c, neigh in enumerate(nbrs):
        L = len(neigh)
        if L<2: continue
        for ii in range(L):
            for jj in range(ii+1, L):
                i, j = neigh[ii], neigh[jj]
                vi = coords[i]-coords[c]
                vj = coords[j]-coords[c]
                cosθ = np.dot(vi,vj)/(np.linalg.norm(vi)*np.linalg.norm(vj))
                cosθ = np.clip(cosθ, -1, 1)
                θ = np.arccos(cosθ)
                E_ang += k_theta*(θ - theta0)**2
    return E_bond + E_ang

# ── Build total‐energy arrays ──────────────────────────────────────────────────
E_crys = np.array([compute_total_energy(f) for f in crystal])
E_hex  = np.array([compute_total_energy(f) for f in hexatic])
E_liq  = np.array([compute_total_energy(f) for f in liquid])

# ── Fit Gaussians and extract σ ───────────────────────────────────────────────
µc, σc = norm.fit(E_crys)
µh, σh = norm.fit(E_hex)
µl, σl = norm.fit(E_liq)

print(f"Crystal:  μ={µc:.1f}, σ={σc:.1f} kcal/mol")
print(f"Hexatic:  μ={µh:.1f}, σ={σh:.1f} kcal/mol")
print(f"Liquid:   μ={µl:.1f}, σ={σl:.1f} kcal/mol")

with open("energy_widths.txt","w") as out:
    out.write(f"phase\tmu\tsigma\n")
    out.write(f"crystal\t{µc:.6f}\t{σc:.6f}\n")
    out.write(f"hexatic\t{µh:.6f}\t{σh:.6f}\n")
    out.write(f"liquid\t{µl:.6f}\t{σl:.6f}\n")

# ── Plot overlaid histograms ───────────────────────────────────────────────────
plt.figure(figsize=(7,4))
bins = 40
plt.hist(E_crys, bins=bins, density=True, alpha=0.5, label=f'Crystal (σ={σc:.1f})')
plt.hist(E_hex,  bins=bins, density=True, alpha=0.5, label=f'Hexatic (σ={σh:.1f})')
plt.hist(E_liq,  bins=bins, density=True, alpha=0.5, label=f'Liquid (σ={σl:.1f})')

# optionally overplot the Gaussian fits
x = np.linspace(min(E_crys.min(),E_hex.min(),E_liq.min()),
                max(E_crys.max(),E_hex.max(),E_liq.max()), 200)
plt.plot(x, norm.pdf(x, µc, σc), 'C0--')
plt.plot(x, norm.pdf(x, µh, σh), 'C1--')
plt.plot(x, norm.pdf(x, µl, σl), 'C2--')

plt.xlabel('Total potential energy (kcal/mol)')
plt.ylabel('Probability density')
plt.title('Total energy distributions & Gaussian fits')
plt.legend()
plt.tight_layout()
plt.savefig("energy_distributions.png", dpi=300)
plt.show()

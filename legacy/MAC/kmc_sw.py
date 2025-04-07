import random, math, subprocess
import numpy as np
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ase.neighborlist import NeighborList

KBT = 0.5
MAX_ITERS = 60000
TARGET_ITER = 20000
LAMMPS_CMD = ["srun", "lmp_mpi", "-in", "minimize.in"]

def minimize_with_lammps(datafile):
    subprocess.run(LAMMPS_CMD + ["-var", f"datafile={datafile}"], check=True)
    # Parse energy from log.lammps
    with open("log.lammps") as f:
        for line in reversed(f.readlines()):
            if "PotEng" in line:
                return float(line.split()[2])
    raise RuntimeError("Could not read energy from log.lammps")

def get_random_bond(atoms):
    nl = NeighborList([1.6]*len(atoms), skin=0.0, self_interaction=False)
    nl.update(atoms)
    i = random.randrange(len(atoms))
    neigh, _ = nl.get_neighbors(i)
    return None if not neigh else (i, random.choice(neigh))

def apply_stone_wales(atoms, bond):
    i, j = bond
    coords = atoms.get_positions()
    center = (coords[i] + coords[j]) / 2
    new = atoms.copy()
    for idx in [i, j]:
        vec = coords[idx] - center
        new.positions[idx] = center + np.array([-vec[1], vec[0], 0.0])
    return new

# Load minimized starting structure
atoms = read_lammps_data("minimized_carbon.data")
E_old = minimize_with_lammps("minimized_carbon.data")

for it in range(1, MAX_ITERS + 1):
    bond = get_random_bond(atoms)
    if bond is None: continue

    new_atoms = apply_stone_wales(atoms, bond)
    write_lammps_data("temp.data", new_atoms, atom_style="atomic", masses={1:12.011})

    E_new = minimize_with_lammps("temp.data")
    if random.random() < math.exp(-(E_new - E_old)/KBT):
        atoms, E_old = new_atoms, E_new

    if it == TARGET_ITER:
        write_lammps_data("data.MAC_iter20000", atoms, atom_style="atomic", masses={1:12.011})
        print(f"Saved MAC at iteration {it}")

print("KMC complete — final energy:", E_old)

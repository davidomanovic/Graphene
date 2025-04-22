import numpy as np

# Parameters
n_atoms = 610
box_x = 40.0  # Å
box_y = 40.0  # Å
box_z = 1.0   # Minimal z-dimension for a 2D simulation
min_distance = 1.2  # Minimum allowed distance between atoms

def is_far_enough(pos, positions, min_distance):
    """Check that pos is at least min_distance away from all positions in the list."""
    for p in positions:
        if np.linalg.norm(pos - p) < min_distance:
            return False
    return True

# Generate positions with rejection sampling
positions = []
attempts = 0
while len(positions) < n_atoms:
    candidate = np.array([np.random.uniform(0, box_x),
                          np.random.uniform(0, box_y),
                          np.random.uniform(0, box_z)])
    if is_far_enough(candidate, positions, min_distance):
        positions.append(candidate)
    attempts += 1
    if attempts > 100000:
        raise RuntimeError("Too many attempts; try lowering the minimum distance or increasing the box size.")

# Write LAMMPS data file
with open("lammps_datafile.data", "w") as f:
    f.write("LAMMPS data file for 2D carbon system\n\n")
    f.write(f"{n_atoms} atoms\n\n")
    f.write("1 atom types\n\n")
    f.write(f"0 {box_x} xlo xhi\n")
    f.write(f"0 {box_y} ylo yhi\n")
    f.write(f"0 {box_z} zlo zhi\n\n")
    f.write("Masses\n\n")
    f.write("1 12.01\n\n")  # Carbon atomic mass

    f.write("Atoms\n\n")
    for i, pos in enumerate(positions, start=1):
        # Format: atom-ID atom-type x y z
        f.write(f"{i} 1 {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n")

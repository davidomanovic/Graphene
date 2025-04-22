import numpy as np

def assign_temperature(timestep, mapping):
    """
    Return the target temperature for a given timestep using the provided mapping.
    Each mapping tuple is either:
      - a 3-tuple: (t_start, t_end, T_const) or 
      - a 4-tuple: (t_start, t_end, T_start, T_end) for a ramp.
    """
    for m in mapping:
        if len(m) == 3:
            t_start, t_end, T_const = m
            if t_start <= timestep < t_end:
                return T_const
        elif len(m) == 4:
            t_start, t_end, T_start, T_end = m
            if t_start <= timestep < t_end:
                frac = (timestep - t_start) / (t_end - t_start)
                return T_start + frac * (T_end - T_start)
    return None

def parse_lammps_dump(filename):
    """
    Parse a LAMMPS dump file into a list of frames.
    Each frame is stored as a dictionary with keys 'timestep' and 'positions' (numpy array with shape (n_atoms, 3)).
    (Adjust the parsing below to fit your dump file format.)
    """
    frames = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            timestep = int(lines[i+1].strip())
            n_atoms = int(lines[i+3].strip())
            i += 9  # Adjust based on header length; here we assume 9 header lines.
            positions = np.zeros((n_atoms, 3))
            for j in range(n_atoms):
                parts = lines[i+j].split()
                positions[j, 0] = float(parts[2])
                positions[j, 1] = float(parts[3])
                positions[j, 2] = float(parts[4])
            frames.append({"timestep": timestep, "positions": positions})
            i += n_atoms
        else:
            i += 1
    return frames

def filter_frames_by_temperature(frames, mapping, target_temp, tolerance=20):
    """
    Return a list of frames whose assigned temperature (computed using mapping) is within ±tolerance of target_temp.
    """
    selected = []
    for frame in frames:
        temp = assign_temperature(frame["timestep"], mapping)
        if temp is None:
            continue
        if abs(temp - target_temp) <= tolerance:
            selected.append(frame)
    return selected

def write_frames_to_file(frames, fname):
    """
    Write the given frames to one text file.
    """
    with open(fname, "w") as f:
        f.write(f"# Positions for selected frames\n")
        for idx, frame in enumerate(frames):
            f.write(f"# Frame {idx}, Timestep {frame['timestep']}\n")
            positions = frame["positions"]
            for j, pos in enumerate(positions):
                f.write(f"{j+1} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            f.write("\n")
    print(f"Wrote {len(frames)} frames into {fname}")

def main():
    # Use the dump file from your input file.
    dump_filename = "dmp.lammpstrj"
    frames = parse_lammps_dump(dump_filename)
    print("Total frames parsed:", len(frames))
    
    # Define your mapping according to your simulation protocol.
    mapping = [
        (0, 1400000, 0, 14000), 
        (1400000, 2400000, 14000), 
        (2400000, 2550700, 14000, 300),
        (2550700, 3550700, 300)
    ]

    # For example, if you want to compute G6 only for 0-300 K:
    target_temp = 5000  # desired temperature (K)
    tolerance = 0     # tolerance in K
    
    selected_frames = filter_frames_by_temperature(frames, mapping, target_temp, tolerance)
    print(f"Selected {len(selected_frames)} frames around {target_temp} ± {tolerance} K.")
    
    # Write all these selected frames into one file.
    write_frames_to_file(selected_frames, f"positions_{target_temp}_K.txt")
    
    # You can now proceed to compute G6 orientational correlations
    # from these selected frames.
    
if __name__ == "__main__":
    main()

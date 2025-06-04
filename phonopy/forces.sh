#!/usr/bin/env bash
set -euo pipefail

# ——————————————————————————————————————————————————————————————————
# CONFIGURATION
# MPI-enabled LAMMPS executable:
LAMMPS_EXE="lmp_mpi"

# Path (relative to each disp-*/) to your LAMMPS template that
# contains the "__DATAFILE__" placeholder:
TEMPLATE="../force_template.in"
# ——————————————————————————————————————————————————————————————————

for disp in disp-*/; do
  # only process real directories
  if [[ ! -d "$disp" ]]; then
    continue
  fi

  echo "=== entering $disp ==="
  (
    cd "$disp"

    # look for exactly one .data file (adjust pattern if needed)
    DATAFILES=( data.lmp )
    if [[ ${#DATAFILES[@]} -ne 1 ]]; then
      echo "⚠️  Expected one .data file, found ${#DATAFILES[@]} in $disp; skipping." >&2
      exit 1
    fi
    DATAFILE="${DATAFILES[0]}"

    # generate the actual LAMMPS input by substituting __DATAFILE__
    sed "s|__DATAFILE__|$DATAFILE|" "$TEMPLATE" > phonopy_lmp.in

    # run LAMMPS to dump forces
    $LAMMPS_EXE -in phonopy_lmp.in
  )
  echo "=== done $disp ==="$'\n'
done

echo "✅ All displacements processed."

for d in disp-*/; do
  dd=${d%/}

  # 1) run LAMMPS as before, producing dump.forces
  lmp_mpi -in in.${dd}.force

  # 2) strip the header and write exactly "id fx fy fz"
  awk '/^ITEM: ATOMS/{hdr=1; next} hdr && NF==4 {print}' dump.forces \
      "${dd}/forces.dat"

  # 3) clean up
  rm dump.forces log.lammps
done

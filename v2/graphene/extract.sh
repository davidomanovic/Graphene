awk '/^ *[0-9]/ && NF>=7 {
  # $1=Step, $2=Temp, $3=E_pair, $4=E_mol, $5=TotEng, $6=Press, $7=Volume
  printf "%s %s %s %s %s %s %s\n", $1,$2,$3,$4,$5,$6,$7
}' log.lammps > thermo_data.txt

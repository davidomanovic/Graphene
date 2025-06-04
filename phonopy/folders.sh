for f in POSCAR-*; do
  # Extract ID (e.g. "001")
  id=${f#POSCAR-}
  folder=$(printf "disp-%s" "$id")
  mkdir -p "$folder"
  mv "$f" "$folder/POSCAR"
done
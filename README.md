# Graphene

Reference implementation of the simulations and post-processing pipeline used in  
**“Ab-initio defect study of disordered two-dimensional graphene structures”**  
(EPFL TPIV - Physics Project II, Spring 2025).

## Reproduce the main figures

```bash
conda env create -f environment.yml        # installs LAMMPS + analysis deps
conda activate kthny-md

bash  run_md.sh            # heats + quenches graphene, writes dump files
python analyse.py          # extracts Γ, g6(r), gT(r), phonon DOS, etc.
jupyter nbconvert figures.ipynb --to html   # regenerates plots 3.1 – 3.17

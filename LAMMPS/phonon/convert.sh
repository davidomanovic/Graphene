#!/bin/bash
awk '/^ITEM: ATOMS/ {p=1; next}
     /^ITEM:/       {p=0}
    p              {print}' vel_crystal.dat > vel_crystal_2.txt

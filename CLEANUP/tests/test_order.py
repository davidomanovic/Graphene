import numpy as np
from hexatic_order.order import g6

import numpy as np
from hexatic_order.order import g6

def make_hex_lattice(nx, ny, a=1.0):
    coords = []
    for i in range(nx):
        for j in range(ny):
            x = i*a + (j%2)*a/2
            y = j*a*np.sqrt(3)/2
            coords.append([x,y])
    return np.array(coords)

def test_g6_perfect_hex():
    pos = make_hex_lattice(5,5, a=1.0)
    dr = 0.2
    Lx = np.ptp(pos[:,0]) + 0.1
    Ly = np.ptp(pos[:,1]) + 0.1

    r_cent, g6_r, cnts = g6(pos, dr, r_cut=1.1, box=(Lx,Ly))

    # find the bin index closest to the nearest-neighbor distance = 1.0
    nn_dist = 1.0
    idx = np.argmin(np.abs(r_cent - nn_dist))

    # now assert on that bin
    assert cnts[idx] > 0, "no pairs in the nearest-neighbour bin!"
    assert np.isclose(g6_r[idx], 1.0, atol=1e-6)
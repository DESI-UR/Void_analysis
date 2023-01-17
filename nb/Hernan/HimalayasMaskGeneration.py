from astropy.io import fits
from astropy.table import Table
from desimodel.footprint import tiles2pix  
import numpy as np

#This is Anand Raichoor's code, modified by Hernan Rincon to generate a mask for himalayas

# read the tiles-specstatus to get the qa-validated tiles.
# tiles must contains (RA, DEC) so that tiles2pix works
fn = "tiles-specstatus.ecsv"
tiles = Table.read(fn)
#need to choose more accurate date for himilayas, I just guessed
sel = (tiles["FAFLAVOR"] == "mainbright") & (tiles["QA"] == "good") & (tiles["LASTNIGHT"] <= 20220601)
tiles = tiles[sel] # <= 3596 tiles

# read my massive healpix map
# which stores in TILEIDS (npixels, npass) the list of tiles covering each pixel
fn = "main-skymap-bright-goal.fits"
hdr = fits.getheader(fn, 1)
d = fits.open(fn)[1].data # fits.open is much faster than fitsio.read...
nside = int((len(d['HPXPIXEL'])/12)**.5)
npass = d["TILEIDS"].shape[1] # npass=4 for bright
goal_ns = d["NPASS"] # number of planned tiles covering each pixel


# now count how many qa-validated tiles are done for each pixel
ns = np.nan + np.zeros(len(d))
ns[goal_ns > 0] = 0.0
for i in range(npass):
    sel = np.in1d(tiles["TILEID"], d["TILEIDS"][:, i])
    if sel.sum() > 0:
        tileids_i = tiles["TILEID"][sel]
        ipixs = tiles2pix(nside, tiles=tiles[sel])
        ns[ipixs] += 1

# fractional coverage
fracns = np.nan + np.zeros(len(d))
sel = goal_ns > 0
fracns[sel] = ns[sel] / goal_ns[sel]

# store results in a new table
myd = Table()
for key in ["HPXPIXEL", "RA", "DEC"]:
    myd[key] = d[key]
myd["DESI"] = goal_ns > 0 # pixels covered by desi
myd["DONE"] = fracns == 1 # pixels where all tiles are obs.+qa-validated
myd.meta["HPXNSIDE"] = hdr["HPXNSIDE"]
myd.meta["HPXNEST"] = hdr["HPXNEST"]
myd.write("himalayas_mask.fits")
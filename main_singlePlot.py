
from pylib.pth import path
from pylib.cbs import cbs_load
from pylib.grd import grd_load
from pylib.xmf import xmf_load
from pylib.gmt import gmt_load

#####################################################################
## Time and radial points of interest.                             ##
#####################################################################
irad  = 15
itime = 100

#####################################################################
## Preparing grid information about the cubed-sphere grid and the  ##
## target lat-lon grid. Based on that, the GMT plotter and XESMF   ##
## regridder are prepared. Loader of cubed-sphere datasets is also ##
## prepared in advance. Upon changing the resolution, it is needed ##
## to set use_weights to False in xmf_load or delete the dir with  ##
## weights in target directory (preferred for speed).              ##
#####################################################################
grd          = grd_load( path2cs = path, resolve = 2. )
csLoader     = cbs_load( path2cs = path )
gmtPlotter   = gmt_load( grid_LL = grd.LL )
xmfRegridder = xmf_load( grid_CS = grd.CS, grid_LL = grd.LL, path2cs = path )

#####################################################################
## Load, rotate and mask.                                          ##
#####################################################################
Eta, W, U, V = csLoader.load2( time    = itime,
                               level   = irad,
                               grid_CS = grd.CS )

#####################################################################
## Regrid.                                                         ##
#####################################################################
Eta_LL = [ xmfRegridder.regrid( Eta ) ]
U_LL   = [ xmfRegridder.regrid( U ) * 100 ]
V_LL   = [ xmfRegridder.regrid( V ) * 100 ]
W_LL   = [ xmfRegridder.regrid( W ) * 100 ]

#####################################################################
## Plotting data.                                                  ##
#####################################################################
gmtPlotter.plot( U_LL, namefig='Azimuthal velocity, T=100' )
gmtPlotter.plot( V_LL, namefig='Meridional velocity, T=100' )
gmtPlotter.plot( W_LL, namefig='Vertical velocity, T=100' )
gmtPlotter.vplot( U_LL, V_LL, namefig='Horizontal speed, T=100' )
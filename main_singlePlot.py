from pylib.pth import path, ntiles
from pylib.cbs import cbs_load
from pylib.grl import grd_load
from pylib.xmf import xmf_load
from pylib.gmt import gmt_load

#####################################################################
## Time and radial points of interest.                             ##
#####################################################################
irad  = 15
itime = 0

#####################################################################
## Preparing grid information about the cubed-sphere grid and the  ##
## target lat-lon grid. Based on that, the GMT plotter and XESMF   ##
## regridder are prepared. Loader of cubed-sphere datasets is also ##
## prepared in advance. Upon changing the resolution, it is needed ##
## to set use_weights to False in xmf_load or delete the dir with  ##
## weights in target directory (preferred for speed).              ##
#####################################################################
grd          = grd_load( path2cs = path, ntiles=ntiles, resolve = 2. )
csLoader     = cbs_load( path2cs = path, ntiles=ntiles )
gmtPlotter   = gmt_load( grid_LL = grd.LL )
xmfRegridder = xmf_load( grid_CS = grd.CS, grid_LL = grd.LL, path2wg = path )

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
gmtPlotter.plot( Eta_LL, namefig=f'Eta' )
gmtPlotter.plot( U_LL, namefig=f'Azimuthal velocity' )
gmtPlotter.plot( V_LL, namefig=f'Meridional velocity' )
gmtPlotter.plot( W_LL, namefig=f'Vertical velocity' )
gmtPlotter.vplot( U_LL, V_LL, namefig=f'Horizontal speed' )
from pylib.pth import path
from pylib.cbs import cbs_load
from pylib.grl import grd_load
from pylib.xmf import xmf_load
from pylib.gmt import gmt_load
from pylib.ani import make_animation

######################################################################
### Number of time steps and radial point of interest.              ##
######################################################################
irad  = 15
ntime = 101

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
xmfRegridder = xmf_load( grid_CS = grd.CS, grid_LL = grd.LL, path = path )

#####################################################################
## Loading, masking and regridding data.                           ##
#####################################################################
Eta_LL = []
U_LL   = []
V_LL   = []
W_LL   = []

for itime in range(ntime):
    Eta, W, U, V = csLoader.load2( time    = itime,
                                   level   = irad,
                                   grid_CS = grd.CS )
    
    Eta_LL.append( xmfRegridder.regrid( Eta ) )
    U_LL.append( xmfRegridder.regrid( U ) * 100 )
    V_LL.append( xmfRegridder.regrid( V ) * 100 )
    W_LL.append( xmfRegridder.regrid( W ) * 100 )

#####################################################################
## Plotting data.                                                  ##
#####################################################################
gmtPlotter.vplot( U_LL, V_LL, namefig='Horizontal speed with lateral vectors' )

make_animation( 'Horizontal speed with lateral vectors ', 'movie-hvec.mp4' )
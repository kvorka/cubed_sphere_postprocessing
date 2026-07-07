
from pylib.cbs import cbs_load
from pylib.grd import grd_load
from pylib.xmf import xmf_load
from pylib.gmt import gmt_load

#####################################################################
## Time and radial points of interest.                             ##
#####################################################################
path  = 'state/ridge_20km_cs32x32x20/'
irad  = 15
itime = 100

#####################################################################
## Preparing grid information about the cubed-sphere grid and the  ##
## target lat-lon grid. Based on that, the GMT plotter and XESMF   ##
## regridder are prepared. Loader of cubed-sphere datasets is also ##
## prepared in advance.                                            ##
#####################################################################
grd          = grd_load( path2cs = path, resolve = 2. )
csLoader     = cbs_load( path2cs = path )
gmtPlotter   = gmt_load( grid_LL = grd.LL )
xmfRegridder = xmf_load( grid_CS = grd.CS, grid_LL = grd.LL, path = path, use_weights = False, method = 'conservative' )

#####################################################################
## Loading, masking and regridding data.                           ##
#####################################################################
data_Eta = csLoader.load( 'Eta', time=itime )
data_W   = csLoader.load( 'W', time=itime, level=irad )
data_U, \
data_V   = csLoader.rotate( csLoader.load( 'U', time=itime, level=irad ), 
                            csLoader.load( 'V', time=itime, level=irad ), grd.CS )

csLoader.mask( irad, grd.CS, data_W, data_U, data_V )

Eta_LL = xmfRegridder.regrid( data_Eta )    
U_LL   = xmfRegridder.regrid( data_U ) * 100
V_LL   = xmfRegridder.regrid( data_V ) * 100
W_LL   = xmfRegridder.regrid( data_W ) * 100

#####################################################################
## Plotting data.                                                  ##
#####################################################################
gmtPlotter.vplot( U_LL, V_LL )
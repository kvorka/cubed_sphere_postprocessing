import gc

from pylib.cs import build_CS_grid, load_CS_data, rotate_CS_data, mask_CS_data
from pylib.ll import build_LL_grid
from pylib.xe import build_regridder, regrid
from pylib.gm import gmt_load
from pylib.an import make_animation

######################################################################
### Time and radial points of interest.                             ##
######################################################################
path  = 'state/ridge_20km_cs32x32x20/'
#path  = 'state/flat_cs32x32x20/'
irad  = 15
ntime = 101

#####################################################################
## Resolution of the regrid and whether use weights. Should be set ##
## to False if you changed the resolution or are going to use a    ##
## different method for interpolation.                             ##
#####################################################################
resolution  = 2.
use_weights = True
method      = 'conservative'

#####################################################################
## Preparing grid information.                                     ##
#####################################################################
grid_LL = build_LL_grid( resolution )
grid_CS = build_CS_grid( path )

#####################################################################
## Preparing plotting for the defined output grid.                 ##
#####################################################################
gmtPlotter = gmt_load( grid_LL )

#####################################################################
## Building regridder.                                             ##
#####################################################################
regridder = build_regridder( grid_CS, grid_LL, 
                             path, use_weights, method )

#####################################################################
## Loading, masking and regridding data.                           ##
#####################################################################
Eta_LL = []
U_LL   = []
V_LL   = []
W_LL   = []

for itime in range(ntime+1):
    data_Eta = load_CS_data( path, 'Eta', time=itime )
    data_W   = load_CS_data( path, 'W', time=itime, level=irad )
    data_U, \
    data_V   = rotate_CS_data( load_CS_data( path, 'U', time=itime, level=irad ), 
                               load_CS_data( path, 'V', time=itime, level=irad ), grid_CS )
    
    mask_CS_data( irad, grid_CS, data_W, data_U, data_V )
    
    Eta_LL.append( regrid( regridder, data_Eta, grid_LL ) )
    U_LL.append( 100 * regrid( regridder, data_U, grid_LL ) )
    V_LL.append( 100 * regrid( regridder, data_V, grid_LL ) )
    W_LL.append( 100 * regrid( regridder, data_W, grid_LL ) )
    
    gc.collect()

#####################################################################
## Plotting data.                                                  ##
#####################################################################
gmtPlotter.plot( U_LL, namefig='Zonal velocity' )

make_animation( 'Zonal velocity ', 'movie-zon.mp4' )
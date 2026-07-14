import warnings
import numpy
warnings.filterwarnings( 'ignore', message='.*ESMF and ESMPy.*' ); import xesmf

from pylib.pth import ntiles, path
from pylib.cbs import cbs_load
from pylib.grl import grd_load

#####################################################################################################
## The new grid dimensions per one tile.                                                           ##
#####################################################################################################
nx = 64
nz = 20

#####################################################################################################
## Loading the cubed-sphere longitudes and latitudes from .mitgrid files of the target grid and    ##
## the grid used in the simulation.                                                                ##
#####################################################################################################
grd1 = grd_load( path2bin = '' )
grd0 = grd_load( path2cs = path, ntiles = ntiles )

#####################################################################################################
## Loading the velocities on the cubed-sphere grid from the simulation output.                     ##
#####################################################################################################
csLoader = cbs_load( path2cs = path, ntiles=ntiles )
Eta      = csLoader.load( 'Eta', time=-1 )
U, V     = csLoader.load3()

#####################################################################################################
## This is rather wild so let us make it short. We want to interpolate from coarser to finer grid. ##
## The cubed-sphere setting is making it cumbersome to interpolate on staggered grid. However, the ##
## initialization files require arrays of shape (6*nn,nn,nz). By some coinsidence, this is exactly ##
## the shape of arrays in the middle of the cells. We will therefore interpolate from middle to    ##
## middle and print those values instead of values on the faces of the shells.                     ##
#####################################################################################################
Unew   = numpy.zeros( (6,nx,nx,nz) )
Vnew   = numpy.zeros( (6,nx,nx,nz) )
Etanew = numpy.zeros( (6,nx,nx)    )

for itile1 in range(6):
    for itile0 in range(ntiles):
        regridder = xesmf.Regridder( ds_in  = grd0.CS[itile0],
                                     ds_out = grd1.CS[itile1],
                                     method = 'conservative',
                                     unmapped_to_nan = True,
                                     reuse_weights = False )
        
        dataEta                     = regridder( Eta[itile0][:,:], skipna=True )
        maskEta                     = ~numpy.isnan( dataEta )
        Etanew[itile0,:,:][maskEta] = dataEta[maskEta]
        
        for iz in range(nz):
            dataU = regridder( U[itile0][iz,:,:], skipna=True )
            dataV = regridder( V[itile0][iz,:,:], skipna=True )

            maskU = ~numpy.isnan( dataU )
            maskV = ~numpy.isnan( dataV )
            
            Unew[itile1,:,:,iz][maskU] = dataU[maskU]
            Vnew[itile1,:,:,iz][maskV] = dataV[maskV]

Eta_mitgcm = numpy.concatenate( [ Etanew[i] for i in range(6) ], axis=0 )
U_mitgcm   = numpy.concatenate( [ Unew[i]   for i in range(6) ], axis=0 )
V_mitgcm   = numpy.concatenate( [ Vnew[i]   for i in range(6) ], axis=0 )

#####################################################################################################
## Subroutine for saving and reading the binary files.                                             ##
#####################################################################################################
def write_bin(filename, array):
    array.astype( dtype='>f8' ).ravel( order='F' ).tofile( filename )

write_bin( 'Eta_init_cs.bin', Eta_mitgcm )
write_bin( 'U_init_cs.bin',   U_mitgcm   )
write_bin( 'V_init_cs.bin',   V_mitgcm   )
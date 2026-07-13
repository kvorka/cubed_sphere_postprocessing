import warnings
warnings.filterwarnings( 'ignore', message='.*ESMF and ESMPy.*' )

import os
import gc
import numpy
import xesmf

class xmf_load:
    def __init__(self, grid_CS, grid_LL, path2wg, use_weights=True, method='conservative'):
        weights_dir = f'{path2wg}xesmfweights'
        self.ntiles = len( grid_CS )
        
        os.makedirs( weights_dir, exist_ok=True )
        reuse_weights = use_weights and all( os.path.exists( f'{weights_dir}/wght.t{i+1:03d}.nc' ) for i in range(self.ntiles) )
        
        self.regridders = []
        self.data_shape = ( grid_LL['lat'].size, 
                            grid_LL['lon'].size )
        
        for i in range(self.ntiles):
            self.regridders.append( xesmf.Regridder( ds_in  = grid_CS[i],
                                                     ds_out = grid_LL,
                                                     method = method,
                                                     unmapped_to_nan = True,
                                                     filename = f'{weights_dir}/wght.t{i+1:03d}.nc',
                                                     reuse_weights = reuse_weights ) )
    
    def regrid(self, data_CS):
        data_out = numpy.full( self.data_shape, numpy.nan, dtype=numpy.float32 )
        
        for i in range(self.ntiles):
            data = self.regridders[i]( data_CS[i], skipna=True )
            mask = ~numpy.isnan( data )
            data_out[mask] = data[mask]
        
        gc.collect(); return data_out

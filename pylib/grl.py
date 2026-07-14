import numpy
import netCDF4

class grd_load:
    def __init__(self, path2cs, ntiles, resolve):
        self.load_CS_grid( path2cs, ntiles )
        self.build_LL_grid( resolve )
    
    def load_CS_grid(self, path2cs, ntiles):
        self.CS = []
        
        for i in range(ntiles):
            grid = netCDF4.Dataset( f'{path2cs}grid.t{i+1:03d}.nc', 'r' )
            
            self.CS.append( { 'lon'     : numpy.ma.filled( grid['XC'][:,:], numpy.nan ).astype( numpy.float32 ),
                              'lat'     : numpy.ma.filled( grid['YC'][:,:], numpy.nan ).astype( numpy.float32 ),
                              'lon_b'   : numpy.ma.filled( grid['XG'][:,:], numpy.nan ).astype( numpy.float32 ),
                              'lat_b'   : numpy.ma.filled( grid['YG'][:,:], numpy.nan ).astype( numpy.float32 ),
                              'angleCS' : numpy.ma.filled( grid['AngleCS'][:,:], numpy.nan ).astype( numpy.float32 ),
                              'angleSN' : numpy.ma.filled( grid['AngleSN'][:,:], numpy.nan ).astype( numpy.float32 ),
                              'hfac'    : numpy.ma.filled( grid['HFacC'][:,:], numpy.nan ).astype( numpy.float32 ) } )
    
    def build_LL_grid(self, res):
        lon_b = numpy.arange(-180.0, 180.0+res, res).astype( numpy.float32 )
        lat_b = numpy.arange(-90.0,   90.0+res, res).astype( numpy.float32 )
        
        lon = ( lon_b[:-1] + lon_b[1:] ) / 2
        lat = ( lat_b[:-1] + lat_b[1:] ) / 2
        
        lon2d, lat2d = numpy.meshgrid( lon, lat )
        
        self.LL = { 'lon'   : lon,
                    'lat'   : lat,
                    'lon_b' : lon_b,
                    'lat_b' : lat_b,
                    'lon2d' : lon2d,
                    'lat2d' : lat2d }
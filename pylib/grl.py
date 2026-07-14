import numpy
import netCDF4

class grd_load:
    def __init__(self, path2cs=None, ntiles=None, path2bin=None, resolve=None):
        if path2cs is not None:
            self.load_CS_grid( path2cs, ntiles )
        elif path2bin is not None:
            self.load_CS_grid2( path2bin )
        
        if resolve is not None:
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
    
    def load_CS_grid2(self, path2bin):
        self.CS = []
        
        tmp = numpy.fromfile( f"tile001.mitgrid", dtype=">f8" )
        nn  = int( numpy.sqrt( len(tmp) / 16 ) ) - 1
        tmp = numpy.reshape( tmp, (nn+1,nn+1,16), order='F' )
        
        self.CS.append( { 'lon'     : tmp[:nn,:nn,0].astype( numpy.float32 ),
                          'lat'     : tmp[:nn,:nn,1].astype( numpy.float32 ),
                          'lon_b'   : tmp[:  ,:  ,5].astype( numpy.float32 ),
                          'lat_b'   : tmp[:  ,:  ,6].astype( numpy.float32 ) } )
        
        for itile in range(1,6):
            tmp = numpy.reshape( numpy.fromfile( f"tile00{itile+1}.mitgrid", dtype=">f8" ), (nn+1,nn+1,16), order='F' )
            
            self.CS.append( { 'lon'     : tmp[:nn,:nn,0].astype( numpy.float32 ),
                              'lat'     : tmp[:nn,:nn,1].astype( numpy.float32 ),
                              'lon_b'   : tmp[:  ,:  ,5].astype( numpy.float32 ),
                              'lat_b'   : tmp[:  ,:  ,6].astype( numpy.float32 ) } )
    
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
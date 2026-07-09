import gc
import numpy
import netCDF4

class cbs_load:
    def __init__(self, path2cs, load_monitor=False, load_data=True):
        if load_data:
            self.tiles = []
            
            for i in range(6):
                self.tiles.append( netCDF4.Dataset(path2cs+'state.0000000000.t00'+str(i+1)+'.nc', 'r') )
        
        if load_monitor:
            self.monitor = netCDF4.Dataset(path2cs+'monitor.0000000000.t001.nc', 'r')
        
    def check_shapes(self, *args):
        for var in args:
            *leading, nlat, nlon = self.tiles[0][var].shape
            
            print( f'Variable: {var}' )
            print( f'Time levels: {leading[0]}' )
            
            if len( leading ) == 2:
                print( f'Radial levels: {leading[1]}' )
            
            print(f'Single tile lats: {nlat}')
            print(f'Single tile lons: {nlon}')
            print()
    
    def check_cfl(self):
        cflU  = numpy.max( numpy.ma.filled( self.monitor['advcfl_uvel_max'] ) )
        cflV  = numpy.max( numpy.ma.filled( self.monitor['advcfl_vvel_max'] ) )
        cflW  = numpy.max( numpy.ma.filled( self.monitor['advcfl_wvel_max'] ) )
        cflWb = numpy.max( numpy.ma.filled( self.monitor['advcfl_W_hf_max'] ) )
        
        print( f'Azimuthal cfl: {cflU}' )
        print( f'Meridional cfl: {cflV}' )
        print( f'Vertical cfl: {cflW}' )
        print( f'Vertical cfl with bathymetry: {cflWb}' )
    
    def load(self, var, time, level=None):
        data_CS = []
        
        for i in range(6):
            if level is None:
                arr = numpy.ma.filled( self.tiles[i][var][time,:,:], numpy.nan ).astype( numpy.float32 )
            else:
                arr = numpy.ma.filled( self.tiles[i][var][time,level,:,:], numpy.nan ).astype( numpy.float32 )
            
            data_CS.append( arr )
        
        gc.collect(); return data_CS
    
    def rotate(self, u_CS, v_CS, grid_CS):
        u_East  = []
        v_North = []
        
        for i in range(6):
            u_C = ( u_CS[i][:,:-1] + u_CS[i][:,1:] ) / 2
            v_C = ( v_CS[i][:-1,:] + v_CS[i][1:,:] ) / 2
            
            u_East.append( grid_CS[i]['angleCS'] * u_C - grid_CS[i]['angleSN'] * v_C )
            v_North.append( grid_CS[i]['angleSN'] * u_C + grid_CS[i]['angleCS'] * v_C )
        
        gc.collect(); return u_East, v_North
    
    def mask(self, level, grid_CS, *args):
        masks = [ ( grid['hfac'][level] == 0 ) for grid in grid_CS ]
        
        for i, mask in enumerate( masks ):
            for f in args:
                f[i][mask] = numpy.nan
    
    def load2(self, time, level, grid_CS):
        data_Eta = self.load( 'Eta', time=time )
        data_W   = self.load( 'W', time=time, level=level )
        data_U, \
        data_V   = self.rotate( self.load( 'U', time=time, level=level ), 
                                self.load( 'V', time=time, level=level ), grid_CS )
        
        self.mask( level, grid_CS, data_W, data_U, data_V )
        
        return data_Eta, data_W, data_U, data_V
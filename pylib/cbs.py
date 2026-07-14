import numpy
import netCDF4
import matplotlib.pyplot

class cbs_load:
    def __init__(self, path2cs, ntiles=None, load_monitor=False, load_data=True, name=None):
        self.name         = name
        self.load_data    = load_data
        self.load_monitor = load_monitor
        
        if self.load_data:
            self.tiles  = []
            self.ntiles = ntiles
            
            for i in range(self.ntiles):
                self.tiles.append( netCDF4.Dataset( f'{path2cs}state.0000000000.t{i+1:03d}.nc', 'r' ) )
        
        if self.load_monitor:
            self.monitor = netCDF4.Dataset(path2cs+'monitor.0000000000.t001.nc', 'r')
        
    def check_data(self, *args):
        if self.load_data:
            for var in args:
                *leading, nlat, nlon = self.tiles[0][var].shape
                
                print( f'Variable: {var}' )
                print( f'Time levels: {leading[0]}' )
                
                if len( leading ) == 2:
                    print( f'Radial levels: {leading[1]}' )
                
                print(f'Single tile lats: {nlat}')
                print(f'Single tile lons: {nlon}')
                print()
        
        if self.load_monitor:
            KE    = numpy.max( numpy.ma.filled( self.monitor['ke_mean'] ) )
            cflU  = numpy.max( numpy.ma.filled( self.monitor['advcfl_uvel_max'] ) )
            cflV  = numpy.max( numpy.ma.filled( self.monitor['advcfl_vvel_max'] ) )
            cflW  = numpy.max( numpy.ma.filled( self.monitor['advcfl_wvel_max'] ) )
            cflWb = numpy.max( numpy.ma.filled( self.monitor['advcfl_W_hf_max'] ) )
            
            print( f'Mean kinetic energy: {KE}' )
            print( f'Azimuthal cfl: {cflU}' )
            print( f'Meridional cfl: {cflV}' )
            print( f'Vertical cfl: {cflW}' )
            print( f'Vertical cfl with bathymetry: {cflWb}' )
            
            time = numpy.ma.filled( self.monitor['T'] )
            KE   = numpy.ma.filled( self.monitor['ke_mean'] )
            
            matplotlib.pyplot.plot( time, KE, linestyle='-' )
            
            matplotlib.pyplot.xlabel( 'Time [s]' )
            matplotlib.pyplot.ylabel( 'KE' )
            matplotlib.pyplot.grid( True )
            matplotlib.pyplot.show()
    
    def load(self, var, time, level=None):
        data_CS = []
        
        for i in range(self.ntiles):
            if level is None:
                arr = numpy.ma.filled( self.tiles[i][var][time,:,:], numpy.nan ).astype( numpy.float32 )
            else:
                arr = numpy.ma.filled( self.tiles[i][var][time,level,:,:], numpy.nan ).astype( numpy.float32 )
            
            data_CS.append( arr )
        
        return data_CS
    
    def rotate(self, u_CS, v_CS, grid_CS):
        u_East  = []
        v_North = []
        
        for i in range(self.ntiles):
            u_C = ( u_CS[i][:,:-1] + u_CS[i][:,1:] ) / 2
            v_C = ( v_CS[i][:-1,:] + v_CS[i][1:,:] ) / 2
            
            u_East.append( grid_CS[i]['angleCS'] * u_C - grid_CS[i]['angleSN'] * v_C )
            v_North.append( grid_CS[i]['angleSN'] * u_C + grid_CS[i]['angleCS'] * v_C )
        
        return u_East, v_North
    
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
    
    def get_KE_series(self, id='ke_mean'):
        time = numpy.ma.filled( self.monitor['T'] )
        KE   = numpy.ma.filled( self.monitor[id]  )
        
        return time, KE
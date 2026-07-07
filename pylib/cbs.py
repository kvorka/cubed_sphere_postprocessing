import gc
import numpy
import netCDF4

class cbs_load:
    def __init__(self, path2cs):
        self.tiles = []
        
        for i in range(6):
            self.tiles.append( netCDF4.Dataset(path2cs+'state.0000000000.t00'+str(i+1)+'.nc', 'r') )

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
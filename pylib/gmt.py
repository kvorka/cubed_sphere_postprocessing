import gc
import numpy
import pygmt

class gmt_load:
    def __init__(self, grid_LL):
        self.projW     = 'W0/12'
        self.frameW    = ['WSne', 'g30', 'ya30']
        self.cpallete  = 'vik'
        self.cpallete2 = 'nuuk'
        self.vstride   = 8
        
        pygmt.config( MAP_FRAME_PEN = '1.0p,black',
                      MAP_GRID_PEN  = '0.5p,gray60' )
        
        x, y = numpy.meshgrid( grid_LL['lon'],
                               grid_LL['lat'], indexing='ij' )
        
        self.region = [ x.min(), x.max(),
                        y.min(), y.max() ]
        
        self.spacing = [ numpy.abs( x[1,0] - x[0,0] ),
                         numpy.abs( y[0,1] - y[0,0] ) ]
        
        self.x = x.ravel()
        self.y = y.ravel()
        
        self.xv = x[::self.vstride,::self.vstride].ravel()
        self.yv = y[::self.vstride,::self.vstride].ravel()
        
        gc.collect()
    
    def prepare_rng(self, data1, data2=None):
        if data2 is None:
            return numpy.max( [ numpy.nanmax( numpy.abs( data1[t] ) ) for t in range( len(data1) ) ] )
        else:
            return numpy.max( [ numpy.nanmax( numpy.sqrt( data1[t]**2+data2[t]**2 ) ) for t in range( len(data1) ) ] )
    
    def prepare_data(self, data):
        return numpy.ravel( numpy.nan_to_num( numpy.transpose( data ), nan=0 ) )
    
    def prepare_cpt(self, maxmin, onesided):
        pygmt.makecpt( cmap       = self.cpallete if not onesided else self.cpallete2,
                       background = True,
                       series     = ( [ -maxmin, +maxmin ] ) if not onesided else ( [ 0, maxmin ] ) )
    
    def prepare_grid(self, zzz, rng, onesidedCpt=False):
        self.prepare_cpt( rng, onesidedCpt )
        
        return pygmt.xyz2grd( x = self.x,
                              y = self.y,
                              z = self.prepare_data( zzz ),
                              spacing = self.spacing,
                              region  = self.region )
    
    def grid_image(self, fig, grid, fout=None):
        fig.colorbar( position = 'JMR+w6c/0.4c+v+o-2c/-4.5c', 
                      frame    = ['a', '+lcm/s'] )
        
        fig.grdimage( region = self.region,
                      projection = self.projW,
                      frame = ( self.frameW if fout is None else [ f'{self.frameW[0]}+t{fout}', *self.frameW[1:] ] ),
                      grid = grid )
    
    def quiv_image(self, fig, speed, angle):
        fig.plot( x = self.xv,
                  y = self.yv,
                  direction = [ self.prepare_data( angle[::self.vstride,::self.vstride] ), 
                                self.prepare_data( speed[::self.vstride,::self.vstride] ) / 4 ],
                  style = 'v0.20c+e+a25',
                  pen   = '0.75p,black' )
    
    def name_fig(self, fout, series, index):
        if fout is not None and series:
            return f'{fout} T={index}'
        #elif fout is not None:
        #    return f'{fout}'
        else:
            return None
    
    def save_fig(self, fig, fout, series, index):
        if fout is None: 
            fig.show( dpi=1000 )
        elif not series:
            fig.savefig( f'{fout}.png', dpi=1000 )
        else:
            fig.savefig( f'{fout} {i}.png', dpi=1000 )
    
    def plot(self, data, namefig=None):
        series = ( False if len( data ) == 1 else True )
        
        valmax = self.prepare_rng( data1 = data )
        
        for i in range( len(data) ):
            fig = pygmt.Figure()
            
            grid = self.prepare_grid( zzz = data, 
                                      rng = valmax )
            
            self.grid_image( fig  = fig, 
                             grid = grid, 
                             fout = self.name_fig( fout   = namefig, 
                                                   series = series, 
                                                   index  = i ) )
            
            self.save_fig( fig    = fig,
                           fout   = namefig,
                           series = series,
                           index  = i )
            
            gc.collect()
    
    def vplot(self, dataU, dataV, namefig=None):
        series = ( False if len( dataU ) == 1 else True )
        
        valmax = self.prepare_rng( data1 = dataU, 
                                   data2 = dataV )
        
        for i in range( len(dataU) ):
            angle = numpy.degrees( numpy.arctan2( dataV[i], dataU[i] ) )
            speed = numpy.sqrt( dataU[i]**2 + dataV[i]**2 )
            
            fig = pygmt.Figure()
            
            grid = self.prepare_grid( zzz = speed, 
                                      rng = valmax, 
                                      onesidedCpt = True )
            
            self.grid_image( fig  = fig, 
                             grid = grid, 
                             fout = self.name_fig( fout   = namefig, 
                                                   series = series, 
                                                   index  = i ) )
            
            self.quiv_image( fig   = fig, 
                             speed = speed, 
                             angle = angle )
            
            self.save_fig( fig    = fig,
                           fout   = namefig,
                           series = series,
                           index  = i )
            
            gc.collect()
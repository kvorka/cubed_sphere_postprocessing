import gc
import numpy
import pygmt

class gmt_load:
    def __init__(self, grid_LL):
        self.projW     = 'W180/12'
        self.frameW    = ['WSne', 'g30', 'ya30']
        self.cpallete  = 'vik'
        self.cpallete2 = 'nuuk'
        self.vstride   = 5
        
        pygmt.config( MAP_FRAME_PEN = '1.0p,black',
                      MAP_GRID_PEN  = '0.5p,gray60' )
        
        x, y = numpy.meshgrid( grid_LL['lon']+180,
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
    
    def prepare_data(self, data):
        return numpy.ravel( numpy.nan_to_num( numpy.transpose( data ), nan=0 ) )
    
    def prepare_cpt(self, maxmin, onesided):
        pygmt.makecpt( cmap       = self.cpallete if not onesided else self.cpallete2,
                       background = True,
                       series     = ( [ -maxmin, +maxmin ] ) if not onesided else ( [ 0, maxmin ] ) )
    
    def prepare_grid(self, z, maxmin, onesidedCpt=False):
        self.prepare_cpt( maxmin, onesidedCpt )
        
        return pygmt.xyz2grd( x = self.x,
                              y = self.y,
                              z = z,
                              spacing = self.spacing,
                              region  = self.region )
    
    def grid_image(self, fig, grid, namefig=None):
        fig.colorbar( position = 'JMR+w6c/0.4c+v+o-2c/-4.5c', 
                      frame    = ['a', '+lcm/s'] )
        
        fig.grdimage( region = self.region,
                      projection = self.projW,
                      frame = ( self.frameW if namefig is None else [ f'{self.frameW[0]}+t{namefig}', *self.frameW[1:] ] ),
                      grid = grid )
    
    def plot_single(self, data, maxmin, namefig=None, show=True):
        fig = pygmt.Figure()
        
        grid = self.prepare_grid( z = self.prepare_data(data), maxmin = maxmin )
        
        self.grid_image( fig, grid, namefig=namefig )
        
        if show: 
            fig.show()
        
        if namefig is not None:
            fig.savefig(f"{namefig}.png")
        
        gc.collect()
    
    def vplot_single(self, data1, data2, maxmin, namefig=None, show=True):
        fig = pygmt.Figure()
        
        grid = self.prepare_grid( z = self.prepare_data(data1), maxmin = maxmin, onesidedCpt=True )
        
        self.grid_image( fig, grid, namefig=namefig )
        
        fig.plot( x = self.xv,
                  y = self.yv,
                  direction = [ self.prepare_data(data2[::self.vstride,::self.vstride]), 
                                self.prepare_data(data1[::self.vstride,::self.vstride]) / 4 ],
                  style="v0.20c+e+a25",
                  pen="0.75p,black" )
        
        fig.show()
        
        gc.collect()
    
    def plot(self, data, namefig=None):
        single = not isinstance(data, list)
        
        if single:
            data = [data]
        
        maxmin = numpy.max( [ numpy.nanmax( numpy.abs( data[t] ) ) for t in range( len(data) ) ] )
        
        for i, field in enumerate(data):
            self.plot_single( data    = field,
                              maxmin  = maxmin,
                              namefig = None if namefig is None else ( namefig if single else f"{namefig} {i}" ),
                              show    = single )
    
    def vplot(self, dataU, dataV, namefig=None):
        angle = numpy.degrees( numpy.arctan2( dataV, dataU ) )
        speed = numpy.sqrt( dataU**2 + dataV**2 )
        
        maxmin = numpy.nanmax( speed )
        
        self.vplot_single( speed, angle, maxmin )
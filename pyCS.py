import warnings; warnings.filterwarnings( 'ignore', message='.*ESMF and ESMPy.*' )

import os
import sys
import gc
import netCDF4
import pygmt
import numpy
import xesmf

def build_CS_grid(path):
    grid_CS  = []
    
    for i in range(6):
        grid = netCDF4.Dataset( f'{path}grid.t00{i+1}.nc', 'r' )
        
        grid_CS.append( 
            { 
                'lon'     : numpy.ma.filled( grid['XC'][:,:], numpy.nan ).astype( numpy.float32 ),
                'lat'     : numpy.ma.filled( grid['YC'][:,:], numpy.nan ).astype( numpy.float32 ),
                'lon_b'   : numpy.ma.filled( grid['XG'][:,:], numpy.nan ).astype( numpy.float32 ),
                'lat_b'   : numpy.ma.filled( grid['YG'][:,:], numpy.nan ).astype( numpy.float32 ),
                'angleCS' : numpy.ma.filled( grid['AngleCS'][:,:], numpy.nan ).astype( numpy.float32 ),
                'angleSN' : numpy.ma.filled( grid['AngleSN'][:,:], numpy.nan ).astype( numpy.float32 ),
                'hfac'    : numpy.ma.filled( grid['HFacC'][:,:], numpy.nan ).astype( numpy.float32 ) 
            } 
        )
    
    gc.collect(); return grid_CS

def load_CS_data(path, var, time, level=None):
    data_CS = []
    
    for i in range(6):
        with netCDF4.Dataset(path+'state.0000000000.t00'+str(i+1)+'.nc', 'r') as ds:
            if level is None:
                arr = numpy.ma.filled( ds[var][time,:,:], numpy.nan ).astype( numpy.float32 )
            else:
                arr = numpy.ma.filled( ds[var][time,level,:,:], numpy.nan ).astype( numpy.float32 )
        
        data_CS.append( arr )
    
    gc.collect(); return data_CS

def rotate_CS_data(u_CS, v_CS, grid_CS):
    u_East  = []
    v_North = []
    
    for i in range(6):
        u_C = ( u_CS[i][:,:-1] + u_CS[i][:,1:] ) / 2
        v_C = ( v_CS[i][:-1,:] + v_CS[i][1:,:] ) / 2
        
        u_East.append( grid_CS[i]['angleCS'] * u_C - grid_CS[i]['angleSN'] * v_C )
        v_North.append( grid_CS[i]['angleSN'] * u_C + grid_CS[i]['angleCS'] * v_C )
    
    gc.collect(); return u_East, v_North

def mask_CS_data(level, grid_CS, *args):
    masks = [ ( grid['hfac'][level] == 0 ) for grid in grid_CS ]
    
    for i, mask in enumerate( masks ):
        for f in args:
            f[i][mask] = numpy.nan

def build_LL_grid(res):
    lon_b = numpy.arange(-180.0, 180.0+res, res).astype( numpy.float32 )
    lat_b = numpy.arange(-90.0,   90.0+res, res).astype( numpy.float32 )
    
    lon = ( lon_b[:-1] + lon_b[1:] ) / 2
    lat = ( lat_b[:-1] + lat_b[1:] ) / 2
    
    lon2d, lat2d = numpy.meshgrid( lon, lat )
    
    gc.collect(); return {  'lon'   : lon,
                            'lat'   : lat,
                            'lon_b' : lon_b,
                            'lat_b' : lat_b,
                            'lon2d' : lon2d,
                            'lat2d' : lat2d }

def build_regridder(grid_CS, grid_LL, path, use_weights, method):
    regridders  = []
    weights_dir = f'{path}xesmfweights'
    
    os.makedirs( weights_dir, exist_ok=True )
    reuse_weights = use_weights and all( os.path.exists(f'{weights_dir}/wght.t00{i+1}.nc') for i in range(6) )
    
    for i in range(6):
        regridders.append( 
            xesmf.Regridder( 
                ds_in=grid_CS[i],
                ds_out=grid_LL,
                method=method,
                unmapped_to_nan=True,
                filename=f'{weights_dir}/wght.t00{i+1}.nc',
                reuse_weights=reuse_weights 
            ) 
        )
    
    return regridders

def regrid(regridder, data_CS, grid_LL):
    data_out = numpy.full( ( grid_LL['lat'].size, 
                          grid_LL['lon'].size), numpy.nan, dtype=numpy.float32 )
    
    for i in range(6):
        data = regridder[i]( data_CS[i], skipna=True )
        mask = ~numpy.isnan( data )
        data_out[mask] = data[mask]
    
    gc.collect(); return data_out

def data_to_fig(x, y, z, proj, frame, outf, cpallete, saturation=None):
    fig = pygmt.Figure()
    
    pygmt.config( MAP_FRAME_PEN='1.0p,black' )
    pygmt.config( MAP_GRID_PEN='0.5p,gray60')

    if (saturation is not None):
        pygmt.makecpt( cmap = cpallete, 
                    series = [-saturation * numpy.max( numpy.abs(z) ), saturation * numpy.max( numpy.abs(z) ) ],
                    background = True )
    else:
        pygmt.makecpt( cmap = cpallete, 
                    series = [-numpy.max( numpy.abs(z) ), +numpy.max( numpy.abs(z) )],
                    background = True )
    
    region = [ x.min(), x.max(),
               y.min(), y.max() ]
    
    spacing = [ numpy.abs( x[1,0] - x[0,0] ),
                numpy.abs( y[0,1] - y[0,0] ) ]
    
    grid = pygmt.xyz2grd( x = numpy.ravel(x),
                       y = numpy.ravel(y),
                       z = numpy.ravel(z),
                       spacing = spacing,
                       region = region )
    
    fig.grdimage( region = region,
                  projection = proj,
                  frame = frame,
                  grid = grid )
    
    fig.show()

def gmt_LL(grid_LL, data_LL):
    x, y = numpy.meshgrid( grid_LL['lon']+180,
                        grid_LL['lat'], indexing='ij')
    
    z = numpy.nan_to_num( numpy.transpose(data_LL), nan=0 )
    
    data_to_fig( x=x,
                 y=y,
                 z=z,
                 proj='W180/12',
                 frame=['WSne', 'g30', 'ya30'],
                 outf='gmtplot.pdf',
                 cpallete='vik' )

if __name__ == '__main__':
    #####################################################################
    ## Time and radial points of interest.                             ##
    #####################################################################
    path  = 'state/ridge_20km_cs32x32x20/'
    irad  = 15
    itime = 100
    
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
    grid_LL  = build_LL_grid( resolution )
    grid_CS  = build_CS_grid( path )
    
    #####################################################################
    ## Building regridder.                                             ##
    #####################################################################
    regridder = build_regridder( grid_CS, grid_LL, 
                                 path, use_weights, method )
    
    #####################################################################
    ## Loading and masking data.                                       ##
    #####################################################################
    data_Eta = load_CS_data( path, 'Eta', time=itime )
    data_W   = load_CS_data( path, 'W', time=itime, level=irad )
    data_U, \
    data_V   = rotate_CS_data( load_CS_data( path, 'U', time=itime, level=irad ), 
                               load_CS_data( path, 'V', time=itime, level=irad ), grid_CS )
    
    mask_CS_data( irad, grid_CS, data_W, data_U, data_V )
    
    #####################################################################
    ## Regridding data.                                                ##
    #####################################################################
    Eta_LL = regrid( regridder, data_Eta, grid_LL )
    U_LL   = 100 * regrid( regridder, data_U, grid_LL )
    V_LL   = 100 * regrid( regridder, data_V, grid_LL )
    W_LL   = 100 * regrid( regridder, data_W, grid_LL )
    
    #####################################################################
    ## Plotting data.                                                  ##
    #####################################################################
    gmt_LL( grid_LL, U_LL )

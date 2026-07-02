import sys
import gc
import os
import netCDF4
import pygmt as pg
import numpy as np
import xesmf as xe
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmcrameri.cm as cmc
from cartopy.mpl.ticker import LatitudeFormatter
from scipy.interpolate import griddata

def build_CS_grid(path):
    grid_CS  = []
    
    for i in range(6):
        grid = netCDF4.Dataset( f'{path}grid.t00{i+1}.nc', 'r' )
        
        grid_CS.append( 
            { 
                'lon'     : np.ma.filled( grid['XC'][:,:], np.nan ).astype( np.float32 ),
                'lat'     : np.ma.filled( grid['YC'][:,:], np.nan ).astype( np.float32 ),
                'lon_b'   : np.ma.filled( grid['XG'][:,:], np.nan ).astype( np.float32 ),
                'lat_b'   : np.ma.filled( grid['YG'][:,:], np.nan ).astype( np.float32 ),
                'angleCS' : np.ma.filled( grid['AngleCS'][:,:], np.nan ).astype( np.float32 ),
                'angleSN' : np.ma.filled( grid['AngleSN'][:,:], np.nan ).astype( np.float32 ),
                'hfac'    : np.ma.filled( grid['HFacC'][:,:], np.nan ).astype( np.float32 ) 
            } 
        )
    
    gc.collect(); return grid_CS

def load_CS_data(path, var, time, level=None):
    data_CS = []
    
    for i in range(6):
        with netCDF4.Dataset(path+'state.0000000000.t00'+str(i+1)+'.nc', 'r') as ds:
            if level is None:
                arr = np.ma.filled( ds[var][time,:,:], np.nan ).astype( np.float32 )
            else:
                arr = np.ma.filled( ds[var][time,level,:,:], np.nan ).astype( np.float32 )
        
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
            f[i][mask] = np.nan

def build_LL_grid(res):
    lon_b = np.arange(-180.0, 180.0+res, res).astype( np.float32 )
    lat_b = np.arange(-90.0,   90.0+res, res).astype( np.float32 )
    
    lon = ( lon_b[:-1] + lon_b[1:] ) / 2
    lat = ( lat_b[:-1] + lat_b[1:] ) / 2
    
    lon2d, lat2d = np.meshgrid( lon, lat )
    
    gc.collect(); return {  'lon'   : lon,
                            'lat'   : lat,
                            'lon_b' : lon_b,
                            'lat_b' : lat_b,
                            'lon2d' : lon2d,
                            'lat2d' : lat2d }

def build_regridder(grid_CS, grid_LL, path, use_weights, method):
    regridders  = []
    weights_dir = f'{path}xeweights'
    
    os.makedirs( weights_dir, exist_ok=True )
    reuse_weights = use_weights and all( os.path.exists(f'{weights_dir}/wght.t00{i+1}.nc') for i in range(6) )
    
    for i in range(6):
        regridders.append( 
            xe.Regridder( 
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
    data_out = np.full( ( grid_LL['lat'].size, 
                          grid_LL['lon'].size), np.nan, dtype=np.float32 )
    
    for i in range(6):
        data = regridder[i]( data_CS[i], skipna=True )
        mask = ~np.isnan( data )
        data_out[mask] = data[mask]
    
    gc.collect(); return data_out

def export_cmap_to_cpt(cpallete, cfile, **kwargs):
    cmap = plt.get_cmap(cpallete, 255)
    
    b = np.array(kwargs.get('B', cmap(0.)))
    f = np.array(kwargs.get('F', cmap(1.)))
    
    na = np.array(kwargs.get('N', (0,0,0))).astype(float)
    
    ext = (np.c_[b[:3],f[:3],na[:3]].T*255).astype(int)
    extstr = 'B {:3d} {:3d} {:3d}\nF {:3d} {:3d} {:3d}\nN {:3d} {:3d} {:3d}'
    
    cols = (cmap(np.linspace(0.,1.,255))[:,:3]*255).astype(int)
    vals = np.linspace(-1,+1,255)
    
    np.savetxt( cfile, 
                np.c_[vals[:-1],cols[:-1],vals[1:],cols[1:]], 
                fmt      = '%e %3d %3d %3d %e %3d %3d %3d', 
                header   = '# COLOR_MODEL = RGB', 
                footer   = extstr.format(*list(ext.flatten())), 
                comments = '' )

def data_to_fig(x, y, z, proj, frame, cmap, outf, saturation=None, interval=None, cpallete='vik'):
    fig = pg.Figure()
    
    pg.config( MAP_FRAME_PEN='1.0p,black' )
    pg.config( MAP_GRID_PEN='0.5p,gray60')

    if (saturation is not None):
        pg.makecpt( cmap = cpallete, 
                    series = [-saturation * np.max( np.abs(z) ), saturation * np.max( np.abs(z) ) ],
                    background = True )
    else:
        pg.makecpt( cmap = cpallete, 
                    series = [-np.max( np.abs(z) ), +np.max( np.abs(z) )],
                    background = True )
    
    region = [ x.min(), x.max(),
               y.min(), y.max() ]
    
    spacing = [ np.abs( x[1,0] - x[0,0] ),
                np.abs( y[0,1] - y[0,0] ) ]
    
    grid = pg.xyz2grd( x = np.ravel(x),
                       y = np.ravel(y),
                       z = np.ravel(z),
                       spacing = spacing,
                       region = region )
    
    fig.grdimage( region = region,
                  projection = proj,
                  frame = frame,
                  grid = grid )
    
    fig.show()

def gmt_LL(grid_LL, data_LL):
    x, y = np.meshgrid( grid_LL['lon']+180,
                        grid_LL['lat'], indexing='ij')
    
    z = np.nan_to_num( np.transpose(data_LL), nan=0 )
    
    data_to_fig( x=x,
                 y=y,
                 z=z,
                 proj='W180/12',
                 frame=['WSne', 'g30', 'ya30'],
                 cmap='mycmap.cpt',
                 outf='gmtplot.pdf')

def plt_LL(grid_LL, data_LL):
    pmax = np.nanmax(np.abs(data_LL))

    fig = plt.figure(figsize=[3, 2])
    ax = plt.axes(projection=ccrs.PlateCarree())
    pcm = ax.pcolormesh(grid_LL['lon'], grid_LL['lat'], data_LL, shading='auto', cmap=cmc.vik, vmin=-pmax, vmax=pmax)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    gl = ax.gridlines(draw_labels=False)
    gl.top_labels = True
    gl.right_labels = True
    plt.colorbar(pcm, ax=ax)
    
    plt.show()

if __name__ == '__main__':
    #####################################################################
    ## Time and radial points of interest.                             ##
    #####################################################################
    path        = 'state/ridge_20km_cs32x32x20/'
    irad        = 15
    itime       = 100
    
    #####################################################################
    ## Resolution of the regrid and whether use weights. Should be set ##
    ## to False if you changed the resolution or are going to use a    ##
    ## different method for interpolation.                             ##
    #####################################################################
    resolution  = 2.
    use_weights = False
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
    #plt_LL( grid_LL, U_LL )
    gmt_LL( grid_LL, U_LL )

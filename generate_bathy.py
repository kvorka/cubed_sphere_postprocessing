import numpy

#####################################################################################################
## Set up of the model.                                                                            ##
#####################################################################################################
bathy     = 2        ## 0-flat, 1-Gauss peak, 2-Gauss ridge
potential = 2        ## 0-obliquity, 1-eccentricity, 2-full tides
nn        = 32       ## cubed sphere single tile resolution
nz        = 20       ## number of radial points
a         = 1561e3   ## radius of the sphere in meters
H         = -1e5     ## depth of the ocean in meters
period    = 306806   ## rotation period in  seconds
obl       = 0.053    ## obliquity
ecc       = 0.0094   ## eccentricity
nt        = 25       ## number of steps within a period
wg        = 10       ## width of the Gaussians in degrees
hg        = 2e4      ## height of the Gaussians in meters
prec      = ">f8"    ## Big-endian float64

#####################################################################################################
## Subroutine for saving and reading the binary files.                                             ##
#####################################################################################################
def write_bin(filename, array):
    array.astype( dtype=prec ).ravel( order='F' ).tofile( filename )

def read_bin(filename):
    return numpy.fromfile( filename, dtype=prec )

#####################################################################################################
## Loading the cubed-sphere longitudes and latitudes from .mitgrid files.                          ##
#####################################################################################################
lon = numpy.zeros( (nn*6, nn), dtype=prec )
lat = numpy.zeros( (nn*6, nn), dtype=prec )

for i in range(6):
    tmp = numpy.reshape( read_bin( f"tile00{i+1}.mitgrid" ), (nn+1,nn+1,16), order='F' )
    
    lon[i*nn:(i+1)*nn,:] = tmp[:nn,:nn,0]
    lat[i*nn:(i+1)*nn,:] = tmp[:nn,:nn,1]

slat, clat = numpy.sin( numpy.deg2rad( lat ) ), numpy.cos( numpy.deg2rad( lat ) )
slon, clon = numpy.sin( numpy.deg2rad( lon ) ), numpy.cos( numpy.deg2rad( lon ) )

s2lat, c2lat = numpy.sin( numpy.deg2rad( 2 * lat ) ), numpy.cos( numpy.deg2rad( 2 * lat ) )
s2lon, c2lon = numpy.sin( numpy.deg2rad( 2 * lon ) ), numpy.cos( numpy.deg2rad( 2 * lon ) )

slat2, clat2 = slat**2, clat**2

#####################################################################################################
## Generating random velocity file.                                                                ##
#####################################################################################################
write_bin( 'initrand_cs.bin', 1e-6 * numpy.random.randn(6*nn, nn, nz) )

#####################################################################################################
## Generating bathymetry file.                                                                     ##
#####################################################################################################
topo = H * numpy.ones( (6*nn, nn) )

if bathy == 0:
    pass
if bathy == 1:
    topo += hg * numpy.exp( -( numpy.rad2deg( numpy.arcsin( clat * slon ) )**2 + lat**2 ) / ( 2 * wg**2 ) )
elif bathy == 2:
    topo += hg * numpy.exp( -( numpy.rad2deg( numpy.arcsin( clat * slon ) )**2          ) / ( 2 * wg**2 ) )

write_bin( 'topo_cs.bin', topo )

#####################################################################################################
## Generating tidal potential file.                                                                ##
#####################################################################################################
omega = 2 * numpy.pi / period
fobl  = +1.50 * omega**2 * a**2 * numpy.deg2rad( obl )
fecc  = -0.75 * omega**2 * a**2 * ecc

times = numpy.arange( 0, period, period / nt )
ctime = numpy.reshape( numpy.cos(omega * times), (1, 1, -1) )
stime = numpy.reshape( numpy.sin(omega * times), (1, 1, -1) )

if potential == 0 or potential == 2:
    static_obl = ( fobl * s2lat * clon )[:,:,numpy.newaxis]

if potential == 1 or potential == 2:
    static_ecc1 = ( fecc * ( 3 * slat2 - 3 * clat2 * c2lon - 1 ) )[:,:,numpy.newaxis]
    static_ecc2 = ( fecc * (            -4 * clat2 * s2lon     ) )[:,:,numpy.newaxis]

match potential:
    case 0: pot = static_obl * ctime
    case 1: pot =                      static_ecc1 * ctime + static_ecc2 * stime
    case 2: pot = static_obl * ctime + static_ecc1 * ctime + static_ecc2 * stime
    case _: pot = numpy.zeros( (6*nn, nn, nt), dtype=prec )

write_bin( 'tidePot_cs.bin', pot )
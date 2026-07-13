import numpy
import argparse

#####################################################################################################
## Set up of the model.                                                                            ##
#####################################################################################################
parser = argparse.ArgumentParser()

parser.add_argument( '-bathy',     '--bathy',     type=str, choices=['flat', 'peak', 'ridge'], default='flat', help='Bathymetry type'      )
parser.add_argument( '-potential', '--potential', type=str, choices=['obl', 'ecc', 'full' ],   default='full', help='Tidal potential type' )

parser.add_argument( '-rs',     '--radius', type=float, default=1561.,   help='Sphere radius in km'          )
parser.add_argument( '-depth' , '--depth',  type=float, default=100.,    help='Ocean depth in km'            )
parser.add_argument( '-period', '--period', type=float, default=306806., help='Rotational period in seconds' )
parser.add_argument( '-obl',    '--obl',    type=float, default=0.053,   help='Obliquity in degrees'         )
parser.add_argument( '-ecc',    '--ecc',    type=float, default=0.0094,  help='Eccentricity'                 )

parser.add_argument( '-nx', '--nx', type=int, default=32, help='CS grid resolution'                  )
parser.add_argument( '-nz', '--nz', type=int, default=20, help='Radial grid resolution'              )
parser.add_argument( '-nt', '--nt', type=int, default=25, help='One period resolution for potential' )

parser.add_argument( '-wg', '--wg', type=float, default=10., help='Width of the Gaussian in degrees' )
parser.add_argument( '-hg', '--hg', type=float, default=20., help='Height of the Gaussian in km'     )

args = parser.parse_args()

bathy     = { 'flat' : 0, 'peak' : 1, 'ridge' : 2 }[args.bathy]
potential = { 'obl'  : 0, 'ecc'  : 1, 'full'  : 2 }[args.potential]

nn        = args.nx
nz        = args.nz
a         = args.radius * ( 1e3)
H         = args.depth  * (-1e3)
period    = args.period
obl       = args.obl
ecc       = args.ecc
nt        = args.nt
wg        = args.wg
hg        = args.hg * 1e3
prec      = ">f8"               ## Big-endian float64

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
################################################################################################################
## Cubed sphere grid generator based on http://mailman.mitgcm.org/pipermail/mitgcm-devel/2013-May/005863.html ##
## Only the conformal centerpole grid is coded in here, as it is the only grid that I have been using.        ##
################################################################################################################
import sys
import numpy
import argparse

def pad_array(a):
    return numpy.pad( a, ((0,1), (0,1), (0,0)), constant_values=0 )

def expand_array(a):
    return numpy.repeat( a[:,:,numpy.newaxis], 6, axis=2 )

def write_blocks(fout, a, prec, machine):
    a.astype( prec ).T.byteswap( sys.byteorder != machine ).tofile( fout )

def angle_between_vectors(vec1, vec2):
    vprod = numpy.sum( vec1 * vec2, axis=2 )
    nrm   = numpy.sqrt( numpy.sum( vec1**2, axis=2 ) * numpy.sum( vec2**2, axis=2 ) )
    
    return numpy.arccos( numpy.clip( vprod / nrm, -1., 1. ) )

def plane_normal(P1, P2):
    plane = numpy.cross( P1, P2, axis=2 )
    
    return plane / numpy.linalg.norm( plane, axis=2, keepdims=True )

def excess_of_quad(v1, v2, v3, v4):
    p1 = plane_normal( v1, v2 )
    p2 = plane_normal( v2, v3 )
    p3 = plane_normal( v3, v4 )
    p4 = plane_normal( v4, v1 )
    
    return 2 * numpy.pi - ( angle_between_vectors( p2, p1 ) + angle_between_vectors( p3, p2 ) +
                            angle_between_vectors( p4, p3 ) + angle_between_vectors( p1, p4 ) )

def rotate_about_axis(lx, ly, lz, angle, axis):
    s, c = numpy.sin(angle), numpy.cos(angle)
    
    if abs(c) < 1.e-9: 
        c, s = 0., numpy.sign(s)
    
    if axis == 'z': return c*lx-s*ly, s*lx+c*ly, lz.copy()
    if axis == 'y': return c*lx+s*lz, ly.copy(), -s*lx+c*lz
    if axis == 'x': return lx.copy(), c*ly-s*lz, +s*ly+c*lz

def permutetiles(b, n):
    c = b.copy()
    a = numpy.zeros_like( b )
    
    for k in range(n):
        a[:,:,0] = c[:,:,1]
        a[:,:,1] = c[::-1,:,3].T
        a[:,:,2] = c[::-1,:,2].T
        a[:,:,3] = c[:,:,4]
        a[:,:,4] = c[:,::-1,0].T
        a[:,:,5] = c[:,::-1,5].T
        
        c, a = a, c
    
    return c

def map_lonlat2xyz(lon, lat):
    return numpy.cos(lat) * numpy.cos(lon), numpy.cos(lat) * numpy.sin(lon), numpy.sin(lat)

def map_xyz2lonlat(x, y, z):
    return numpy.arctan2(y, x), numpy.arctan2(z, numpy.hypot(x, y))

def map_xy2xyz(xi, yi):
    thrd  = 1. / 3
    z_a0  = 1.j
    z_a1  = 1.j - 1
    z_a2  = 0.j - 1 + numpy.sqrt(3.)
    polyA = numpy.array( [ -0.00009749136078, -0.00010518717478, -0.00011468147908, -0.00012466390509, -0.00013660647356, 
                           -0.00014967258633, -0.00016510295548, -0.00018247947703, -0.00020297175587, -0.00022659577923, 
                           -0.00025464473946, -0.00028769063795, -0.00032745130951, -0.00037537743098, -0.00043418115349, 
                           -0.00050696402446, -0.00059869243613, -0.00071604933286, -0.00086946107050, -0.00107458043205,
                           -0.00135682443774, -0.00175869000970, -0.00235482619663, -0.00329250387158, -0.00486626515498, 
                           -0.00791314396396, -0.01895884801823, -0.05573055466344, -0.38183513110512,  1.47713057321600 ] )
    
    xc = numpy.abs(xi)
    yc = numpy.abs(yi)
    zc = (1 - xc) + 1j * (1 - yc)
    
    zc = ( numpy.where( yc > xc, zc.imag + z_a0 * zc.real, zc ) / 2 ) ** 4
    zc = z_a0**thrd * ( numpy.polyval(polyA, zc) * zc * z_a0 )**thrd
    zc = ( zc - z_a2 ) / ( z_a1 * z_a2 * zc / 2 + z_a1 )
    
    z = 2 / ( 1 + numpy.abs(zc)**2 )
    zc = zc * z
    zc = numpy.where( yc > xc, z_a0 * zc.conj(), zc )
    
    x = numpy.sign( xi ) * numpy.abs( zc.real )
    y = numpy.sign( yi ) * numpy.abs( zc.imag )
    z = z - 1
    
    return x, y, z

def reduce_E(E, nratio):
    def sum_blocks(mat, rows, cols):
        pad_r = max( 0, rows * nratio - mat.shape[0] )
        pad_c = max( 0, cols * nratio - mat.shape[1] )
        
        if pad_r > 0 or pad_c > 0:
            mat = numpy.pad( mat, ((0, pad_r), (0, pad_c)), mode='wrap' )
            
        return mat[:rows*nratio,:cols*nratio].reshape(rows, nratio, cols, nratio).sum(axis=(1, 3))
    
    nx = E.shape[0] // nratio + 1
    n2 = nratio // 2
    
    Ec = sum_blocks( E, nx-1, nx-1 )
    Ev = sum_blocks( numpy.roll( E, n2, axis=1 ), nx-1, nx )
    Ez = sum_blocks( numpy.roll( E, (n2, n2), axis=(0, 1) ), nx, nx )
    
    for mat in ( Ec, Ez, Ev ):
        mat[:,:] = ( mat[:,:] + mat[::-1, :] ) / 2
        mat[:,:] = ( mat[:,:] + mat[:, ::-1] ) / 2
    
    Ez[ numpy.ix_( [0, -1], [0, -1] ) ] *= 0.75
    
    return Ec, Ez, Ev

def reduce_dx(dx, nratio):
    nxf = dx.shape[1]
    n2  = nratio // 2
    
    kg = numpy.arange(  0, nxf,   nratio )
    kc = numpy.arange( n2, nxf,   nratio )
    jg = numpy.arange(  0, nxf-1, nratio )
    
    jc = numpy.append( nxf-n2-1, kc )
    
    dxg = dx[ jg[:,None], kg ]
    dxf = dx[ jg[:,None], kc ]
    dxc = dx[ jc[:,None], kc ]
    dxv = dx[ jc[:,None], kg ]
    
    for _ in range(1, nratio):
        jg = numpy.mod( jg+1, nxf-1 )
        jc = numpy.mod( jc+1, nxf-1 )
        
        for mat, j, k in ( (dxg,jg,kg), (dxf,jg,kc), (dxc,jc,kc), (dxv,jc,kg) ):
            mat[:,:] += dx[ j[:,None], k ]
            mat[:,:]  = ( mat[:,:] + mat[::-1,:] ) / 2
            mat[:,:]  = ( mat[:,:] + mat[:,::-1] ) / 2
        
    return dxg, dxc, dxf, dxv

def calc_fvgrid(lx1, ly1, lz1):
    vertices = numpy.stack( ( lx1, ly1, lz1 ), axis=2 )
    
    dx1 = angle_between_vectors( vertices[ :-1,:,:], vertices[1:  ,:,:] )
    dy  = angle_between_vectors( vertices[:, :-1,:], vertices[:,1:  ,:] )
    E   = excess_of_quad( vertices[:-1,:-1,:], vertices[1:,:-1,:], vertices[1:,1:,:], vertices[:-1,1:,:] )
    
    dx = ( dx1 + dy.T ) / 2
    dy = (       dx.T )
    E  = (  E  +  E.T ) / 2
    
    return dx, dy, E

def centerpole_geometry(nx, lx1, ly1, lz1):
    idx = slice( nx-1, None, -1 )
    
    lonP, latP = map_xyz2lonlat( lx1, ly1, lz1 )
    latP       = ( latP + latP.T ) / 2
    
    lonP[lonP >= numpy.pi] -= 2 * numpy.pi
    lonP = ( lonP - 1.5 * numpy.pi - lonP.T ) / 2
    numpy.fill_diagonal( lonP, -0.75 * numpy.pi )
    
    latP = numpy.concatenate( ( latP[:-1,:  ],           latP[idx,:] ), axis=0 )
    latP = numpy.concatenate( ( latP[:  ,:-1],           latP[:,idx] ), axis=1 )
    lonP = numpy.concatenate( ( lonP[:-1,:  ], -numpy.pi-lonP[idx,:] ), axis=0 )
    lonP = numpy.concatenate( ( lonP[:  ,:-1],          -lonP[:,idx] ), axis=1 )
    
    lonE, latE = map_xyz2lonlat( *rotate_about_axis( lx1, ly1, lz1, numpy.pi/2, 'x' ) )
    
    lonE[0, :] = -0.75 * numpy.pi
    latE[:,-1] =  0.
    latE[:, 0] = -latP[0:nx,0]
    
    latE = numpy.concatenate( ( latE[:-1,:  ],           latE[idx,:] ), axis=0 )
    latE = numpy.concatenate( ( latE[:  ,:-1],          -latE[:,idx] ), axis=1 )
    lonE = numpy.concatenate( ( lonE[:-1,:  ], -numpy.pi-lonE[idx,:] ), axis=0 )
    lonE = numpy.concatenate( ( lonE[:  ,:-1],           lonE[:,idx] ), axis=1 )

    return latP, lonP, latE, lonE

def convertMITgrid(xc, yc, xg, yg, dxc, dyc, dxg, dyg, dxf, dyf, dxv, dyu, rac, raw, ras, raz, prec, machine):
    all1 = slice( None )
    all2 = slice( None, -1 )
    rev1 = slice( None, None, -1 )
    rev2 = slice( -2,   None, -1 )
    
    xc,  yc,  xg,  yg,  dxc, dyc, dxg, dyg, \
    dxf, dyf, dxv, dyu, rac, raw, ras, raz  = map( pad_array, (  xc,  yc,  xg,  yg,  dxc, dyc, dxg, dyg, \
                                                                 dxf, dyf, dxv, dyu, rac, raw, ras, raz  ) )
    
    for arr in ( xg, yg, dxv, dyu, raz ):
        arr[ -1, -1, all1 ] = numpy.nan
    
    for arr, (t1, t2) in ( (xg, (0, 3)), (yg, (2, 5)), (raz, (0, 3)) ):
        arr[  0, -1, [0,2,4] ] = arr[ 0,0,t1 ]
        arr[ -1,  0, [1,3,5] ] = arr[ 0,0,t2 ]
        
        arr[   -1, all1, [0,2,4] ] = arr[    0, all1, [1,3,5] ]
        arr[ all1,   -1, [0,2,4] ] = arr[    0, rev1, [2,4,0] ].T
        arr[ -1,   all1, [1,3,5] ] = arr[ rev1,    0, [3,5,1] ].T
        arr[ all1,   -1, [1,3,5] ] = arr[ all1,    0, [2,4,0] ]
    
    for arr in ( dxv, dyu ):
        arr[  0, -1, [0,2,4] ] = dxv[ 0, 0, 0 ]
        arr[ -1,  0, [1,3,5] ] = dxv[ 0, 0, 3 ]
    
    for u, v in ( (dxv, dyu), (dyu, dxv) ):
        u[    -1, all1, [0,2,4] ] = u[    0, all1, [1,3,5] ]
        u[  all1,   -1, [0,2,4] ] = v[    0, rev1, [2,4,0] ].T
        u[    -1, all1, [1,3,5] ] = v[ rev1,    0, [3,5,1] ].T
        u[  all1,   -1, [1,3,5] ] = u[ all1,    0, [2,4,0] ]
    
    for x_arr, y_arr in ( (dxc, dyc), (raw, ras), (dyg, dxg) ):
        x_arr[   -1, all2, all1 ] = numpy.nan
        y_arr[ all2,   -1, all1 ] = numpy.nan
        
        x_arr[   -1, all1, [0,2,4] ] = x_arr[    0, all1, [1,3,5] ]
        x_arr[   -1, all2, [1,3,5] ] = y_arr[ rev2,    0, [3,5,1] ].T
        y_arr[ all1,   -1, [1,3,5] ] = y_arr[ all1,    0, [2,4,0] ]
        y_arr[ all2,   -1, [0,2,4] ] = x_arr[    0, rev2, [2,4,0] ].T
    
    for i in range(6):
        with open(f"tile{i+1:03d}.mitgrid", 'wb') as fout:
            for grid in [ xc, yc, dxf, dyf, rac, xg, yg, dxv, dyu, raz, dxc, dyc, raw, ras, dxg, dyg ]:
                write_blocks( fout, grid[:,:,i], prec, machine )

def build_grid(nx, Rsphere, nratio=4, prec='d', machine='big'):
    nx  += 1
    nxf  = nratio * (nx - 1) + 1
    
    idx1 = slice( (nxf-1)//2-1, None, -1 )
    idx2 = slice( (nxf+1)//2-1, None, -1 )
    
    idxp = slice( 1+nratio//2,  None, nratio )
    idxm = slice(   nratio//2-1, -2,  nratio )
    idxc = slice(   nratio//2,  None, nratio )
    idxd = slice(   None,       None, nratio )
    
    q      = numpy.linspace( -1., 0., (nxf-1)//2+1 )
    lx, ly = numpy.meshgrid( q, q, indexing='ij' )
    
    nxlx          = lx.shape[0]
    lx1, ly1, lz1 = map_xy2xyz( lx, ly )
    
    dx, _, E = calc_fvgrid( lx1, ly1, lz1 )
    
    dx = numpy.concatenate( ( dx,        dx[idx1,:] ), axis=0 )
    dx = numpy.concatenate( ( dx[:,:-1], dx[:,idx2] ), axis=1 )
    E  = numpy.concatenate( ( E,          E[idx1,:] ), axis=0 )
    E  = numpy.concatenate( ( E,          E[:,idx1] ), axis=1 )
    
    dxg, dxc, dxf, dxv = reduce_dx(dx, nratio)
    Ec, Ez, Ev         = reduce_E(E, nratio)
    
    dyg, dyc, dyf, dyu, Eu = dxg.T, dxc.T, dxf.T, dxv.T, Ev.T
    
    LatG = numpy.zeros( ( 2*nxlx-1, 2*nxlx-1, 6 ) )
    LonG = numpy.zeros( ( 2*nxlx-1, 2*nxlx-1, 6 ) )
    
    latP, lonP, latE, lonE = centerpole_geometry( nxlx, lx1, ly1, lz1 )
    
    LatG[:,:,0], LonG[:,:,0] =  latE,           lonE   - numpy.pi * ( 0.5 - 2 * ( lonE <= -numpy.pi/2 ) )
    LatG[:,:,1], LonG[:,:,1] =  latE,           lonE
    LatG[:,:,2], LonG[:,:,2] =  latP,           lonP
    LatG[:,:,3], LonG[:,:,3] =  latE[:,::-1].T, lonE.T + numpy.pi / 2
    LatG[:,:,4], LonG[:,:,4] =  latE[:,::-1].T, lonE.T + numpy.pi
    LatG[:,:,5], LonG[:,:,5] = -latP,           lonP[::-1,::-1].T
    
    LatG = permutetiles( LatG, 2 )
    LonG = permutetiles( LonG, 2 )
    
    Q   = numpy.concatenate( ( q[:-1], -q[::-1] ) )
    XYZ = numpy.stack( map_lonlat2xyz( LonG, LatG ), axis=0 )
    
    QX, QY = numpy.meshgrid( Q[idxp]-Q[idxm], Q[idxp]-Q[idxm], indexing='ij' )
    
    dXYZdx = ( XYZ[:,idxp,idxc,:] - XYZ[:,idxm,idxc,:] ) / expand_array( QX )
    dXYZdy = ( XYZ[:,idxc,idxp,:] - XYZ[:,idxc,idxm,:] ) / expand_array( QY )
    
    Q11 = numpy.sum( dXYZdx * dXYZdx, axis=0 )
    Q22 = numpy.sum( dXYZdy * dXYZdy, axis=0 )
    Q12 = numpy.sum( dXYZdx * dXYZdy, axis=0 )
    
    latG = LatG[idxd,idxd,:]
    lonG = LonG[idxd,idxd,:]
    latC = LatG[idxc,idxc,:]
    lonC = LonG[idxc,idxc,:]
    
    Xlon, Ylon = -numpy.sin(lonC), +numpy.cos(lonC)
    
    TUu =  ( dXYZdx[0] * Xlon + dXYZdx[1] * Ylon )
    TVu = -( dXYZdy[0] * Xlon + dXYZdy[1] * Ylon )
    
    Xlat = -numpy.sin(latC) * numpy.cos(lonC) 
    Ylat = -numpy.sin(latC) * numpy.sin(lonC) 
    Zlat = +numpy.cos(latC)
    
    TUv = -( dXYZdx[0] * Xlat + dXYZdx[1] * Ylat + dXYZdx[2] * Zlat )
    TVv =  ( dXYZdy[0] * Xlat + dXYZdy[1] * Ylat + dXYZdy[2] * Zlat )
    
    det = numpy.sqrt( TUu * TVv - TUv * TVu )
    
    TUu /= det
    TUv /= det
    TVu /= det
    TVv /= det
    
    convertMITgrid( *[ numpy.degrees( arr ) for arr in (lonC, latC) ],
                    *[ numpy.degrees( arr[0:nx-1,0:nx-1,:] ) for arr in (lonG, latG) ],
                    *[ Rsphere * expand_array( arr[0:nx-1,0:nx-1] ) for arr in (dxc, dyc, dxg, dyg, dxf, dyf, dxv, dyu) ],
                    *[ Rsphere**2 * expand_array(arr[0:nx-1,0:nx-1]) for arr in (Ec, Eu, Ev, Ez) ], prec, machine )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument( '-nx', '--nx',     type=int,   default=32,    help='CS grid resolution' )
    parser.add_argument( '-rs', '--radius', type=float, default=1561., help='Sphere radius in km' )
    
    args = parser.parse_args()
    
    build_grid( Rsphere = args.radius * 1e3, 
                nx      = args.nx )
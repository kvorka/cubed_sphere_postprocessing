import sys
import numpy
import argparse

def pad_array(a):
    return numpy.pad( a, ((0,1), (0,1), (0,0)), constant_values=0 )

def expand_array(a):
    return numpy.repeat( a[:,:,numpy.newaxis], 6, axis=2 )

def coord2vector(x, y, z):
    return numpy.stack( (x, y, z), axis=2 )

def angle_between_vectors(vec1, vec2):
    vprod = numpy.sum( vec1 * vec2, axis=2 )
    nrm   = numpy.sqrt( numpy.sum( vec1**2, axis=2 ) * numpy.sum( vec2**2, axis=2 ) )
    
    return numpy.arccos( numpy.clip( vprod / nrm, -1., 1. ) )

def plane_normal(P1, P2):
    plane = numpy.cross( P1, P2, axis=2 )
    mag   = numpy.linalg.norm( plane, axis=2, keepdims=True )
    
    return plane / mag

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
    def mobius(zc):
        a = 1j - 1
        b = numpy.sqrt(3.)-1
        
        A = numpy.array([ -0.00009749136078, -0.00010518717478, -0.00011468147908, -0.00012466390509, -0.00013660647356, 
                          -0.00014967258633, -0.00016510295548, -0.00018247947703, -0.00020297175587, -0.00022659577923, 
                          -0.00025464473946, -0.00028769063795, -0.00032745130951, -0.00037537743098, -0.00043418115349, 
                          -0.00050696402446, -0.00059869243613, -0.00071604933286, -0.00086946107050, -0.00107458043205,
                          -0.00135682443774, -0.00175869000970, -0.00235482619663, -0.00329250387158, -0.00486626515498, 
                          -0.00791314396396, -0.01895884801823, -0.05573055466344, -0.38183513110512,  1.47713057321600 ] )
        
        z = 1j**(1./3) * ( numpy.polyval(A, zc) * zc * 1j )**(1./3)
        
        return ( z - b ) / ( a * b * z / 2 + a )
    
    xc = numpy.abs(xi)
    yc = numpy.abs(yi)
    zc = (1 - xc) + 1j * (1 - yc)
    
    W = mobius( ( numpy.where( yc > xc, zc.imag + 1j * zc.real, zc ) / 2 ) ** 4 )
    
    z = 2 / ( 1 + numpy.abs(W)**2 )
    W = W * z
    W = numpy.where( yc > xc, 1j * W.conj(), W )
    
    x = numpy.sign( xi ) * numpy.abs( W.real )
    y = numpy.sign( yi ) * numpy.abs( W.imag )
    z = z - 1
    
    return x, y, z

def conf_d(qq):
    n = len( qq )
    q = numpy.interp( numpy.linspace( 0, n-1, 2*n-1 ), numpy.arange( n ), qq )
    
    lx, ly, lz = map_xy2xyz( *numpy.meshgrid( q, q, indexing='ij' ) )
    
    vertices = coord2vector( lx[::2,::2], ly[::2,::2], lz[::2,::2] )
    
    return angle_between_vectors( vertices[:-1,:-1,:], vertices[1:,:-1,:] )

def rescale_coordinate(q, method='conf'):
    nxf  = len( q )
    dxg  = conf_d( q )
    nxf4 = ( nxf - 1 ) // 4
    
    match method:
        case 'q=0'  : D = numpy.max( dxg[ :, : ], axis=1 )
        case 'q=1/2': D =            dxg[ :, nxf4 ]
        case 'q=78' : D =            dxg[ :, nxf4 // 4 ]
        case 'q=1'  : D = numpy.min( dxg[ :, : ], axis=1 )
        case 'q=i3':  D =            dxg[ :, 2 ]
        case _:       D =            dxg[ :, 0 ]
    
    s = numpy.cumsum( numpy.r_[ 0.0, D ] )
    S = numpy.cumsum( numpy.r_[ 0.0, numpy.full_like( D, numpy.max(s) / ( nxf - 1 ) ) ] )
    
    match method:
        case 'conf':
            return q.copy()
        case 'q=0' | 'q=1' | 'q=1/2' | 'q=7/8' | 'q=78' | 'q=i3':
            return numpy.interp(S, s, q)
        case 'tan' | 'tan2':
            factor = 2.0 / 3.0 if method == 'tan' else 1.0 / 5.0
            return numpy.tan(factor * q) / numpy.tan(factor * numpy.max(numpy.abs(q)))
        case 'new':
            dq = numpy.ones(nxf)
            dq[0], dq[1] = 0.0, 1.5
            return q[0] + numpy.cumsum(dq / numpy.sum(dq) * (q[-1] - q[0]))

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
        mat = ( mat + mat + mat[::-1, :] + mat[:, ::-1] ) / 4
    
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
            mat += dx[ j[:,None], k ]
            mat  = ( mat + mat[::-1,:] ) / 2
            mat  = ( mat + mat[:,::-1] ) / 2
        
    return dxg, dxc, dxf, dxv

def calc_fvgrid(lx,ly):
    nxf = lx.shape[0]
    
    lx1, ly1, lz1 = map_xy2xyz( lx, ly )
    vertices      = coord2vector( lx1, ly1, lz1 )
    
    dx1 = angle_between_vectors( vertices[ :-1,:,:], vertices[1:  ,:,:] )
    dy  = angle_between_vectors( vertices[:, :-1,:], vertices[:,1:  ,:] )
    E   = excess_of_quad( vertices[:-1,:-1,:], vertices[1:,:-1,:], vertices[1:,1:,:], vertices[:-1,1:,:] )
    
    dx = ( dx1 + dy.T ) / 2
    dy = (       dx.T )
    E  = (  E  +  E.T ) / 2
    
    return dx, dy, E

def calc_geocoords_cornerpole(lx, ly, tile):
    ## Is for sure buggy
    nx  = lx.shape[0]
    nxf = 2 * nx - 1
    
    lx1, ly1, lz1 = map_xy2xyz(lx, ly)
    
    lx1 = numpy.concatenate( ( lx1[:-1, :], -lx1[nx-1::-1, :] ), axis=0 ) 
    lx1 = numpy.concatenate( ( lx1[:, :-1],  lx1[:, nx-1::-1] ), axis=1 )
    ly1 = numpy.concatenate( ( ly1[:-1, :],  ly1[nx-1::-1, :] ), axis=0 ) 
    ly1 = numpy.concatenate( ( ly1[:, :-1], -ly1[:, nx-1::-1] ), axis=1 )
    
    lx1 = ( lx1 + ly1.T ) / 2
    ly1 =         lx1.T
    lz1 = ( lz1 + lz1.T ) / 2
    
    lz1 = numpy.concatenate( ( lz1[:-1, :], lz1[nx-1::-1, :] ), axis=0 ) 
    lz1 = numpy.concatenate( ( lz1[:, :-1], lz1[:, nx-1::-1] ), axis=1 )
    
    lx1, ly1, lz1 = rotate_about_axis( lx1, ly1, lz1, -numpy.pi / 4, 'z' )
    lx1, ly1, lz1 = rotate_about_axis( lx1, ly1, lz1, numpy.arctan( numpy.sqrt(2.) ), 'y' )
    
    lx1 = ( lx1 + lx1.T ) / 2
    ly1 = ( ly1 - ly1.T ) / 2
    lz1 = ( lz1 + lz1.T ) / 2
    
    lonP, latP = map_xyz2lonlat(lx1, ly1, lz1)
    numpy.fill_diagonal( lonP, 0. )
    
    lonP = ( lonP - lonP.T ) / 2
    latP = ( latP + latP.T ) / 2
    
    latP = latP[:,::-1]
    lonP = -lonP[:,::-1]
    
    if tile == 1:
        lat = latP[::-1, :]
        lon = -lonP[::-1, :] - (2.0 / 3.0) * numpy.pi
    elif tile == 2:
        lat = latP
        lon = lonP
    elif tile == 3:
        lat = latP[:, ::-1]
        lon = -lonP[:, ::-1] + (2.0 / 3.0) * numpy.pi
    elif tile == 4:
        lat = -latP[::-1, :]
        lon = lonP[::-1, :] + (1.0 / 3.0) * numpy.pi
    elif tile == 5:
        lat = -latP[::-1, ::-1]
        lon = -lonP[::-1, ::-1] + numpy.pi
    elif tile == 6:
        lat = -latP[:, ::-1]
        lon = lonP[:, ::-1] - (1.0 / 3.0) * numpy.pi
    
    lon = ( lon + numpy.pi ) % ( 2 * numpy.pi ) - numpy.pi
    
    return lat, lon

def calc_geocoords_centerpole(lx, ly, tile):
    nx  = lx.shape[0]
    
    idx1 = slice( None, -1 )
    idx2 = slice( nx-1, None, -1 )
    idx3 = slice( None, None, -1 )
    
    lx1, ly1, lz1 = map_xy2xyz( lx, ly )
    lonP, latP    = map_xyz2lonlat( lx1, ly1, lz1 )
    
    latP = ( latP + latP.T ) / 2
    
    lonP[lonP >= numpy.pi] -= 2 * numpy.pi
    lonP = ( lonP - 3./2. * numpy.pi - lonP.T ) / 2.
    numpy.fill_diagonal( lonP, -0.75 * numpy.pi )
    
    latP = numpy.concatenate( ( latP[idx1,:],           latP[idx2,:] ), axis=0 )
    latP = numpy.concatenate( ( latP[:,idx1],           latP[:,idx2] ), axis=1 )
    lonP = numpy.concatenate( ( lonP[idx1,:], -numpy.pi-lonP[idx2,:] ), axis=0 )
    lonP = numpy.concatenate( ( lonP[:,idx1],          -lonP[:,idx2] ), axis=1 )
    
    lx1, ly1, lz1 = rotate_about_axis( lx1, ly1, lz1, numpy.pi/2, 'x' )
    lonE, latE    = map_xyz2lonlat( lx1, ly1, lz1 )
    
    lonE[0, :] = -0.75 * numpy.pi
    latE[:,-1] = 0.
    latE[:, 0] = -latP[0:nx,0]
    
    latE = numpy.concatenate( ( latE[idx1,:],           latE[idx2,:] ), axis=0 )
    latE = numpy.concatenate( ( latE[:,idx1],          -latE[:,idx2] ), axis=1 )
    lonE = numpy.concatenate( ( lonE[idx1,:], -numpy.pi-lonE[idx2,:] ), axis=0 )
    lonE = numpy.concatenate( ( lonE[:,idx1],           lonE[:,idx2] ), axis=1 )
    
    match tile:
        case 1: return latE, lonE - numpy.pi * ( 0.5 - 2 * ( lonE <= -numpy.pi/2 ) )
        case 2: return latE, lonE
        case 3: return latP, lonP
        case 4: return latE[:, idx3].T, lonE.T + numpy.pi/2
        case 5: return latE[:, idx3].T, lonE.T + numpy.pi
        case 6: return -latP, lonP[idx3, idx3].T

def write_blocks(fout, a, prec, machine):
    a.astype( prec ).T.byteswap( sys.byteorder != machine ).tofile( fout )

def write_tile(file_out, a, prec, machine):
    with open(file_out + '.bin', 'wb') as fout:
        a.astype( prec ).transpose(2,1,0).byteswap( sys.byteorder != machine ).tofile(fout)

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

def build_grid(nx, Rsphere, nratio=4, method='conf', ornt='c', prec='d', machine='big'):
    calc_geocoords = calc_geocoords_centerpole if ornt == 'c' else calc_geocoords_cornerpole
    
    nx  += 1
    nxf  = nratio * (nx - 1) + 1
    
    idx = slice( 0, nx-1 )
    
    idx_1 = slice( (nxf-1)//2-1, None, -1 )
    idx_2 = slice( (nxf+1)//2-1, None, -1 )
    
    idx_p = slice( 1+nratio//2,  None, nratio )
    idx_m = slice(   nratio//2-1, -2,  nratio )
    idx_c = slice(   nratio//2,  None, nratio )
    idx_d = slice(   None,       None, nratio )
    
    q      = rescale_coordinate( numpy.linspace( -1., 0., (nxf-1)//2+1 ), method )
    lx, ly = numpy.meshgrid( q, q, indexing='ij' )
    
    dx, _, E = calc_fvgrid( lx, ly )  
    
    dx = numpy.concatenate( ( dx,        dx[idx_1,:] ), axis=0 )
    dx = numpy.concatenate( ( dx[:,:-1], dx[:,idx_2] ), axis=1 )
    E  = numpy.concatenate( ( E,          E[idx_1,:] ), axis=0 )
    E  = numpy.concatenate( ( E,          E[:,idx_1] ), axis=1 )
    
    dxg, dxc, dxf, dxv = reduce_dx(dx, nratio)
    Ec, Ez, Ev         = reduce_E(E, nratio)
    
    dyg, dyc, dyf, dyu, Eu = dxg.T, dxc.T, dxf.T, dxv.T, Ev.T
    
    LatG = numpy.zeros( ( 2*lx.shape[0]-1, 2*lx.shape[0]-1, 6 ) )
    LonG = numpy.zeros( ( 2*lx.shape[0]-1, 2*lx.shape[0]-1, 6 ) )
    
    for n in range(6):
        LatG[:,:,n], LonG[:,:,n] = calc_geocoords( lx, ly, n+1 )
    
    if ornt == 'c':
        LatG = permutetiles(LatG, 2)
        LonG = permutetiles(LonG, 2)
    
    Q   = numpy.concatenate( (q[:-1], -q[::-1]) )
    XYZ = numpy.stack( map_lonlat2xyz( LonG, LatG ), axis=0 )
    
    QX, QY = numpy.meshgrid( Q[idx_p]-Q[idx_m], Q[idx_p]-Q[idx_m], indexing='ij' )
    
    dXYZdx = ( XYZ[:,idx_p,idx_c,:] - XYZ[:,idx_m,idx_c,:] ) / expand_array( QX )
    dXYZdy = ( XYZ[:,idx_c,idx_p,:] - XYZ[:,idx_c,idx_m,:] ) / expand_array( QY )
    
    Q11 = numpy.sum( dXYZdx * dXYZdx, axis=0 )
    Q22 = numpy.sum( dXYZdy * dXYZdy, axis=0 )
    Q12 = numpy.sum( dXYZdx * dXYZdy, axis=0 )
    
    latG = LatG[idx_d,idx_d,:]
    lonG = LonG[idx_d,idx_d,:]
    latC = LatG[idx_c,idx_c,:]
    lonC = LonG[idx_c,idx_c,:]
    
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
                    *[ numpy.degrees( arr[idx,idx,:] ) for arr in (lonG, latG) ],
                    *[ Rsphere * expand_array( arr[idx,idx] ) for arr in (dxc, dyc, dxg, dyg, dxf, dyf, dxv, dyu) ],
                    *[ Rsphere**2 * expand_array(arr[idx, idx]) for arr in (Ec, Eu, Ev, Ez) ], prec, machine )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument( '-n', '--n',      type=int,   default=32,    help='CS grid resolution' )
    parser.add_argument( '-r', '--radius', type=float, default=1561., help='Sphere radius in km' )
    
    args = parser.parse_args()
    
    n = args.n
    r = args.radius * 1e3
    
    build_grid( Rsphere = r, nx = n )
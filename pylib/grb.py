import os
import sys
import numpy
import array

class grd_build:
    def __init__(self, path2cs, nx, Rsphere):
        gengrid( Rsphere=Rsphere, nx=nx, path=path2cs )

def pad_array(a):
    return numpy.pad( a, ((0, 1), (0, 1), (0, 0)), constant_values=0 )

def expand_array(a):
    return numpy.repeat( a[:,:,numpy.newaxis], 6, axis=2 )

def coord2vector(x, y, z):
    return numpy.stack( (x, y, z), axis=2 )

def angle_between_vectors(vec1, vec2):
    vprod = numpy.sum( vec1 * vec2, axis=2 )
    nrm   = numpy.sqrt( numpy.sum( vec1**2, axis=2 ) * 
                        numpy.sum( vec2**2, axis=2 ) )
    
    return numpy.arccos( numpy.clip( vprod / nrm, -1., 1. ) )

def plane_normal(P1, P2):
    plane = numpy.cross( P1, P2, axis=2 )
    mag   = numpy.linalg.norm( plane, axis=2, keepdims=True )
    
    return plane / mag

def rotate_about_zaxis(lx, ly, lz, angle):
    s = numpy.sin( angle )
    c = numpy.cos( angle )
    
    if c < 1.e-9:
       c = 0.
       s = numpy.sign( s )
    
    x = c * lx - s * ly
    y = s * lx + c * ly
    
    return x, y, lz.copy()

def rotate_about_yaxis(lx, ly, lz, angle):
    s = numpy.sin( angle )
    c = numpy.cos( angle )
    
    if c < 1.e-9:
       c = 0.
       s = numpy.sign( s )
    
    x = +c * lx + s * lz
    z = -s * lx + c * lz
    
    return x, ly.copy(), z

def rotate_about_xaxis(lx, ly, lz, angle):
    s = numpy.sin( angle )
    c = numpy.cos( angle )
    
    if c < 1.e-9:
       c = 0.
       s = numpy.sign( s )
    
    y = c * ly - s * lz
    z = s * ly + c * lz
    
    return lx.copy(), y, z

def map_lonlat2xyz(lon, lat):
    x = numpy.cos(lat) * numpy.cos(lon)
    y = numpy.cos(lat) * numpy.sin(lon)
    z = numpy.sin(lat)
    
    return x, y, z

def map_xyz2lonlat(x, y, z):
    a  = numpy.sqrt( x**2 + y**2 )
    a1 = numpy.where( a == 0.0, 1.0, a )
    
    x1 = numpy.where( x == 0.0, 1.0, x )
    y1 = numpy.where( x == 0.0, numpy.inf, y )
    z1 = numpy.where( a == 0.0, numpy.sign( numpy.where( z == 0.0, 1.0, z ) ) * numpy.inf, z )
    
    lon = numpy.arctan( y1 / x1 )
    lon = numpy.where( ( x <  0.0 ) & ( y >= 0.0 ), +numpy.pi + lon, lon )
    lon = numpy.where( ( x <= 0.0 ) & ( y <  0.0 ), -numpy.pi + lon, lon)
    
    lat = numpy.arctan( z1 / a1 )
    lat = numpy.where( ( a == 0.0 ) & ( z == 0.0 ), numpy.nan, lat )
    
    return numpy.arctan2(y, x), numpy.arctan2(z, numpy.hypot(x, y))
    #return lon, lat

def WofZ(z):
    A = [ 1.47713057321600, -0.38183513110512, -0.05573055466344, \
         -0.01895884801823, -0.00791314396396, -0.00486626515498, \
         -0.00329250387158, -0.00235482619663, -0.00175869000970, \
         -0.00135682443774, -0.00107458043205, -0.00086946107050, \
         -0.00071604933286, -0.00059869243613, -0.00050696402446, \
         -0.00043418115349, -0.00037537743098, -0.00032745130951, \
         -0.00028769063795, -0.00025464473946, -0.00022659577923, \
         -0.00020297175587, -0.00018247947703, -0.00016510295548, \
         -0.00014967258633, -0.00013660647356, -0.00012466390509, \
         -0.00011468147908, -0.00010518717478, -0.00009749136078  ]
    
    return numpy.polyval( A[::-1], z ) * z

def map_xy2xyz(xi, yi):
    xc = numpy.abs(xi)
    yc = numpy.abs(yi)
    
    mask_kxy = yc > xc
    mask_kx  = xi < 0.0
    mask_ky  = yi < 0.0
    
    x1_orig = xc.copy()
    y1_orig = yc.copy()
    
    xc = 1.0 - xc
    yc = 1.0 - yc
    
    xc = numpy.where( mask_kxy, 1.0 - y1_orig, xc )
    yc = numpy.where( mask_kxy, 1.0 - x1_orig, yc )
    
    zi = ( (xc + 1j * yc) / 2 )**4
    W  = WofZ(zi)
    
    thrd = 1./3.
    i3 = 1j**thrd
    ra = numpy.sqrt(3.0) - 1.0
    cb = 1j - 1.0
    cc = ra * cb / 2.0
    
    W = i3 * (W * 1j)**thrd
    W = (W - ra) / (cb + cc * W)
    
    x1 = numpy.real(W)
    y1 = numpy.imag(W)
    
    H = 2.0 / (1.0 + x1**2 + y1**2)
    x1 = x1 * H
    y1 = y1 * H
    z1 = H - 1.0
    
    t1 = x1.copy()
    x1 = numpy.where( mask_kxy, y1, x1 )
    y1 = numpy.where( mask_kxy, t1, y1 )
    
    x1 = numpy.where( mask_kx, -x1, x1 )
    y1 = numpy.where( mask_ky, -y1, y1 )
    
    x1 = numpy.where( xi == 0.0, 0.0, x1 )
    y1 = numpy.where( yi == 0.0, 0.0, y1 )
    
    return x1, y1, z1

def permutetiles(b, n):
    c = b.copy()
    a = numpy.zeros_like( b )
    
    for k in range(n):
        a[:,:,0] = c[:,:,1]
        a[:,:,1] = c[::-1,:,3].transpose()
        a[:,:,2] = c[::-1,:,2].transpose()
        a[:,:,3] = c[:,:,4]
        a[:,:,4] = c[:,::-1,0].transpose()
        a[:,:,5] = c[:,::-1,5].transpose()
        
        c, a = a, c
    
    return c

def conf_d(qq):
    nx  = numpy.size(qq)
    nxf = 2 * (nx - 1) + 1
    
    q       = numpy.zeros(nxf)
    q[::2]  = qq
    q[1::2] = ( q[:-2:2] + q[2::2] ) / 2
    
    lx, ly = numpy.meshgrid( q, q, indexing='ij' )
    
    lx1, ly1, lz1 = map_xy2xyz(lx, ly)
    
    vertices = coord2vector( lx1[::2,::2], ly1[::2,::2], lz1[::2,::2] )
    
    return angle_between_vectors( vertices[ :-1,:-1,:], vertices[1:  ,:-1,:] )

def rescale_coordinate(q, method='conf'):
    nxf = numpy.size(q)
    nx  = round((nxf - 1) / 2)
    dxg = conf_d(q)
    
    if method == 'q=0':
        D = numpy.max(dxg, axis=1)
    elif method == 'q=1/2':
        D = dxg[:,round(nx/2)]
    elif method == 'q=78':
        D = dxg[:,round(nx/8)]
    elif method == 'q=1':
        D = numpy.min(dxg, axis=1)
    elif method == 'q=i3':
        D = dxg[:,2]
    else:
        D = dxg[:,0]
    
    s      = numpy.cumsum( numpy.concatenate( ( [0.], D ) ) )
    dS_val = numpy.max(s) / (nxf - 1)
    dS     = numpy.full_like(D, dS_val)
    
    S = numpy.cumsum( numpy.concatenate( ( [0.], dS ) ) )
    
    if method == 'conf':
        Q = q.copy()
    elif method in ['q=0', 'q=1', 'q=1/2', 'q=7/8', 'q=78', 'q=i3']:
        Q = numpy.interp(S, s, q)
    elif method == 'tan':
        Q = numpy.tan(2.0 / 3.0 * q) / numpy.tan(2.0 / 3.0 * numpy.max( numpy.abs(q) ) )
    elif method == 'tan2':
        Q = numpy.tan(1.0 / 5.0 * q) / numpy.tan(1.0 / 5.0 * numpy.max( numpy.abs(q) ) )
    elif method == 'new':
        dq = numpy.ones(nxf)
        dq[0] = 0.0
        dq[1] = 1.5
        dq    = dq / numpy.sum(dq) * ( q[-1] - q[0] )
        Q     = q[0] + numpy.cumsum(dq)
    
    return Q

def reduce_E(E, nratio):
    def sum_blocks(mat, rows, cols):
        pad_r = max( 0, rows * nratio - mat.shape[0] )
        pad_c = max( 0, cols * nratio - mat.shape[1] )
        
        if pad_r > 0 or pad_c > 0:
            mat = numpy.pad( mat, ((0, pad_r), (0, pad_c)), mode='wrap' )
            
        return mat[:rows*nratio, :cols*nratio].reshape(rows, nratio, cols, nratio).sum(axis=(1, 3))
    
    nxf      = E.shape[0] + 1
    nx       = (nxf - 1) // nratio + 1
    n_blocks = nx - 1
    
    if nratio == 1:
        Ec = E
        Ev = ( numpy.roll( Ec, 1, axis=1 ) + Ec ) / 2
        Ez = ( numpy.roll( Ev, 1, axis=0 ) + Ev ) / 2
        
    else:
        Ec = sum_blocks( E, n_blocks, n_blocks)
        Ev = sum_blocks( numpy.roll( E, nratio // 2, axis=1 ), n_blocks, nx )
        Ez = sum_blocks( numpy.roll( E, (nratio // 2, nratio // 2), axis=(0, 1)), nx, nx)
        
        for mat in ( Ec, Ez, Ev ):
            mat[:] = ( mat + mat[::-1,:]) / 4 + ( mat + mat[:, ::-1] ) / 4
        
        Ez[ 0, 0] *= 0.75
        Ez[ 0,-1] *= 0.75
        Ez[-1, 0] *= 0.75
        Ez[-1,-1] *= 0.75
        
    return Ec, Ez, Ev

#old version, according to AI is not working correctly
#def reduce_dx(dx, nratio):
#    def sym_in_place(mat):
#        mat[:] = ( mat + mat[::-1, :] ) / 2
#        mat[:] = ( mat + mat[:, ::-1] ) / 2
#    
#    nxf = dx.shape[1]
#    
#    if nratio == 1:
#        dxg =   dx
#        dxf = ( dx[:,:-1] + dx[:,1:] ) / 2
#        dxv = ( numpy.roll( dx,  1, axis=0 ) + dx ) / 2
#        dxc = ( numpy.roll( dxf, 1, axis=0 ) + dxf) / 2
#        
#        return dxg, dxc, dxf, dxv
#    
#    jg = numpy.arange(0, nxf - 1, nratio)
#    jc = numpy.append(nxf - nratio // 2 - 1, numpy.arange(nratio // 2, nxf, nratio))
#    kg = numpy.arange(0, nxf, nratio)
#    kc = numpy.arange(nratio // 2, nxf, nratio)
#    
#    dxg = dx[ jg[:,numpy.newaxis], kg ]
#    dxf = dx[ jg[:,numpy.newaxis], kc ]
#    dxc = dx[ jc[:,numpy.newaxis], kc ]
#    dxv = dx[ jc[:,numpy.newaxis], kg ]
#    
#    for n in range(1,nratio):
#        jg = numpy.mod( jg + 1, nxf - 1 )
#        jc = numpy.mod( jc + 1, nxf - 1 )
#        
#        dxg += dx[ jg[:,numpy.newaxis], kg ]
#        dxf += dx[ jg[:,numpy.newaxis], kc ]
#        dxc += dx[ jc[:,numpy.newaxis], kc ]
#        dxv += dx[ jc[:,numpy.newaxis], kg ]
#        
#        sym_in_place( dxg )
#        sym_in_place( dxf )
#        sym_in_place( dxc )
#        sym_in_place( dxv )
#
#    return dxg, dxc, dxf, dxv

def reduce_dx(dx, nratio):
    nxf = dx.shape[1]
    
    if nratio == 1:
        dxg = ( dx )
        dxf = ( dx[:, :-1] + dx[:, 1:] ) / 2
        dxv = ( numpy.roll(dx, 1, axis=0) + dx ) / 2
        dxc = ( numpy.roll(dxf, 1, axis=0) + dxf ) / 2
        
        return dxg, dxc, dxf, dxv
    
    rows_g = dx.shape[0] // nratio
    
    def sum_rows(mat):
        return mat[:rows_g * nratio,:].reshape(rows_g, nratio, mat.shape[1]).sum(axis=1)
    
    dx_g = dx
    dx_c = numpy.roll(dx, nratio // 2, axis=0)
    
    kg = numpy.arange(0, nxf, nratio)
    kc = numpy.arange(nratio // 2, nxf, nratio)
    
    dxg = sum_rows(dx_g)[:,kg]
    dxf = sum_rows(dx_g)[:,kc]
    dxc = sum_rows(dx_c)[:,kc]
    dxv = sum_rows(dx_c)[:,kg]
    
    for mat in (dxg, dxc, dxf, dxv):
        mat[:] = ( mat + mat[::-1, :] ) / 2
        mat[:] = ( mat + mat[:, ::-1] ) / 2
    
    return dxg, dxc, dxf, dxv

def excess_of_quad(v1, v2, v3, v4):
    p1 = plane_normal( v1, v2 )
    p2 = plane_normal( v2, v3 )
    p3 = plane_normal( v3, v4 )
    p4 = plane_normal( v4, v1 )

    sum_angles = ( angle_between_vectors( p2, p1 ) +
                   angle_between_vectors( p3, p2 ) +
                   angle_between_vectors( p4, p3 ) +
                   angle_between_vectors( p1, p4 ) )
    
    return 2 * numpy.pi - sum_angles

def calc_fvgrid(lx,ly):
    nxf = lx.shape[0]
    
    lx1, ly1, lz1 = map_xy2xyz( lx, ly )
    vertices      = coord2vector( lx1, ly1, lz1 )
    
    dx = angle_between_vectors( vertices[ :-1,:,:], vertices[1:  ,:,:] )
    dy = angle_between_vectors( vertices[:, :-1,:], vertices[:,1:  ,:] )
    E = excess_of_quad( vertices[:-1,:-1,:], vertices[1:,:-1,:], vertices[1:,1:,:], vertices[:-1,1:,:] )
    
    dx = ( dx + dy.transpose() ) / 2
    dy =        dx.transpose()
    E  = ( E +   E.transpose() ) / 2
    
    idx_upper_dx = numpy.triu_indices(nxf - 1, k=1)
    dx[idx_upper_dx[1], idx_upper_dx[0]] = dy[idx_upper_dx[0], idx_upper_dx[1]]
    
    idx_upper_dy = numpy.triu_indices(dy.shape[0], k=1)
    dy[idx_upper_dy[1], idx_upper_dy[0]] = dx[idx_upper_dy[0], idx_upper_dy[1]]
    
    idx_upper_E = numpy.triu_indices(E.shape[0], k=1)
    E[idx_upper_E[1], idx_upper_E[0]] = E[idx_upper_E[0], idx_upper_E[1]]
    
    return dx, dy, E

def calc_geocoords_cornerpole(lx, ly, tile):
    nx  = lx.shape[0]
    nxf = 2 * nx - 1
    
    lx1, ly1, lz1 = map_xy2xyz(lx, ly)
    
    lx1 = numpy.concatenate( ( lx1[:-1, :], -lx1[nx-1::-1, :] ), axis=0 ) 
    lx1 = numpy.concatenate( ( lx1[:, :-1],  lx1[:, nx-1::-1] ), axis=1 )
    
    ly1 = numpy.concatenate( ( ly1[:-1, :],  ly1[nx-1::-1, :] ), axis=0 ) 
    ly1 = numpy.concatenate( ( ly1[:, :-1], -ly1[:, nx-1::-1] ), axis=1 )
    
    lx1 = ( lx1 + ly1.transpose() ) / 2
    ly1 =         lx1.transpose()
    lz1 = ( lz1 + lz1.transpose() ) / 2
    
    lz1 = numpy.concatenate( ( lz1[:-1, :], lz1[nx-1::-1, :] ), axis=0 ) 
    lz1 = numpy.concatenate( ( lz1[:, :-1], lz1[:, nx-1::-1] ), axis=1 )
    
    lx1, ly1, lz1 = rotate_about_zaxis( lx1, ly1, lz1, -numpy.pi / 4 )
    lx1, ly1, lz1 = rotate_about_yaxis( lx1, ly1, lz1, numpy.arctan( numpy.sqrt(2.) ) )
    
    lx1 = ( lx1 + lx1.transpose() ) / 2
    ly1 = ( ly1 - ly1.transpose() ) / 2
    lz1 = ( lz1 + lz1.transpose() ) / 2
    
    lonP, latP = map_xyz2lonlat(lx1, ly1, lz1)
    numpy.fill_diagonal( lonP, 0. )
    
    lonP = ( lonP - lonP.transpose() ) / 2
    latP = ( latP + latP.transpose() ) / 2
    
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
    nxf = 2 * nx - 1
    
    lx1, ly1, lz1 = map_xy2xyz(lx, ly)
    lonP, latP    = map_xyz2lonlat(lx1, ly1, lz1)
    
    latP = ( latP + latP.transpose() ) / 2
    
    lonP = ( lonP + numpy.pi) % (2 * numpy.pi) - numpy.pi
    lonP = ( lonP - 1.5 * numpy.pi - lonP.transpose() ) / 2
    numpy.fill_diagonal( lonP, -0.75 * numpy.pi )
    
    latP = numpy.concatenate( ( latP[:-1, :],           latP[nx-1::-1, :] ), axis=0 )
    latP = numpy.concatenate( ( latP[:, :-1],           latP[:, nx-1::-1] ), axis=1 )
    lonP = numpy.concatenate( ( lonP[:-1, :], -numpy.pi-lonP[nx-1::-1, :] ), axis=0 )
    lonP = numpy.concatenate( ( lonP[:, :-1],          -lonP[:, nx-1::-1] ), axis=1 )
    
    lx1, ly1, lz1 = rotate_about_xaxis( lx1, ly1, lz1, numpy.pi / 2 )
    lonE, latE = map_xyz2lonlat(lx1, ly1, lz1)
    
    lonE[0,:]  = -0.75 * numpy.pi
    latE[:,-1] = 0.
    latE[:,0]  = -latP[0:nx,0]
    
    latE = numpy.concatenate( ( latE[:-1, :],           latE[nx-1::-1, :] ), axis=0 )
    latE = numpy.concatenate( ( latE[:, :-1],          -latE[:, nx-1::-1] ), axis=1 )
    lonE = numpy.concatenate( ( lonE[:-1, :], -numpy.pi-lonE[nx-1::-1, :] ), axis=0 )
    lonE = numpy.concatenate( ( lonE[:, :-1],           lonE[:, nx-1::-1] ), axis=1 )
    
    if tile == 1:
        lat = latE
        lon = lonE - numpy.pi / 2
        lon = (lon + numpy.pi) % (2 * numpy.pi) - numpy.pi
    elif tile == 2:
        lat = latE
        lon = lonE
    elif tile == 3:
        lat = latP
        lon = lonP
    elif tile == 4:
        lat = latE[:,::-1].transpose()
        lon = lonE.transpose() + numpy.pi / 2
    elif tile == 5:
        lat = latE[:,::-1].transpose()
        lon = lonE.transpose() + numpy.pi
    elif tile == 6:
        lat = -latP
        lon = lonP[::-1,::-1].transpose()
    
    return lat, lon

def write_blocks(fout, a, prec, machine):
    a.astype( prec ).transpose().byteswap( sys.byteorder != machine ).tofile( fout )

def write_tile(file_out, a, prec, machine):
    with open(file_out + '.bin', 'wb') as fout:
        a.astype( prec ).transpose(2,1,0).byteswap( sys.byteorder != machine ).tofile(fout)

def convertMITgrid(xc, yc, xg, yg, dxc, dyc, dxg, dyg, dxf, dyf, dxv, dyu, rac, raw, ras, raz, newdir, prec, machine):
    xg  = pad_array(xg);   xg[-1,-1,:]  = numpy.nan
    yg  = pad_array(yg);   yg[-1,-1,:]  = numpy.nan
    dxv = pad_array(dxv);  dxv[-1,-1,:] = numpy.nan
    dyu = pad_array(dyu);  dyu[-1,-1,:] = numpy.nan
    raz = pad_array(raz);  raz[-1,-1,:] = numpy.nan
    
    xg[0,-1,[0,2,4]] = xg[0,0,0]
    xg[-1,0,[1,3,5]] = xg[0,0,3]
    xg[-1,:,[0,2,4]] = xg[0,:,[1,3,5]]
    xg[:,-1,[0,2,4]] = xg[0,::-1,[2,4,0]].transpose()
    xg[-1,:,[1,3,5]] = xg[::-1,0,[3,5,1]].transpose()
    xg[:,-1,[1,3,5]] = xg[:,0,[2,4,0]]
    
    yg[0,-1,[0,2,4]] = yg[0,0,2]
    yg[-1,0,[1,3,5]] = yg[0,0,5]
    yg[-1,:,[0,2,4]] = yg[0,:,[1,3,5]]
    yg[:,-1,[0,2,4]] = yg[0,::-1,[2,4,0]].transpose()
    yg[-1,:,[1,3,5]] = yg[::-1,0,[3,5,1]].transpose()
    yg[:,-1,[1,3,5]] = yg[:,0,[2,4,0]]
    
    raz[0,-1,[0,2,4]] = raz[0,0,0]
    raz[-1,0,[1,3,5]] = raz[0,0,3]
    raz[-1,:,[0,2,4]] = raz[0,:,[1,3,5]]
    raz[:,-1,[0,2,4]] = raz[0,::-1,[2,4,0]].transpose()
    raz[-1,:,[1,3,5]] = raz[::-1,0,[3,5,1]].transpose()
    raz[:,-1,[1,3,5]] = raz[:,0,[2,4,0]]
    
    dxv[0,-1,[0,2,4]] = dxv[0,0,0]
    dxv[-1,0,[1,3,5]] = dxv[0,0,3]
    dyu[0,-1,[0,2,4]] = dxv[0,0,0]
    dyu[-1,0,[1,3,5]] = dxv[0,0,3]
    
    dxv[-1,:,[0,2,4]] = dxv[0,:,[1,3,5]]
    dxv[:,-1,[0,2,4]] = dyu[0,::-1,[2,4,0]].transpose()
    dxv[-1,:,[1,3,5]] = dyu[::-1,0,[3,5,1]].transpose()
    dxv[:,-1,[1,3,5]] = dxv[:,0,[2,4,0]]
    
    dyu[-1,:,[0,2,4]] = dyu[0,:,[1,3,5]]
    dyu[:,-1,[0,2,4]] = dxv[0,::-1,[2,4,0]].transpose()
    dyu[-1,:,[1,3,5]] = dxv[::-1,0,[3,5,1]].transpose()
    dyu[:,-1,[1,3,5]] = dyu[:,0,[2,4,0]]
    
    xc  = pad_array(xc)
    yc  = pad_array(yc)
    dxf = pad_array(dxf)
    dyf = pad_array(dyf)
    rac = pad_array(rac)
    
    dxc = pad_array(dxc); dxc[-1,:-1,:] = numpy.nan
    dyc = pad_array(dyc); dyc[:-1,-1,:] = numpy.nan
    raw = pad_array(raw); raw[-1,:-1,:] = numpy.nan
    ras = pad_array(ras); ras[:-1,-1,:] = numpy.nan
    dxg = pad_array(dxg); dxg[:-1,-1,:] = numpy.nan
    dyg = pad_array(dyg); dyg[-1,:-1,:] = numpy.nan
    
    dxc[-1,:,[0,2,4]]   = dxc[0,:,[1,3,5]]
    dxc[-1,:-1,[1,3,5]] = dyc[-2::-1,0,[3,5,1]].transpose()
    dyc[:,-1,[1,3,5]]   = dyc[:,0,[2,4,0]]
    dyc[:-1,-1,[0,2,4]] = dxc[0,-2::-1,[2,4,0]].transpose()

    raw[-1,:,[0,2,4]]   = raw[0,:,[1,3,5]]
    raw[-1,:-1,[1,3,5]] = ras[-2::-1,0,[3,5,1]].transpose()
    ras[:,-1,[1,3,5]]   = ras[:,0,[2,4,0]]
    ras[:-1,-1,[0,2,4]] = raw[0,-2::-1,[2,4,0]].transpose()

    dyg[-1,:,[0,2,4]]   = dyg[0,:,[1,3,5]]
    dyg[-1,:-1,[1,3,5]] = dxg[-2::-1,0,[3,5,1]].transpose()
    dxg[:,-1,[1,3,5]]   = dxg[:,0,[2,4,0]]
    dxg[:-1,-1,[0,2,4]] = dyg[0,-2::-1,[2,4,0]].transpose()
    
    all_grids = [xc, yc, dxf, dyf, rac, xg, yg, dxv, dyu, raz, dxc, dyc, raw, ras, dxg, dyg]
    
    for i in range(6):
        tile_filename = f"{newdir}/tile{i+1:03d}.mitgrid"
        with open(tile_filename, 'wb') as fout:
            for grid in all_grids:
                write_blocks(fout, grid[:, :, i], prec, machine)

def gengrid(Rsphere, nx, path, nratio=4, method='conf', ornt='c', prec='d', machine='big'):
    nx  = nx + 1
    nxf = nratio * (nx - 1) + 1
    
    q = numpy.linspace( -1., 0., (nxf - 1) // 2 + 1 )
    q = rescale_coordinate( q, method )
    lx, ly = numpy.meshgrid(q, q, indexing='ij')
    
    dx, dy, E = calc_fvgrid(lx, ly)
    del dy
    
    dx = numpy.concatenate( ( dx, dx[(nxf - 1) // 2 - 1::-1,:]),         axis=0 )
    dx = numpy.concatenate( ( dx[:,:-1], dx[:, (nxf + 1) // 2 - 1::-1]), axis=1 )
    E  = numpy.concatenate( ( E, E[(nxf - 1) // 2 - 1::-1, :]),          axis=0 )
    E  = numpy.concatenate( ( E, E[:, (nxf - 1) // 2 - 1::-1]),          axis=1 )
    
    dxg, dxc, dxf, dxv = reduce_dx(dx, nratio)
    Ec, Ez, Ev = reduce_E(E, nratio)
    del dx, E
    
    dyg = dxg.transpose()
    dyc = dxc.transpose()
    dyf = dxf.transpose()
    dyu = dxv.transpose()
    Eu  = Ev.transpose()
    
    nxf_geo = 2 * lx.shape[0] - 1
    LatG = numpy.zeros( (nxf_geo, nxf_geo, 6) )
    LonG = numpy.zeros( (nxf_geo, nxf_geo, 6) )
    
    if ornt == 'c':
        for n in range(6):
            LatG[:, :, n], LonG[:, :, n] = calc_geocoords_centerpole(lx, ly, n + 1)
        
        LatG = permutetiles(LatG, 2)
        LonG = permutetiles(LonG, 2)
    
    elif ornt == 'v':
        for n in range(6): 
            LatG[:, :, n], LonG[:, :, n] = calc_geocoords_cornerpole(lx, ly, n + 1)
    
    if nratio != 1:
        Q = numpy.concatenate((q[:-1], -q[::-1]))
        qx = Q[1 + nratio // 2::nratio] - Q[nratio // 2 - 1:-2:nratio]
        
        QX, QY = numpy.meshgrid(qx, qx, indexing='ij')
        del qx, Q
        
        Xf, Yf, Zf = map_lonlat2xyz(LonG, LatG)
        
        dXdx = ( Xf[1+nratio//2::nratio,nratio//2::nratio,:] - Xf[nratio//2-1:-2:nratio,nratio//2::nratio,:] ) / expand_array( QX )
        dYdx = ( Yf[1+nratio//2::nratio,nratio//2::nratio,:] - Yf[nratio//2-1:-2:nratio,nratio//2::nratio,:] ) / expand_array( QX )
        dZdx = ( Zf[1+nratio//2::nratio,nratio//2::nratio,:] - Zf[nratio//2-1:-2:nratio,nratio//2::nratio,:] ) / expand_array( QX )
        
        dXdy = ( Xf[nratio//2::nratio,1+nratio//2::nratio,:] - Xf[nratio//2::nratio,nratio//2-1:-2:nratio,:] ) / expand_array( QY )
        dYdy = ( Yf[nratio//2::nratio,1+nratio//2::nratio,:] - Yf[nratio//2::nratio,nratio//2-1:-2:nratio,:] ) / expand_array( QY )
        dZdy = ( Zf[nratio//2::nratio,1+nratio//2::nratio,:] - Zf[nratio//2::nratio,nratio//2-1:-2:nratio,:] ) / expand_array( QY )
        
        Q11 = dXdx * dXdx + dYdx * dYdx + dZdx * dZdx
        Q22 = dXdy * dXdy + dYdy * dYdy + dZdy * dZdy
        Q12 = dXdx * dXdy + dYdx * dYdy + dZdx * dZdy
        
        del Xf, Yf, Zf, QX, QY
    
    else:
        Q11 = Q12 = Q22 = 0.
    
    latG = LatG[::nratio,::nratio,:]
    lonG = LonG[::nratio,::nratio,:]
    
    if nratio == 1:
        latC = ( latG[:-1,:-1,:] + latG[1:,:-1,:] + latG[:-1,1:,:] + latG[1:,1:,:] ) / 4
        lonC = ( lonG[:-1,:-1,:] + lonG[1:,:-1,:] + lonG[:-1,1:,:] + lonG[1:,1:,:] ) / 4
    else:
        latC = LatG[nratio//2::nratio,nratio//2::nratio,:]
        lonC = LonG[nratio//2::nratio,nratio//2::nratio,:]
    
    del LatG, LonG
    
    if nratio != 1:
        Xlon = -numpy.sin(lonC) 
        Ylon = +numpy.cos(lonC) 
        Zlon = 0. * lonC
        
        TUu =  ( dXdx * Xlon + dYdx * Ylon + dZdx * Zlon )
        TVu = -( dXdy * Xlon + dYdy * Ylon + dZdy * Zlon )
        
        Xlat = -numpy.sin(latC) * numpy.cos(lonC) 
        Ylat = -numpy.sin(latC) * numpy.sin(lonC) 
        Zlat = +numpy.cos(latC)
        
        TUv = -( dXdx * Xlat + dYdx * Ylat + dZdx * Zlat )
        TVv =  ( dXdy * Xlat + dYdy * Ylat + dZdy * Zlat )
        
        det = numpy.sqrt(TUu * TVv - TUv * TVu)
        
        TUu /= det
        TUv /= det
        TVu /= det
        TVv /= det
        
        del dXdx, dXdy, dYdx, dYdy, dZdx, dZdy
    
    XG, YG, ZG = map_lonlat2xyz(lonG, latG)
    
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'grid.info'), 'w') as fout:
        fout.write(f"nx={nx-1}\n")
        fout.write(f"nratio={nratio}\n")
        fout.write(f"nxf={(nx-1)*nratio}\n")
        fout.write(f"mapping={method}\n")
        fout.write(f"orientation={ornt}\n")
    
    idx = slice(0,nx-1)

    lonG1, latG1 = lonG[idx,idx,:].copy(), latG[idx,idx,:].copy()
    
    dxg1, dyg1 = dxg[idx,idx].copy(), dyg[idx,idx].copy()
    dxf1, dyf1 = dxf[idx,idx].copy(), dyf[idx,idx].copy()
    dxc1, dyc1 = dxc[idx,idx].copy(), dyc[idx,idx].copy()
    dxv1, dyu1 = dxv[idx,idx].copy(), dyu[idx,idx].copy()
    
    Ec1, Eu1, Ev1, Ez1 = Ec[idx,idx].copy(), Eu[idx,idx].copy(), Ev[idx,idx].copy(), Ez[idx,idx].copy()
    
    convertMITgrid( numpy.degrees( lonC, ), 
                    numpy.degrees( latC, ), 
                    numpy.degrees( lonG1 ), 
                    numpy.degrees( latG1 ),
                    Rsphere * expand_array(dxc1),  
                    Rsphere * expand_array(dyc1),
                    Rsphere * expand_array(dxg1),  
                    Rsphere * expand_array(dyg1),
                    Rsphere * expand_array(dxf1),  
                    Rsphere * expand_array(dyf1),
                    Rsphere * expand_array(dxv1),  
                    Rsphere * expand_array(dyu1),
                    Rsphere**2 * expand_array(Ec1), 
                    Rsphere**2 * expand_array(Eu1),
                    Rsphere**2 * expand_array(Ev1), 
                    Rsphere**2 * expand_array(Ez1),
                    path, 
                    prec, 
                    machine )
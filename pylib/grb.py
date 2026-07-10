import sys
import numpy
import array

class grd_build:
    def __init__(self, path2cs, nx, Rsphere, nratio=4, method='conf', ornt='c', prec='d', machine='big'):
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
                        *[ Rsphere**2 * expand_array(arr[idx, idx]) for arr in (Ec, Eu, Ev, Ez) ],
                        path2cs, prec, machine )

def pad_array(a):
    return numpy.pad( a, ((0, 1), (0, 1), (0, 0)), constant_values=0 )

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

def map_lonlat2xyz(lon, lat):
    return numpy.cos(lat) * numpy.cos(lon), numpy.cos(lat) * numpy.sin(lon), numpy.sin(lat)

## Old
def map_xyz2lonlat(x, y, z):
    a  = numpy.sqrt( x**2 + y**2 )
    z1 = z.copy()
    
    kii, mii = numpy.where( a == 0. )
    
    z1[kii,mii] = z1[kii,mii] * numpy.inf
    a[kii,mii]  = 1.
    
    lat = numpy.arctan( z1 / a )
    
    x1 = x.copy()
    y1 = y.copy()
    
    kii, mii = numpy.where( x == 0. )
    
    y1[kii,mii] = numpy.inf
    x1[kii,mii] = 1.
    
    lon = numpy.arctan( y1 / x1 )
    
    kx, mx = numpy.where( x < 0. )
    
    for i, item in enumerate(kx):
        if y[kx[i],mx[i]] >= 0.:
            lon[kx[i],mx[i]] = numpy.pi + lon[kx[i],mx[i]]
    
    kx, mx = numpy.where( x <= 0. )
    
    for i, item in enumerate(kx):
        if y[kx[i],mx[i]] < 0.:
           lon[kx[i],mx[i]] = -numpy.pi + lon[kx[i],mx[i]]
    
    return lon,lat

def map_xy2xyz(xi, yi):
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
    nxf      = E.shape[0] + 1
    nx       = (nxf-1) // nratio + 1
    n_blocks = nx - 1
    n2       = nratio // 2
    
    def sum_blocks(mat, rows, cols):
        pad_r = max( 0, rows * nratio - mat.shape[0] )
        pad_c = max( 0, cols * nratio - mat.shape[1] )
        
        if pad_r > 0 or pad_c > 0:
            mat = numpy.pad( mat, ((0, pad_r), (0, pad_c)), mode='wrap' )
            
        return mat[:rows*nratio, :cols*nratio].reshape(rows, nratio, cols, nratio).sum(axis=(1, 3))
    
    Ec = sum_blocks( E, n_blocks, n_blocks)
    Ev = sum_blocks( numpy.roll( E, n2, axis=1 ), n_blocks, nx )
    Ez = sum_blocks( numpy.roll( E, (n2,n2), axis=(0, 1)), nx, nx)
    
    for mat in ( Ec, Ez, Ev ):
        mat[:] = ( mat + mat[::-1,:]) / 4 + ( mat + mat[:,::-1] ) / 4
    
    Ez[ 0, 0] *= 0.75
    Ez[ 0,-1] *= 0.75
    Ez[-1, 0] *= 0.75
    Ez[-1,-1] *= 0.75
    
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

## Is for sure buggy
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
    
    lx1, ly1, lz1 = rotate_about_axis( lx1, ly1, lz1, -numpy.pi / 4, 'z' )
    lx1, ly1, lz1 = rotate_about_axis( lx1, ly1, lz1, numpy.arctan( numpy.sqrt(2.) ), 'y' )
    
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
    a.astype( prec ).transpose().byteswap( sys.byteorder != machine ).tofile( fout )

def write_tile(file_out, a, prec, machine):
    with open(file_out + '.bin', 'wb') as fout:
        a.astype( prec ).transpose(2,1,0).byteswap( sys.byteorder != machine ).tofile(fout)

def convertMITgrid(xc, yc, xg, yg, dxc, dyc, dxg, dyg, dxf, dyf, dxv, dyu, rac, raw, ras, raz, newdir, prec, machine):
    xc,  yc,  xg,  yg, \
    dxc, dyc, dxg, dyg, \
    dxf, dyf, dxv, dyu, \
    rac, raw, ras, raz  = map( pad_array, (  xc,  yc,  xg,  yg, \
                                            dxc, dyc, dxg, dyg, \
                                            dxf, dyf, dxv, dyu, \
                                            rac, raw, ras, raz  ) )
    
    for arr in (xg, yg, dxv, dyu, raz):
        arr[-1,-1,:] = numpy.nan
    
    for arr, (t1, t2) in ((xg, (0, 3)), (yg, (2, 5)), (raz, (0, 3))):
        arr[ 0,-1,[0,2,4]] = arr[0,0,t1]
        arr[-1, 0,[1,3,5]] = arr[0,0,t2]
        
        arr[-1, :,[0,2,4]] = arr[0,:   ,[1, 3, 5]]
        arr[ :,-1,[0,2,4]] = arr[0,::-1,[2, 4, 0]].transpose()
        arr[-1, :,[1,3,5]] = arr[::-1,0,[3, 5, 1]].transpose()
        arr[ :,-1,[1,3,5]] = arr[:,0,[2, 4, 0]]
    
    dxv[ 0,-1,[0,2,4]] = dxv[0,0,0]
    dxv[-1, 0,[1,3,5]] = dxv[0,0,3]
    dyu[ 0,-1,[0,2,4]] = dxv[0,0,0]
    dyu[-1, 0,[1,3,5]] = dxv[0,0,3]
    
    for u, v in ((dxv, dyu), (dyu, dxv)):
        u[-1, :,[0, 2, 4]] = u[0, :, [1, 3, 5]]
        u[ :,-1,[0, 2, 4]] = v[0, ::-1, [2, 4, 0]].transpose()
        u[-1, :,[1, 3, 5]] = v[::-1, 0, [3, 5, 1]].transpose()
        u[ :,-1,[1, 3, 5]] = u[:, 0, [2, 4, 0]]
    
    for arr in (dxc, raw, dyg):
        arr[-1,:-1,:] = numpy.nan
    
    for arr in (dyc, ras, dxg):
        arr[:-1,-1,:] = numpy.nan
    
    for x_arr, y_arr in ((dxc, dyc), (raw, ras), (dyg, dxg)):
        x_arr[-1, :, [0, 2, 4]]   = x_arr[0, :, [1, 3, 5]]
        x_arr[-1, :-1, [1, 3, 5]] = y_arr[-2::-1, 0, [3, 5, 1]].transpose()
        y_arr[:, -1, [1, 3, 5]]   = y_arr[:, 0, [2, 4, 0]]
        y_arr[:-1, -1, [0, 2, 4]] = x_arr[0, -2::-1, [2, 4, 0]].transpose()
    
    all_grids = [xc, yc, dxf, dyf, rac, xg, yg, dxv, dyu, raz, dxc, dyc, raw, ras, dxg, dyg]
    
    for i in range(6):
        with open(f"{newdir}/tile{i+1:03d}.mitgrid", 'wb') as fout:
            for grid in all_grids:
                write_blocks( fout, grid[:,:,i], prec, machine )
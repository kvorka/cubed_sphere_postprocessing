import sys
import numpy
import array

class grd_build:
    def __init__(self, path2cs, nx, Rsphere):
        gengrid( Rsphere=Rsphere, nx=nx, path=path2cs )

def pad_array(a):
    nx = numpy.shape(a)[0]
    ny = numpy.shape(a)[1]
    nt = numpy.shape(a)[2]
    
    if ( nx != ny or nt != 6 ):
        print('Wrong size for tiles nx != ny; or number of tiles != 6')
        exit('check array size in the source code')
    
    a1 = numpy.zeros( (nx+1,ny+1,nt) )
    a1[:-1,:-1,:] = a[:,:,:]
    
    return a1

def write_blocks(fout, a, prec, machine):
    for i in range( a.shape[1] ):
        float_array = array.array( prec, a[:,i] )
        
        if sys.byteorder != machine:
            float_array.byteswap()
        
        float_array.tofile( fout )

def convertMITgrid(xc, yc, xg, yg, dxc, dyc, dxg, dyg, dxf, dyf, dxv, dyu, rac, raw, ras, raz, newdir, prec, machine):
    nx = xc.shape[0]
    ny = xc.shape[1]
    nt = xc.shape[2]
    
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
    
    dyu[0,-1,[0,2,4]] = dxv[0,0,0].copy()
    dyu[-1,0,[1,3,5]] = dxv[0,0,3].copy()
    
    dxv[-1,:,[0,2,4]] = dxv[0,:,[1,3,5]]
    
    dxv[:,-1,[0,2,4]] = dyu[0,::-1,[2,4,0]].transpose().copy()
    dxv[-1,:,[1,3,5]] = dyu[::-1,0,[3,5,1]].transpose().copy()
    
    dxv[:,-1,[1,3,5]] = dxv[:,0,[2,4,0]]
    dyu[-1,:,[0,2,4]] = dyu[0,:,[1,3,5]]
    
    dyu[:,-1,[0,2,4]] = dxv[0,::-1,[2,4,0]].transpose().copy()
    dyu[-1,:,[1,3,5]] = dxv[::-1,0,[3,5,1]].transpose().copy()
    
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
    dxc[-1,:-1,[1,3,5]] = dyc[-2::-1,0,[3,5,1]].transpose().copy()
    dyc[:,-1,[1,3,5]]   = dyc[:,0,[2,4,0]]
    dyc[:-1,-1,[0,2,4]] = dxc[0,-2::-1,[2,4,0]].transpose().copy()

    raw[-1,:,[0,2,4]]   = raw[0,:,[1,3,5]]
    raw[-1,:-1,[1,3,5]] = ras[-2::-1,0,[3,5,1]].transpose().copy()
    ras[:,-1,[1,3,5]]   = ras[:,0,[2,4,0]]
    ras[:-1,-1,[0,2,4]] = raw[0,-2::-1,[2,4,0]].transpose().copy()

    dyg[-1,:,[0,2,4]]   = dyg[0,:,[1,3,5]]
    dyg[-1,:-1,[1,3,5]] = dxg[-2::-1,0,[3,5,1]].transpose().copy()
    dxg[:,-1,[1,3,5]]   = dxg[:,0,[2,4,0]]
    dxg[:-1,-1,[0,2,4]] = dyg[0,-2::-1,[2,4,0]].transpose().copy()
    
    for i in range(6):
        fout = open(newdir+'/'+'tile'+"%03d"%(i+1)+'.mitgrid', 'wb')
        
        write_blocks(fout,  xc[:,:,i], prec, machine)
        write_blocks(fout,  yc[:,:,i], prec, machine)
        write_blocks(fout, dxf[:,:,i], prec, machine)
        write_blocks(fout, dyf[:,:,i], prec, machine)
        write_blocks(fout, rac[:,:,i], prec, machine)
        write_blocks(fout,  xg[:,:,i], prec, machine)
        write_blocks(fout,  yg[:,:,i], prec, machine)
        write_blocks(fout, dxv[:,:,i], prec, machine)
        write_blocks(fout, dyu[:,:,i], prec, machine)
        write_blocks(fout, raz[:,:,i], prec, machine)
        write_blocks(fout, dxc[:,:,i], prec, machine)
        write_blocks(fout, dyc[:,:,i], prec, machine)
        write_blocks(fout, raw[:,:,i], prec, machine)
        write_blocks(fout, ras[:,:,i], prec, machine)
        write_blocks(fout, dxg[:,:,i], prec, machine)
        write_blocks(fout, dyg[:,:,i], prec, machine)
        
        fout.close()

def write_tile(file_out, a, prec, machine):
    fout = open(file_out+'.bin','wb')
    
    for n in range( a.shape[2] ):
        for i in range ( a.shape[1] ):
            float_array = array.array( prec, a[:,i,n] )
            
            if sys.byteorder != machine:
                float_array.byteswap()
            
            float_array.tofile(fout)
    
    fout.close()

def write_tiles(file_out, a, prec, machine):
    for n in range( a.shape[2] ):
        fout = open(file_out+'.'+"%03d"%(n+1)+'.bin','wb')
        
        for i in range( a.shape[1] ):
            float_array = array.array( prec, a[:,i,n] )
            
            if sys.byteorder != machine:
                float_array.byteswap()
            
            float_array.tofile(fout)
        
        fout.close()

def expand(a):
    a1 = numpy.zeros( ( a.shape[0], a.shape[1], 6 ) )
    
    for i in range(6):
        a1[:,:,i] = a[:,:]
    
    return a1

def permutetiles(b, n):
    c = b.copy()
    a = numpy.zeros( (b.shape[0], b.shape[1], b.shape[2]) )
    
    for k in range(n):
        a[:,:,0] = c[:,:,1]
        a[:,:,1] = c[::-1,:,3].transpose()
        a[:,:,2] = c[::-1,:,2].transpose()
        a[:,:,3] = c[:,:,4]
        a[:,:,4] = c[:,::-1,0].transpose()
        a[:,:,5] = c[:,::-1,5].transpose()
        
        c, a = a, c
    
    return c

def rescale_coordinate(q, method='conf'):
    nxf = numpy.size( q )
    nx  = round( (nxf-1)/2 )
    dxg = conf_d( q )
    
    if ( method == 'q=0' ):
        D = numpy.max( dxg, axis=1 )
    elif ( method == 'q=1/2' ):
        D = dxg[:,round(nx/2)].copy()
    elif ( method == 'q=78' ):
        D = dxg[:,round(nx/8)].copy()
    elif ( method == 'q=1' ):
        D = numpy.min( dxg, axis=1 )
    elif ( method == 'q=i3' ):
        D = dxg[:,2].copy()
    else:
        D = dxg[:,0]
    
    s  = numpy.cumsum( numpy.concatenate( ( [0.], D ) ) )
    dS = numpy.max(s)/(nxf-1) + 0. * D
    S  = numpy.cumsum( numpy.concatenate( ( [0.], dS ) ) )
    
    if ( method == 'conf' ):
        Q = q.copy()
    elif method in [ 'q=0', 'q=1', 'q=1/2', 'q=7/8', 'q=i3']:
        Q = numpy.interp(S, s, q)
    elif ( method == 'tan' ):
        Q = numpy.tan( 2./3. * q ) / numpy.tan( 2./3. * numpy.max( numpy.abs(q) ) )
    elif ( method == 'tan2' ):
        Q = numpy.tan( 1./5. * q ) / numpy.tan( 1./5. * numpy.max( numpy.abs(q) ) )
    elif ( method == 'new' ):
        dq    = numpy.ones(nxf)
        dq[0] = 0.
        dq[1] = 1.5
        dq    = dq / numpy.sum(dq) * ( q[-1:] - q[0] )
        Q     = q[0] + numpy.cumsum(dq)
    else:
       print('Unknown method')
    
    return Q

def map_lonlat2xyz(lon, lat):
    a = numpy.cos(lat)

    z =     numpy.sin(lat)
    x = a * numpy.cos(lon)
    y = a * numpy.sin(lon)
    
    return x, y, z

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

def Acoeffs():
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
    
    return A

def Bcoeffs():
    B = [ 0.67698822171341, 0.11847295533659, 0.05317179075349, \
          0.02965811274764, 0.01912447871071, 0.01342566129383, \
          0.00998873721022, 0.00774869352561, 0.00620347278164, \
          0.00509011141874, 0.00425981415542, 0.00362309163280, \
          0.00312341651697, 0.00272361113245, 0.00239838233411, \
          0.00213002038153, 0.00190581436893, 0.00171644267546, \
          0.00155493871562, 0.00141600812949, 0.00129556691848, \
          0.00119042232809, 0.00109804804853, 0.00101642312253, \
          0.00094391466713, 0.00087919127990, 0.00082115825576, \
          0.00076890854394, 0.00072168520663, 0.00067885239089  ]
    
    return B

def WofZ(z):
    A = Acoeffs()
    w = 0.
    
    for j in range(29,-1,-1):
        w = ( w + A[j] ) * z
    
    return w

def map_xy2xyz(xi, yi):
    if numpy.size(xi) != numpy.size(yi):
       raise SystemExit('Arguments x,y should be same size')
    
    xc = numpy.abs(xi)
    yc = numpy.abs(yi)
    
    kx, mx = numpy.where( xi < 0. )
    ky, my = numpy.where( yi < 0. )
    
    kxy, mxy = numpy.where( yc > xc )
    
    x1 = xc.copy()
    y1 = yc.copy()
    
    xc = 1 - xc
    yc = 1 - yc
    
    for i, item in enumerate(kxy):
        xc[kxy[i],mxy[i]] = 1 - y1[kxy[i],mxy[i]]
        yc[kxy[i],mxy[i]] = 1 - x1[kxy[i],mxy[i]]
    
    zi = ( ( xc + 1j * yc ) / 2. )**4
    
    W = WofZ(zi)
    
    thrd = numpy.float64( 1./3. )
    i3   = 1j**thrd
    
    ra = numpy.sqrt(3.)-1.
    cb = 1j-1.
    cc = ra * cb / 2.
    
    W = i3 * ( W * 1j )**thrd
    W = ( W - ra ) / ( cb + cc * W )
    
    x1 = numpy.real(W)
    y1 = numpy.imag(W)
    
    H = 2. / (1. + x1**2 + y1**2 )
    
    x1 = x1 * H
    y1 = y1 * H
    z1 = H - 1.
    
    t1 = x1.copy()
    
    for i, item in enumerate(kxy):
        x1[kxy[i],mxy[i]] = y1[kxy[i],mxy[i]]
        y1[kxy[i],mxy[i]] = t1[kxy[i],mxy[i]]
    
    for i, item in enumerate(kx):
        x1[kx[i],mx[i]] = -x1[kx[i],mx[i]]
    
    for i, item in enumerate(ky):
        y1[ky[i],my[i]] = -y1[ky[i],my[i]]
    
    x1[ numpy.where( xi==0. ) ] = 0.
    y1[ numpy.where( yi==0. ) ] = 0.
    
    if ( numpy.size( numpy.where( xi == 0. ) ) / 2 == xi.size ):
        x1 = x1.reshape( xi.shape[0], xi.shape[1], order='F')
    
    if ( numpy.size( numpy.where( yi == 0. ) ) / 2 == yi.size ):
        y1 = y1.reshape( yi.shape[0], yi.shape[1], order='F')
    
    return x1, y1, z1

def angle_between_vectors(vec1, vec2):
    if numpy.ndim(vec1) == 3:
       vprod = vec1[:,:,0] * vec2[:,:,0] + vec1[:,:,1] * vec2[:,:,1] + vec1[:,:,2] * vec2[:,:,2]
       nrm1  = vec1[:,:,0]**2 + vec1[:,:,1]**2 + vec1[:,:,2]**2
       nrm2  = vec2[:,:,0]**2 + vec2[:,:,1]**2 + vec2[:,:,2]**2
    else:
       exit('Number of dimensions are not coded for')
    
    slope = vprod / numpy.sqrt( nrm1 * nrm2 )
    
    slope[ numpy.where( slope >  1 ) ] =  1.
    slope[ numpy.where( slope <- 1 ) ] = -1.
    
    return numpy.arccos( slope )

def coord2vector(x, y, z):
    nx = x.shape[0]
    ny = x.shape[1]
    
    vec = numpy.zeros( (nx, ny, 3) )
    
    vec[:,:,0] = x.copy()
    vec[:,:,1] = y.copy()
    vec[:,:,2] = z.copy()
    
    return vec

def conf_d(qq):
    nx  = numpy.size( qq )
    nxf = 2*(nx-1)+1
    
    q       = numpy.zeros( nxf )
    q[::2]  = qq.copy()
    
    q[1::2] = ( q[::2][:-1] + q[1::2] ) / 2.
    
    lx, ly = numpy.meshgrid( q, q )
    lx = lx.transpose()
    ly = ly.transpose()
    
    lx1, ly1, lz1 = map_xy2xyz( lx, ly )
    
    xg = lx1[::2,::2].copy()
    yg = ly1[::2,::2].copy()
    zg = lz1[::2,::2].copy()
    
    vertices = coord2vector( xg, yg, zg )

    return angle_between_vectors( vertices[:-1,:-1,:], vertices[1::,:-1,:] )

def reduce_E(E, nratio):
    nxf = E.shape[0] + 1
    
    if nratio == 1:
        Ec = E.copy()
        Ev = ( numpy.append( [ E[:,-1] ],  E , axis=0 ) + numpy.append( E,  [ E[:,0]  ], axis=0 ) ) / 2.
        Ez = ( numpy.append( [ Ev[-1,:] ], Ev, axis=0 ) + numpy.append( Ev, [ Ev[0,:] ]         ) ) / 2.
    
    elif nratio%2 == 0:
        nx = (nxf-1) // nratio + 1
        
        if nx != numpy.floor(nx):
           print('nx = '+str(nx))
           exit('nxf/nratio must be an integer')
        
        Ec = numpy.zeros( (nx-1, nx-1), order='F' )
        Ev = numpy.zeros( (nx-1, nx  ), order='F' )
        Ez = numpy.zeros( (nx  , nx  ), order='F' )
        
        kg = numpy.arange( 0, nxf-1, nratio ) - 1  
        kc = numpy.append( nxf-nratio//2-1, numpy.arange( nratio//2, nxf, nratio ) ) - 1
        
        for m in range(nratio):
            kg = numpy.mod(kg+1,nxf-1)
            kc = numpy.mod(kc+1,nxf-1)
            jg = numpy.arange(0,nxf-1,nratio)-1
            jc = numpy.append(nxf-nratio//2-1,numpy.arange(nratio//2,nxf,nratio))-1
            
            for n in range(nratio):
                jg = numpy.mod(jg+1,nxf-1)
                jc = numpy.mod(jc+1,nxf-1)
                
                Ec = Ec + numpy.take( E, jg, axis=0 ).take( kg, axis=1 )
                Ev = Ev + numpy.take( E, jg, axis=0 ).take( kc, axis=1 ) 
                Ez = Ez + numpy.take( E, jc, axis=0 ).take( kc, axis=1 ) 
        
        Ec = ( Ec + Ec[::-1,:] ) / 2.
        Ec = ( Ec + Ec[:,::-1] ) / 2.
        Ez = ( Ez + Ez[::-1,:] ) / 2.
        Ez = ( Ez + Ez[:,::-1] ) / 2.
        Ev = ( Ev + Ev[::-1,:] ) / 2.
        Ev = ( Ev + Ev[:,::-1] ) / 2.
        
        Ez[0,  0] = Ez[0,  0] * ( 3. / 4. )
        Ez[0, -1] = Ez[0, -1] * ( 3. / 4. )
        Ez[-1, 0] = Ez[-1, 0] * ( 3. / 4. )
        Ez[-1,-1] = Ez[-1,-1] * ( 3. / 4. )
        
    elif nratio%2 != 0:
        print('nratio = '+str(nratio))
        exit('nratio must be multiple of 2 to be able to reduce gid')
    
    return Ec, Ez, Ev

def reduce_dx(dx, nratio):
    if nratio == 1:
        dxg = dx.copy()
        dxf = ( dx[:,:-1] + dx[:,1:] ) / 2.
        
        dxv = ( numpy.append( [ dx[-1,:] ], dx , axis=0 ) + numpy.append( dx , [ dx[0,:] ], axis=0 ) ) / 2.    
        dxc = ( numpy.append( [ dxf[-1,:]], dxf, axis=0 ) + numpy.append( dxf, [ dxf[0,:]], axis=0 ) ) / 2. 
    
    elif nratio%2 == 0:
        nxf = dx.shape[1]
         
        kg = numpy.arange( 0, nxf, nratio ) 
        kc = numpy.arange( nratio//2, nxf, nratio) 
        jg = numpy.arange( 0, nxf-1, nratio ) 
        jc = numpy.append( nxf-nratio//2-1, numpy.arange( nratio//2, nxf, nratio ) )
        
        dxg = numpy.take( dx, jg, axis=0 ).take( kg, axis=1 )
        dxf = numpy.take( dx, jg, axis=0 ).take( kc, axis=1 )
        dxc = numpy.take( dx, jc, axis=0 ).take( kc, axis=1 )
        dxv = numpy.take( dx, jc, axis=0 ).take( kg, axis=1 )
        
        for n in range(1,nratio):
            jg = numpy.mod(jg+1,nxf-1)
            jc = numpy.mod(jc+1,nxf-1)
            
            dxg = dxg + numpy.take( dx, jg, axis=0 ).take( kg, axis=1 )
            dxf = dxf + numpy.take( dx, jg, axis=0 ).take( kc, axis=1 )
            dxc = dxc + numpy.take( dx, jc, axis=0 ).take( kc, axis=1 )
            dxv = dxv + numpy.take( dx, jc, axis=0 ).take( kg, axis=1 )
            
            dxf = ( dxf + dxf[::-1,:] ) / 2.
            dxf = ( dxf + dxf[:,::-1] ) / 2.
            dxg = ( dxg + dxg[::-1,:] ) / 2.
            dxg = ( dxg + dxg[:,::-1] ) / 2.
            dxc = ( dxc + dxc[::-1,:] ) / 2.
            dxc = ( dxc + dxc[:,::-1] ) / 2.
            dxv = ( dxv + dxv[::-1,:] ) / 2.
            dxv = ( dxv + dxv[:,::-1] ) / 2.
    
    elif nratio%2 != 0:
        print('nratio = ',+str(nratio))
        exit('nratio must be multiple of 2 to be able to reduce grid')

    return dxg, dxc, dxf, dxv

def plane_normal(P1, P2):
    plane = numpy.zeros( P1.shape )

    plane[:,:,0] = P1[:,:,1] * P2[:,:,2] - P1[:,:,2] * P2[:,:,1]
    plane[:,:,1] = P1[:,:,2] * P2[:,:,0] - P1[:,:,0] * P2[:,:,2]
    plane[:,:,2] = P1[:,:,0] * P2[:,:,1] - P1[:,:,1] * P2[:,:,0]
    
    mag = numpy.sqrt( plane[:,:,0]**2 + plane[:,:,1]**2 + plane[:,:,2]**2 )
    
    plane[:,:,0] = plane[:,:,0] / mag
    plane[:,:,1] = plane[:,:,1] / mag
    plane[:,:,2] = plane[:,:,2] / mag
    
    return plane

def excess_of_quad(V1, V2, V3, V4):
    plane1 = plane_normal(V1,V2)
    plane2 = plane_normal(V2,V3)
    plane3 = plane_normal(V3,V4)
    plane4 = plane_normal(V4,V1)

    angle12 = numpy.pi - angle_between_vectors(plane2,plane1)
    angle23 = numpy.pi - angle_between_vectors(plane3,plane2)
    angle34 = numpy.pi - angle_between_vectors(plane4,plane3)
    angle41 = numpy.pi - angle_between_vectors(plane1,plane4)
    
    return angle12 + angle23 + angle34 + angle41 - 2 * numpy.pi

def calc_fvgrid(lx,ly):
    nxf = lx.shape[0]
    
    lx1, ly1, lz1 = map_xy2xyz( lx, ly )
    vertices      = coord2vector( lx1, ly1, lz1 )
    
    dx = angle_between_vectors( vertices[:-1,:,:], vertices[1::,:,:] )
    dy = angle_between_vectors( vertices[:,:-1,:], vertices[:,1::,:] )
    E  = excess_of_quad( vertices[:-1,:-1,:], vertices[1::,:-1,:], vertices[1::,1::,:], vertices[:-1,1::,:] )
    
    dx = ( dx + dy.transpose() ) / 2.
    dy = dx.transpose()
    E  = ( E + E.transpose() ) / 2.
    
    for j in range(1,nxf):
        dx[:-nxf+j,j] = dy[j,:-nxf+j].transpose()
    
    for j in range (0,nxf-1):
        dy[:-(nxf-1)+j,j] = dx[j,:-(nxf-1)+j].transpose()
    
    for j in range (1,nxf-1):
        E[:-nxf+j,j] = E[j,:-nxf+j].transpose()
    
    return dx, dy, E

def rotate_about_zaxis(lx, ly, lz, angle):
    s = numpy.sin( angle )
    c = numpy.cos( angle )
    
    if c < 1.e-9:
       c = 0.
       s = numpy.sign( s )
    
    x = c * lx - s * ly
    y = s * lx + c * ly
    z = lz.copy()
    
    return x, y, z

def rotate_about_yaxis(lx, ly, lz, angle):
    s = numpy.sin( angle )
    c = numpy.cos( angle )
    
    if c < 1.e-9:
       c = 0.
       s = numpy.sign( s )
    
    x = +c * lx + s * lz
    y = ly.copy()
    z = -s * lx + c * lz
    
    return x, y, z

def rotate_about_xaxis(lx, ly, lz, angle):
    s = numpy.sin( angle )
    c = numpy.cos( angle )
    
    if c < 1.e-9:
       c = 0.
       s = numpy.sign( s )
    
    x = lx.copy()
    y = c * ly - s * lz
    z = s * ly + c * lz
    
    return x, y, z

def calc_geocoords_cornerpole(lx, ly, tile):
    nx  = lx.shape[0]
    nxf = 2 * nx - 1
    
    lx1, ly1, lz1 = map_xy2xyz(lx, ly)
    
    lx1 = numpy.concatenate( ( lx1[:-1,:], -lx1[nx-1::-1,:] ), axis=0 ) 
    lx1 = numpy.concatenate( ( lx1[:,:-1],  lx1[:,nx-1::-1] ), axis=1 )
    
    ly1 = numpy.concatenate( ( ly1[:-1,:],  ly1[nx-1::-1,:] ), axis=0 ) 
    ly1 = numpy.concatenate( ( ly1[:,:-1], -ly1[:,nx-1::-1] ), axis=1 )
    
    lx1 = ( lx1 + ly1.transpose() ) / 2. 
    ly1 = lx1.transpose()
    lz1 = ( lz1 + lz1.transpose() ) / 2.
    
    lz1 = numpy.concatenate( ( lz1[:-1,:], lz1[nx-1::-1,:] ), axis=0 ) 
    lz1 = numpy.concatenate( ( lz1[:,:-1], lz1[:,nx-1::-1] ), axis=1 )
    
    lx1, ly1, lz1 = rotate_about_zaxis( lx1, ly1, lz1, -numpy.pi/4. )
    lx1, ly1, lz1 = rotate_about_yaxis( lx1, ly1, lz1,  numpy.arctan( numpy.sqrt(2.) ) )
    
    lx1 = ( lx1 + lx1.transpose() ) / 2.
    ly1 = ( ly1 - ly1.transpose() ) / 2.
    lz1 = ( lz1 + lz1.transpose() ) / 2.

    lonP, latP = map_xyz2lonlat( lx1, ly1, lz1 )
    
    for j in range(nx):
        lonP[j,j] = 0.
    
    lonP = ( lonP - lonP.transpose() ) / 2.
    latP = ( latP + latP.transpose() ) / 2.
    
    latP =  latP[:,::-1]
    lonP = -lonP[:,::-1]
    
    if tile == 1:
        lat =  latP[::-1,:]
        lon = -lonP[::-1,:] - 2./3. * numpy.pi
    elif tile == 2:
        lat = latP
        lon = lonP
    elif tile == 3:
        lat =  latP[:,::-1]
        lon = -lonP[:,::-1] + 2./3. * numpy.pi
    elif tile == 4:
        lat = -latP[::-1,:]
        lon =  lonP[::-1,:] + 1./3. * numpy.pi
    elif tile == 5:
        lat = -latP[::-1,::-1]
        lon = -lonP[::-1,::-1] + numpy.pi
    elif tile == 6:
        lat = -latP[:,::-1]
        lon =  lonP[:,::-1] - 1./3. * numpy.pi
    else:
        print('tile = '+str(tile))
        exit('Illegal value for tile, should be within [0,5]')
    
    lon[ numpy.where( lon >   numpy.pi ) ] -= 2 * numpy.pi
    lon[ numpy.where( lon <= -numpy.pi ) ] += 2 * numpy.pi
    
    return lat, lon

def calc_geocoords_centerpole(lx, ly, tile):
    nx  = lx.shape[0]
    nxf = 2 * nx - 1
    
    lx1, ly1, lz1 = map_xy2xyz( lx, ly )
    lonP, latP    = map_xyz2lonlat( lx1, ly1, lz1 )
    
    lonP[numpy.where( lonP >= numpy.pi ) ] -= 2. * numpy.pi
    
    latP = ( latP                    + latP.transpose() ) / 2.
    lonP = ( lonP - 3./2. * numpy.pi - lonP.transpose() ) / 2.
    
    for j in range(nx):
        lonP[j,j] = -3./4. * numpy.pi
    
    latP = numpy.concatenate( ( latP[:-1,:],           latP[nx-1::-1,:] ), axis=0 )
    latP = numpy.concatenate( ( latP[:,:-1],           latP[:,nx-1::-1] ), axis=1 )
    lonP = numpy.concatenate( ( lonP[:-1,:], -numpy.pi-lonP[nx-1::-1,:] ), axis=0 )
    lonP = numpy.concatenate( ( lonP[:,:-1],          -lonP[:,nx-1::-1] ), axis=1 )
    
    lx1, ly1, lz1 = rotate_about_xaxis( lx1, ly1, lz1, numpy.pi/2 )
    
    lonE, latE = map_xyz2lonlat( lx1, ly1, lz1 )
    
    lonE[0,:]  = -3./4. * numpy.pi
    latE[:,-1] = 0.
    latE[:,0]  = -latP[0:nx,0]
    
    latE = numpy.concatenate( (latE[:-1,:],           latE[nx-1::-1,:] ), axis=0 )
    latE = numpy.concatenate( (latE[:,:-1],          -latE[:,nx-1::-1] ), axis=1 )
    lonE = numpy.concatenate( (lonE[:-1,:], -numpy.pi-lonE[nx-1::-1,:] ), axis=0 )
    lonE = numpy.concatenate( (lonE[:,:-1],           lonE[:,nx-1::-1] ), axis=1 )
    
    if tile == 1:
        lat = latE
        lon = lonE - numpy.pi/2 
        lon[ numpy.where(lon <= -numpy.pi) ] += 2 * numpy.pi
    elif tile == 2:
        lat = latE
        lon = lonE
    elif tile == 3:
        lat = latP
        lon = lonP
    elif tile == 4:
        lat = latE[:,::-1].transpose()
        lon = lonE.transpose() + numpy.pi/2
    elif tile == 5:
        lat = latE[:,::-1].transpose()
        lon = lonE.transpose() + numpy.pi
    elif tile == 6:
        lat = -latP
        lon =  lonP[::-1,::-1].transpose()
    else:
        print('tile = '+str(tile))
        exit('Illegal value for tile, should be within [1,6]')
    
    return lat, lon

def gengrid(Rsphere, nx, path, nratio=4, method='conf', ornt='c', prec='d', machine='big'):
    if (nx%2 != 0):
        print('the resolution nx = '+str(nx))
        print('nx needs to be an even number. nx = nx-1 = '+str(nx-1)+' is used')
        sys.exit()
    
    nx  = nx+1
    nxf = nratio*(nx-1)+1
    
    q = numpy.linspace( -1., 0., int((nxf-1.)/2)+1 )
    q = rescale_coordinate( q, method )
    
    lx, ly = numpy.meshgrid( q, q )
     
    lx = lx.transpose()
    ly = ly.transpose()
    
    dx, dy, E = calc_fvgrid( lx, ly )  
    
    del(dy)
    
    dx = numpy.concatenate( ( dx,        dx[(nxf-1)//2-1::-1,:]), axis=0 )
    dx = numpy.concatenate( ( dx[:,:-1], dx[:,(nxf+1)//2-1::-1]), axis=1 )
    E  = numpy.concatenate( ( E,         E[(nxf-1)//2-1::-1,:]),  axis=0 )
    E  = numpy.concatenate( ( E,         E[:,(nxf-1)//2-1::-1]),  axis=1 )
    
    dxg, dxc, dxf, dxv = reduce_dx(dx, nratio)
    Ec, Ez, Ev = reduce_E(E, nratio)
    
    del(dx,E)
    
    dyg = dxg.transpose()
    dyc = dxc.transpose()
    dyf = dxf.transpose()
    dyu = dxv.transpose()
    Eu  = Ev.transpose()
    
    nxf = 2 * lx.shape[0] - 1
    
    LatG = numpy.zeros( (nxf,nxf,6) )
    LonG = numpy.zeros( (nxf,nxf,6) )
    
    if ornt == 'c':
       for n in range(6):
           LatG[:,:,n], LonG[:,:,n] = calc_geocoords_centerpole(lx,ly,n+1)
       
       LatG = permutetiles( LatG, 2 )
       LonG = permutetiles( LonG, 2 )
      
    elif ornt == 'v':
       for n in range(6): 
           LatG[:,:,n], LonG[:,:,n] = calc_geocoords_cornerpole(lx,ly,n+1)
    
    else:
       print('ornt = ', + ornt)
       exit('Unknown orientation')
    
    if nratio != 1:
       Q = q.copy()
       Q = numpy.concatenate( (Q[:-1],-q[::-1]) )
       
       qx = Q[1+nratio//2::nratio]-Q[nratio//2-1:-2:nratio]
       
       QX,QY=numpy.meshgrid( qx, qx )
        
       QX = QX.transpose()
       QY = QY.transpose()
       
       del(qx,Q)
       
       Xf,Yf,Zf=map_lonlat2xyz(LonG,LatG)
       dXdx=( Xf[1+nratio//2::nratio, nratio//2::nratio,:] \
             -Xf[nratio//2-1:-2:nratio, nratio//2::nratio,:] )/expand(QX)
       dYdx=( Yf[1+nratio//2::nratio ,nratio//2::nratio,:] \
             -Yf[nratio//2-1:-2:nratio, nratio//2::nratio,:] )/expand(QX)
       dZdx=( Zf[1+nratio//2::nratio ,nratio//2::nratio,:] \
             -Zf[nratio//2-1:-2:nratio, nratio//2::nratio,:] )/expand(QX)
       dXdy=( Xf[nratio//2::nratio,1+nratio//2::nratio, :] \
             -Xf[nratio//2::nratio, nratio//2-1:-2:nratio,:] )/expand(QY)
       dYdy=( Yf[nratio//2::nratio,1+nratio//2::nratio, :] \
             -Yf[nratio//2::nratio, nratio//2-1:-2:nratio,:] )/expand(QY)
       dZdy=( Zf[nratio//2::nratio,1+nratio//2::nratio, :] \
             -Zf[nratio//2::nratio, nratio//2-1:-2:nratio,:] )/expand(QY)
       Q11=dXdx*dXdx + dYdx*dYdx + dZdx*dZdx
       Q22=dXdy*dXdy + dYdy*dYdy + dZdy*dZdy
       Q12=dXdx*dXdy + dYdx*dYdy + dZdx*dZdy
       del(Xf, Yf, Zf, QX, QY)
    else:
       Q11=0.
       Q12=0.
       Q22=0.

    # Sub-sample to obtain model coordinates
    latG=LatG[::nratio,::nratio,:].copy()
    lonG=LonG[::nratio,::nratio,:].copy()
    if nratio==1:
       latC=( latG[:-1,:-1,:] + latG[1:,:-1,:] \
             +latG[:-1, 1:,:] + latG[1:,1:, :] )/4.
       lonC=( lonG[:-1,:-1,:] + lonG[1:,:-1,:] \
             +lonG[:-1,1: ,:] + lonG[1:,1:, :] )/4.
    else:
       latC=LatG[nratio//2::nratio,nratio//2::nratio,:]
       lonC=LonG[nratio//2::nratio,nratio//2::nratio,:]
    del(LatG, LonG)  # tidy up

    if nratio!=1:
       # Flow rotation tensor
       Xlon=-numpy.sin(lonC) 
       Ylon= numpy.cos(lonC) 
       Zlon= 0.*lonC
       TUu = (dXdx*Xlon + dYdx*Ylon + dZdx*Zlon)
       TVu =-(dXdy*Xlon + dYdy*Ylon + dZdy*Zlon)
       Xlat=-numpy.sin(latC) * numpy.cos(lonC) 
       Ylat=-numpy.sin(latC) * numpy.sin(lonC) 
       Zlat= numpy.cos(latC)
       TUv =-(dXdx*Xlat + dYdx*Ylat + dZdx*Zlat)
       TVv = (dXdy*Xlat + dYdy*Ylat + dZdy*Zlat)
       det = numpy.sqrt(TUu*TVv-TUv*TVu)
       TUu=TUu/det
       TUv=TUv/det
       TVu=TVu/det
       TVv=TVv/det
       del(dXdx, dXdy, dYdx, dYdy, dZdx, dZdy)

    # 3D coordinates for plotting
    XG,YG,ZG=map_lonlat2xyz(lonG,latG)
    
    fid=open(path+'/'+'grid.info','w')
    fid.write('nx='+str(nx-1)+'\n')
    fid.write('nratio='+str(nratio)+'\n')
    fid.write('nxf='+str((nx-1)*nratio)+'\n')
    fid.write('mapping='+method+'\n')
    fid.write('orientation='+ornt+'\n')
    fid.close()
    
    lonG1=lonG[0:nx-1,0:nx-1,:].copy()
    latG1=latG[0:nx-1,0:nx-1,:].copy()
    
    dxg1=dxg[0:nx-1,0:nx-1].copy()
    dyg1=dyg[0:nx-1,0:nx-1].copy()
    dxf1=dxf[0:nx-1,0:nx-1].copy()
    dyf1=dyf[0:nx-1,0:nx-1].copy()
    dxc1=dxc[0:nx-1,0:nx-1].copy()
    dyc1=dyc[0:nx-1,0:nx-1].copy()
    dxv1=dxv[0:nx-1,0:nx-1].copy()
    dyu1=dyu[0:nx-1,0:nx-1].copy()
    Ec1=Ec[0:nx-1,0:nx-1].copy()
    Eu1=Eu[0:nx-1,0:nx-1].copy()
    Ev1=Ev[0:nx-1,0:nx-1].copy()
    Ez1=Ez[0:nx-1,0:nx-1].copy()
    
    # generate tile*.mitgrid files 
    convertMITgrid(180./numpy.pi*lonC,180./numpy.pi*latC,180./numpy.pi*lonG1,180./numpy.pi*latG1, \
                               Rsphere*expand(dxc1),  Rsphere*expand(dyc1),   \
                               Rsphere*expand(dxg1),  Rsphere*expand(dyg1),   \
                               Rsphere*expand(dxf1),  Rsphere*expand(dyf1),   \
                               Rsphere*expand(dxv1),  Rsphere*expand(dyu1),   \
                               Rsphere**2*expand(Ec1),Rsphere**2*expand(Eu1), \
                               Rsphere**2*expand(Ev1),Rsphere**2*expand(Ez1), \
                               path,prec,machine)
import numpy as numpy
import re
import argparse

def parse_mitgcm_data(filename):
    params = {}
    
    patterns = { 'rotationPeriod': re.compile(r'^\s*rotationPeriod\s*=\s*([0-9.E+-]+)', re.IGNORECASE),
                 'rSphere': re.compile(r'^\s*rSphere\s*=\s*([0-9.E+-]+)', re.IGNORECASE),
                 'delR': re.compile(r'^\s*delR\s*=\s*([^,\n]+)', re.IGNORECASE) }
    
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().startswith('#') or line.strip().startswith('##'):
                continue
                
            for key, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    params[key] = match.group(1).strip()

    radius = float(params['rSphere'])
    period = float(params['rotationPeriod'])
    
    depth    = 0.0
    nz       = 0
    delr_str = params['delR']
    
    parts = delr_str.split(',')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if '*' in part:
            count_str, thickness_str = part.split('*')
            count = int(count_str)
            thickness = float(thickness_str)
            
            nz += count
            depth += count * thickness
        else:
            try:
                nz += 1
                depth += float(part)
            except ValueError:
                pass
                
    return radius, period, depth, nz

def readMITgrid(path, nx):
    lon = numpy.zeros((6 * nx, nx))
    lat = numpy.zeros((6 * nx, nx))
    
    for i in range(1, 7):
        tmp = numpy.fromfile( f'{path}tile00{i}.mitgrid', dtype='>f8').reshape((16,nx+1,nx+1))
        lon[(i-1)*nx:i*nx,:] = tmp[0,:nx,:nx]
        lat[(i-1)*nx:i*nx,:] = tmp[1,:nx,:nx]
            
    return lon, lat

def random_noise(mag, nz, nx):
    return mag * numpy.random.randn(nz, 6 * nx, nx)

def flat_bathymetry(depth, nx):
    return depth * numpy.ones((6 * nx, nx))

def gauss_ridge(height, width, lons, lats):
    return height * numpy.exp( -( numpy.degrees( numpy.arcsin( numpy.cos( numpy.radians(lats) ) * 
                                                      numpy.sin( numpy.radians(lons) ) ) ) )**2 / ( 2 * width**2 ) )

def save2file(fout, data):
    data.astype('>f8').tofile(fout)

def tidal_time_series(period, nt):
    return numpy.arange(0, period, period / nt)[:nt]

def obliquity_tides(a, omega, theta0, t, lons, lats):
    nt = len(t)
    nx = lons.shape[1]
    
    x1 = numpy.radians(lons)
    y1 = numpy.radians(lats)
    
    pot_obl = numpy.zeros((nt, 6 * nx, nx))
    
    for i in range(nt):
        pot_obl[i,:,:] = 1.5 * (omega**2) * (a**2) * theta0 * numpy.sin(2*y1) * numpy.cos(x1) * numpy.cos(omega * t[i])
    
    return pot_obl

def eccentricity_tides(a, omega, ecc, t, lons, lats):
    nt = len(t)
    nx = lons.shape[1]

    x1 = numpy.radians(lons)
    y1 = numpy.radians(lats)
    
    pot_ecc = numpy.zeros((nt, 6 * nx, nx))
    
    for i in range(nt):
        term1 = (3 * numpy.sin(y1)**2 - 1) * numpy.cos(omega * t[i])
        term2 = numpy.cos(y1)**2 * (3 * numpy.cos(2*x1) * numpy.cos(omega * t[i]) + 4 * numpy.sin(2*x1) * numpy.sin(omega * t[i]))
        pot_ecc[i, :, :] = -0.75 * (omega**2) * (a**2) * ecc * (term1 - term2)
    
    return pot_ecc

def full_tides(a, omega, theta0, ecc, t, lons, lats):
    return eccentricity_tides(a, omega, ecc, t, lons, lats) + obliquity_tides(a, omega, theta0, t, lons, lats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument( '-n', '--nx', type=int, default=32, help='CS grid num' )
    parser.add_argument( '-b', '--bathy', type=str, default='flat', choices=['flat', 'ridge'], help='Bathy type' )
    parser.add_argument( '-p', '--potential', type=str, default='full', choices=['ecc', 'obl', 'full'], help='Tidal potential' )

    args = parser.parse_args()

    nx        = args.nx
    bathy     = args.bathy
    potential = args.potential

    a, period, H, nz = parse_mitgcm_data( 'data' )
    
    omega  = 2 * numpy.pi / period
    theta0 = 0.053 / 180 * numpy.pi
    ecc    = 0.0094
    nt     = 25
    
    t = tidal_time_series( period, nt )
    
    lon, lat = readMITgrid( 
                             path = 'state/grid_check/custom/', 
                             nx   = nx
                          )
    
    save2file(
                fout = 'initrand_cs.bin',
                data = random_noise( 
                                      mag = 1e-6, 
                                      nz  = nz, 
                                      nx  = nx 
                                   ) 
             )
    
    if args.bathy == 'flat':
        save2file(
                    fout = 'topo_cs.bin',
                    data = flat_bathymetry( 
                                             depth = H, 
                                             nx    = nx 
                                          ) 
                 )
    else:
        save2file(
                    fout = 'topo_cs.bin',
                    data = gauss_ridge( 
                                         height = 2e4, 
                                         width  = 10, 
                                         lons   = lon, 
                                         lats = lat
                                      )
                 )
    
    if args.potential == 'obl':
        save2file(
                    fout = 'tidePot_cs.bin',
                    data = obliquity_tides( 
                                             a      = a, 
                                             omega  = omega, 
                                             theta0 = theta0, 
                                             t      = t, 
                                             lons   = lon, 
                                             lats   = lat
                                          ) 
                 )
    elif args.potential == 'ecc':
        save2file(
                    fout = 'tidePot_cs.bin',
                    data = eccentricity_tides( 
                                                a     = a, 
                                                omega = omega, 
                                                ecc   = ecc, 
                                                t     = t, 
                                                lons  = lon, 
                                                lats  = lat 
                                             )
                 )
    else:
        save2file(
                    fout = 'tidePot_cs.bin',
                    data = full_tides(
                                        a      = a, 
                                        omega  = omega, 
                                        theta0 = theta0,
                                        ecc    = ecc,
                                        t      = t,
                                        lons   = lon,
                                        lats   = lat 
                                      )
                 )
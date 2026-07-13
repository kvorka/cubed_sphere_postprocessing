#!/bin/bash
nx=32
nz=20
bathy='flat'

potential='full'      ## Tidal potential type
radius=1561.0         ## Outer radius of the computational domain in km
depth=100.0           ## Ocean depth in km
period=306806.0       ## Rotational period
obliquity=0.053       ## Obliquity in degrees
eccentricity=0.0094   ## Eccentricity
gaussWidth=10.0       ## Width of the Gaussian ridge/peak in degrees
gaussHeight=20.0      ## Height of the Gaussian ridge/peak in km
nPeriod=25            ## Timesteps in one period for potential sampling

python generate_grid.py --nx="$nx" \
                        --radius="$radius"

python generate_bathy.py --nx="$nx" \
                         --nz="$nz" \
                         --bathy="$bathy" \
                         --potential="$potential" \
                         --radius="$radius" \
                         --depth="$depth" \
                         --period="$period" \
                         --obl="$obliquity" \
                         --ecc="$eccentricity" \
                         --nt="$nPeriod" \
                         --wg="$gaussWidth" \
                         --hg="$gaussHeight"

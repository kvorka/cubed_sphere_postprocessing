#!/bin/bash
copy_from="/nfsjk/kvorka/MITgcm-tides-custom/run_cs32x32x20_Ah1e4_Av1e1_ridge/"
copy_to="state/ridge_run/"

if [ ! -d "$copy_to" ]; then
    mkdir -p "$copy_to"
fi

cp "$copy_from"monitor.0000000000.t00* "$copy_to"
cp "$copy_from"state.0000000000.t00* "$copy_to"
cp "$copy_from"grid.t00* "$copy_to"
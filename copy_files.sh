#!/bin/bash
copy_data() {
    if [ ! -d "$2" ]; then
        mkdir -p "$2"
    fi
    
    cp "$1"monitor.0000000000.t* "$2"
    cp "$1"state.0000000000.t* "$2"
    cp "$1"grid.t* "$2"
}

copy_data "/nfsjk/kvorka/MITgcm-tides-custom/run_exch2/" "state/run_exch2_64x64x20/"
#copy_data "/nfsjk/kvorka/MITgcm-tides-custom/run_cs32x32x20_Ah1e4_Av1e1_ridge/" "state/ridge_32x32x20_Ah1e4_Av1e1/"
#copy_data "/nfsjk/kvorka/MITgcm-tides-custom/run_cs32x32x20_Ah1e3_Av1e0_ridge/" "state/ridge_32x32x20_Ah1e3_Av1e0/"
#copy_data "/nfsjk/kvorka/MITgcm-tides-custom/run_cs32x32x20_Ah1e4_Av1e1_flat/" "state/flat_32x32x20_Ah1e4_Av1e1/"
#copy_data "/nfsjk/kvorka/MITgcm-tides-custom/run_cs32x32x20_Ah1e3_Av1e0_flat/" "state/flat_32x32x20_Ah1e3_Av1e0/"

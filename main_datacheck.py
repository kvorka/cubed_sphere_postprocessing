from pylib.pth import path, ntiles
from pylib.cbs import cbs_load

#####################################################################
## Load dataset of interest.                                       ##
#####################################################################
csLoader = cbs_load( path2cs      = path,
                     ntiles       = ntiles,
                     load_monitor = True,
                     load_data    = True )

#####################################################################
## Check the data.                                                 ##
#####################################################################
csLoader.check_data( 'Eta', 'U', 'V', 'W' )
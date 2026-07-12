from pylib.pth import path
from pylib.cbs import cbs_load

#####################################################################
## Load dataset of interest.                                       ##
#####################################################################
csLoader = cbs_load( path2cs      = path,
                     load_monitor = True,
                     load_data    = True )

#####################################################################
## Check the shape of the data.                                    ##
#####################################################################
csLoader.check_shapes( 'Eta', 'U', 'V', 'W' )

#####################################################################
## Check the CFL criterions.                                       ##
#####################################################################
csLoader.check_cfl()
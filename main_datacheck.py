from pylib.pth import path
from pylib.cbs import cbs_load

#####################################################################
## Load dataset of interest.                                       ##
#####################################################################
csLoader = cbs_load( path2cs = path )

#####################################################################
## Check the shape of the data.                                    ##
#####################################################################
csLoader.check( 'Eta', 'U', 'V', 'W' )
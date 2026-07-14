import matplotlib.pyplot
from pylib.cbs import cbs_load

cs = [ cbs_load( path2cs = 'state/flat_32x32x20_Ah1e4_Av1e1/',  load_monitor = True, load_data = False, name='KE flat'  ),
       cbs_load( path2cs = 'state/ridge_32x32x20_Ah1e4_Av1e1/', load_monitor = True, load_data = False, name='KE ridge' ) ]

for cs1 in cs:
    matplotlib.pyplot.plot( *cs1.get_KE_series( id='ke_mean' ), linestyle='-', label=cs1.name )

matplotlib.pyplot.xlabel( 'Time [s]' )
matplotlib.pyplot.ylabel( 'KE' )
matplotlib.pyplot.grid( True )
matplotlib.pyplot.legend()

matplotlib.pyplot.xlim(1e7, 7e7)
matplotlib.pyplot.ylim(0.0, 7e-5)

matplotlib.pyplot.show()
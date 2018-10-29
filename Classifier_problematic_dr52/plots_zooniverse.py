import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table
from Inspect_GALAH_class import *

# ## Assuming you want to look at specific lines (and check RV), then use define the 'help_lines' variable before plotting, otherwise define 'help_line=False'
help_lines=[
    (r'$\mathrm{H_\alpha}$',6562.7970),
    (r'$\mathrm{H_\beta}$' ,4861.3230),
    (r'$\mathrm{Li}$'      ,6707.7635)
    ]

spectra_dir = '/media/storage/HERMES_REDUCED/dr5.3/'
galah_params = Table.read(spectra_dir+'sobject_iraf_53.fits')
galah_params = galah_params[np.logical_and(galah_params['flag_guess']==0, galah_params['red_flag']<64)]
galah_params = galah_params[np.logical_and(galah_params['sobject_id']>140308000000000, galah_params['sobject_id']<140309000000000)]
sobjects = galah_params['sobject_id'][::1]
print sobjects
'''
spectra_dir = '/media/storage/HERMES_REDUCED/dr5.2/'
galah_params = Table.read('/home/klemen/data4_mount/'+'sobject_iraf_iDR2_171103_sme.fits')
galah_params = galah_params[galah_params['Vsini_sme']>60]
sobjects = galah_params['sobject_id']
print sobjects
'''

#spectra_dir = '/media/storage/HERMES_REDUCED/dr5.3/'
#sobjects = [140810002201339,140810002201340,140810002201341,140810002201342,140810002201343,140810002201344,140810002201345]

for s_id in sobjects:
	print 'Working on a sobject_id '+str(s_id)
	# Now let's create the class FITS for a given sobject_id and use the provided functions on it!
	fits = fits_class(sobject_id=s_id, directory=spectra_dir)

	# ## Plot normalised spectrum on 4 axes
	print 'Plotting'
	fits.plot_norm_spectrum_on4axes(help_lines=help_lines, savefig='DR53_test_plots2')

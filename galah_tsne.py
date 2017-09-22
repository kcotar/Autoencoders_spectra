import imp, os, socket

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from astropy.table import Table, Column

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# different tSNE implementations and speedups
# from tsne import bh_sne
# from MulticoreTSNE import MulticoreTSNE as TSNE

pc_name = socket.gethostname()

# input data
if pc_name == 'gigli' or pc_name == 'klemen-P5K-E':
    tsne_path = '/home/klemen/tSNE_test/'
    galah_data_input = '/home/klemen/GALAH_data/'
    imp.load_source('helper_functions', '../tSNE_test/helper_functions.py')
    imp.load_source('tsne_functions', '../tSNE_test/tsne_functions.py')
    from tsne_functions import *
else:
    tsne_path = '/data4/cotar/'
    galah_data_input = '/data4/cotar/'
from helper_functions import move_to_dir
from tsne_functions import *

galah_param_file = 'sobject_iraf_52_reduced.fits'
spectra_file_list = ['galah_dr52_ccd1_4710_4910_wvlstep_0.04_lin_RF.csv',
                     'galah_dr52_ccd2_5640_5880_wvlstep_0.05_lin_RF.csv',
                     'galah_dr52_ccd3_6475_6745_wvlstep_0.06_lin_RF.csv',
                     'galah_dr52_ccd4_7700_7895_wvlstep_0.07_lin_RF.csv']

# --------------------------------------------------------
# ---------------- Various algorithm settings ------------
# --------------------------------------------------------

# algorithm settings
output_results = True
output_plots = True
limited_rows = True
n_lim_rows = 5000
snr_cut = False  # do not use in this configuration
run_tsne_test = True
run_float_tsne = True

# reading settings
spectra_get_cols = [4000, 4000, 4000, 2016]

# --------------------------------------------------------
# ---------------- MAIN PROGRAM --------------------------
# --------------------------------------------------------

galah_param = Table.read(galah_data_input + galah_param_file)

spectral_data = list([])
for i_band in [0, 1, 2, 3]:
    suffix = ''

    spectra_file = spectra_file_list[i_band]
    # data availability check
    if not os.path.isfile(galah_data_input + spectra_file):
        print 'Spectral file not found: '+spectra_file
        continue

    print 'Working on '+spectra_file
    # determine cols to be read from csv file
    spectra_file_split = spectra_file.split('_')
    wlv_begin = float(spectra_file_split[3])
    wlv_end = float(spectra_file_split[4])
    wlv_step = float(spectra_file_split[6])
    wvl_range = np.arange(wlv_begin, wlv_end, wlv_step)
    n_wvl = len(wvl_range)
    # select the middle portion of the spectra, that should be free of nan values
    col_start = int(n_wvl/2. - spectra_get_cols[i_band]/2.)
    col_end = int(col_start + spectra_get_cols[i_band])

    if snr_cut and not limited_rows:
        snr_percentile = 10.
        snr_col = 'snr_c'+str(i_band+1)+'_guess'  # as ccd numbering starts with 1
        print 'Cutting off {:.1f}% of spectra with low snr value ('.format(snr_percentile)+snr_col+').'
        snr_percentile_value = np.percentile(galah_param[snr_col], snr_percentile)
        skip_rows = np.where(galah_param[snr_col] < snr_percentile_value)[0]
        suffix += '_snrcut'
    elif limited_rows:
        print 'Only limited number ('+str(n_lim_rows)+') of spectra rows will be read'
        skip_rows = np.arange(n_lim_rows, len(galah_param))
        suffix += '_subset'
    else:
        skip_rows = None

    if run_float_tsne:
        suffix += '_float'

    # --------------------------------------------------------
    # ---------------- Data reading and handling -------------
    # --------------------------------------------------------
    print 'Reading spectral data from {:07.2f} to {:07.2f}'.format(wvl_range[col_start], wvl_range[col_end])
    spectral_data.append(pd.read_csv(galah_data_input + spectra_file, sep=',', header=None, na_values='nan', dtype=np.float16,  # float16 is more than enough
                                     usecols=range(col_start, col_end), skiprows=skip_rows).values)

spectral_data = np.hstack(spectral_data)
print spectral_data.shape
n_wvl_total = spectral_data.shape[1]

# somehow handle cols with nan values, delete cols or fill in data
idx_bad_spectra = np.where(np.logical_not(np.isfinite(spectral_data)))
n_bad_spectra = len(idx_bad_spectra[0])
if n_bad_spectra > 0:
    print 'Correcting '+str(n_bad_spectra)+' bad flux values in read spectra.'
    spectral_data[idx_bad_spectra] = 1.  # remove nan values with theoretical continuum flux value

# handle extreme values in data
val_large = 2.5
idx_large = np.where(spectral_data > val_large)
n_bad_spectra = len(idx_large[0])
if n_bad_spectra > 0:
    print 'Correcting '+str(n_bad_spectra)+' large flux values.'
    spectral_data[idx_large] = val_large  # remove nan values with theoretical continuum flux value

val_negative = -0.1
idx_negative = np.where(spectral_data < val_negative)
n_bad_spectra = len(idx_negative[0])
if n_bad_spectra > 0:
    print 'Correcting '+str(n_bad_spectra)+' negative flux values.'
    spectral_data[idx_negative] = val_negative  # remove nan values with theoretical continuum flux value

if limited_rows:
    galah_param = galah_param[:n_lim_rows]

# run t-SNE projection
if run_tsne_test:
    print 'Running tSNE on input spectra'
    perp = 30
    theta = 0.3
    seed = 1337
    tsne_result = bh_tsne(spectral_data, no_dims=2, perplexity=perp, theta=theta, randseed=seed, verbose=True,
                          distance='euclidean', path=tsne_path, use_floats=run_float_tsne)

    tsne_ax1, tsne_ax2 = tsne_results_to_columns(tsne_result)

    csv_out_filename = 'tsne_results_galah_perp'+str(perp)+'_theta'+str(theta)+'_ccd1234'+suffix

    if output_results:
        tsne_data = galah_param['sobject_id', 'galah_id']
        tsne_data.add_column(Column(name='tsne_axis_1', data=tsne_ax1, dtype=np.float32))
        tsne_data.add_column(Column(name='tsne_axis_2', data=tsne_ax2, dtype=np.float32))
        tsne_data.write(csv_out_filename, format='ascii.csv')

    if output_plots:
        # plot tSNE results
        move_to_dir(csv_out_filename[:-4])

        print 'Plotting final tSNE results'
        plot_tsne_results(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'],
                          galah_param, ['teff_guess', 'logg_guess', 'feh_guess', 'rv_guess'],
                          suffix='', prefix='', ps=0.2)

import imp
from astropy.table import Table, join
import numpy as np
from glob import glob

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import *
imp.load_source('helper2', '../tSNE_test/cannon3_functions.py')
from helper2 import *

cannon_data = Table.read('/home/klemen/GALAH_data/sobject_iraf_iDR2_171103_cannon.fits')
abund_cannon = get_abundance_cols3(cannon_data.colnames)
abund_cannon_flag = flag_cols(abund_cannon)
abund_cannon_use = list(abund_cannon_flag)
for c in ['sobject_id', 'flag_guess', 'red_flag', 'flag_cannon', 'snr_c1_iraf', 'snr_c2_iraf', 'snr_c3_iraf', 'snr_c4_iraf', 'ra', 'dec', 'galah_id']:
    abund_cannon_use.append(c)
cannon_data = cannon_data[abund_cannon_use]

in_dir = '/home/klemen/Autoencoders_spectra/Cannon3.0_SME_20171111_multiple_29_stride2_ext0/'

data_raw_list = list([])
for fits in glob(in_dir+'*_run*.fits'):
    print fits
    abund_ann = Table.read(fits)
    abund_cols = get_abundance_colsann(abund_ann.colnames)
    abund_cols_flags = flag_cols(abund_cols)
    print abund_cols

    data_raw = abund_ann[abund_cols].to_pandas().values
    data_raw_list.append(data_raw)
    print data_raw.shape

data_raw_all = np.stack(data_raw_list)
data_raw_median = np.median(data_raw_all, axis=0)
data_raw_std = np.std(data_raw_all, axis=0)

for i_c in range(len(abund_cols)):
    print abund_cols[i_c]
    # write results to final array and compute std for every abundance measurement
    abund_ann[abund_cols[i_c]] = data_raw_median[:, i_c]
    abund_ann['e_'+abund_cols[i_c]] = data_raw_std[:, i_c]
    std_abund = np.nanstd(abund_ann[abund_cols[i_c]])
    # filter outliers by setting them to nan
    # idx_std_out = np.abs(abund_ann[abund_cols[i_c]] - np.nanmean(abund_ann[abund_cols[i_c]])) >= 5.*std_abund
    # abund_ann[abund_cols[i_c]][idx_std_out] = np.nan
    # abund_ann['e_'+abund_cols[i_c]][idx_std_out] = np.nan
    #
    # print ' STD before:{:.2f} after:{:.2f}'.format(std_abund, np.nanstd(abund_ann[abund_cols[i_c]]))
    # print '  - masked:', np.sum(idx_std_out)

sme_cols = [c for c in abund_ann.colnames if 'sme' in c]
abund_ann.remove_columns(sme_cols)

# subset of data to be able to use cannon abund flags

abund_ann_cannon = join(abund_ann, cannon_data, keys='sobject_id')
# rename cols
for i_c in range(len(abund_cols)):
    abund_ann_cannon[abund_cannon_flag[i_c]].name = abund_cols_flags[i_c]

abund_ann_cannon.write(in_dir+'galah_abund_ANN_SME3.0.1_stacked_median.fits', overwrite=True)

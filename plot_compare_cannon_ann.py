import os
from astropy.table import Table, join
from socket import gethostname
from itertools import combinations, combinations_with_replacement

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# PC hostname
pc_name = gethostname()

# input data
if pc_name == 'gigli' or pc_name == 'klemen-P5K-E':
    galah_data_input = '/home/klemen/GALAH_data/'
else:
    galah_data_input = '/data4/cotar/'


cannon_data = Table.read(galah_data_input+'sobject_iraf_iDR2_171103_cannon.fits')
ann_data = Table.read(galah_data_input+'galah_abund_ANN_SME3.0.1_.fits')

cannon_abundances_list = [col for col in cannon_data.colnames if '_abund_cannon' in col and 'e_' not in col and 'flag_' not in col and len(col.split('_'))<4]
ann_abundances_list = [col for col in ann_data.colnames if '_abund_ann' in col and 'e_' not in col and 'flag_' not in col]

joined_data = join(ann_data, cannon_data, join_type='inner')

os.chdir('Cannon_vs_ann_')
for col in ann_abundances_list:
    elem = col.split('_')[0]
    print elem
    idx_ok = joined_data['flag_'+elem+'_abund_cannon'] <= 3
    perc_ok = 100.*np.sum(idx_ok)/len(joined_data)
    print ' Percent ok', perc_ok
    lim = (np.nanpercentile(joined_data[elem+'_abund_cannon'], 1), np.nanpercentile(joined_data[elem+'_abund_cannon'], 99))
    plt.scatter(joined_data[elem+'_abund_cannon'][idx_ok], joined_data[elem+'_abund_ann'][idx_ok], s=1, alpha=0.05, lw=0, c='black')
    plt.plot([lim[0], lim[1]], [lim[0], lim[1]], linestyle='dashed', c='red', alpha=0.5)
    plt.title('Element: '+elem+'  unflagged: {:.1f}%'.format(perc_ok))
    plt.xlabel('CANNON')
    plt.ylabel('Artificial neural network')
    plt.ylim(lim)
    plt.xlim(lim)
    plt.savefig(elem+'_cannon_ann.png', dpi=200)
    plt.close()

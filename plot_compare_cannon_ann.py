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
    galah_data_input = '/home/klemen/data4_mount/'
else:
    galah_data_input = '/data4/cotar/'


cannon_data = Table.read(galah_data_input+'sobject_iraf_iDR2_171103_cannon.fits')
ann_data = Table.read(galah_data_input+'galah_abund_ANN_SME3.0.1_all_stacked_median_ext0.fits')
cannon_data = cannon_data.filled(-1)
ann_data = ann_data.filled(-1)

cannon_abundances_list = [col for col in cannon_data.colnames if '_abund_cannon' in col and 'e_' not in col and 'flag_' not in col and len(col.split('_'))<4]
ann_abundances_list = [col for col in ann_data.colnames if '_abund_ann' in col and 'e_' not in col and 'flag_' not in col]

joined_data = join(ann_data, cannon_data, join_type='inner')

os.chdir('Cannon_vs_ann-res-renorm-stacked_all_ext0')

# s_ids = joined_data[np.logical_and(joined_data['Vsini_cannon'] >= 70, joined_data['flag_cannon'] == 0)]['sobject_id']
# print ','.join([str(s) for s in s_ids])

for col in ['Teff', 'Feh', 'Vsini', 'Vmic']:
    print col
    idx_ok = joined_data['flag_cannon'] == 0
    # idx_ok = np.in1d(joined_data['sobject_id'], [140413003701012,140413003701131,140707000101287,140711001301368,140810004701332,150409002101355,150409004101052,150411005101001,150412004601071,150531000101103,150601001601160,150827003401147,150828004201049,150830006601171,151111001601391,151225002701359,151231004901363,160331005801120,160415003601029,160419003101278,160420003801213,160426003501052,160426006701061,160522002101125,160522003601141,160524002701254,160529003401004,160530002201189,160531001601186,160531004601173,160724003501157,160815002101301,160923004201066,160923004201122,161106005101025,161213001601307,161217004601039,161217006101165,161218003101222,161219001801307,161219002601152,170107003601190,170107004801016,170108002701388,170108003301086,170112002601230,170112003101387,170115002201023,170128001601196,170206003701001,170220004601122,170414005601188,170415002501201,170415002501266,170509006701173,170510006801116,170517001801021,170517001801156,170724001601293,170828001601140,170828002701151,170830001101087])
    perc_ok = 100. * np.sum(idx_ok) / len(joined_data)
    print ' Percent ok', perc_ok
    lim = (np.nanpercentile(joined_data[col + '_cannon'], 0.1), np.nanpercentile(joined_data[col + '_cannon'], 100))
    plt.scatter(joined_data[col + '_cannon'][idx_ok], joined_data[col + '_ann'][idx_ok], s=1, alpha=1, lw=0, c='black')
    plt.plot([lim[0], lim[1]], [lim[0], lim[1]], linestyle='dashed', c='red', alpha=0.5)
    plt.title('Parameter: ' + col + '  unflagged: {:.1f}%'.format(perc_ok))
    plt.xlabel('CANNON')
    plt.xlabel('CANNON')
    plt.ylabel('Artificial neural network')
    plt.ylim(lim)
    plt.xlim(lim)
    plt.savefig(col + '_cannon_ann.png', dpi=200)
    plt.close()

for col in ann_abundances_list:
    elem = col.split('_')[0]
    print elem
    # idx_ok = joined_data['flag_'+elem+'_abund_cannon'] < 4
    idx_ok = np.logical_or(joined_data['flag_'+elem+'_abund_cannon'] == 2, joined_data['flag_'+elem+'_abund_cannon'] == 0)
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


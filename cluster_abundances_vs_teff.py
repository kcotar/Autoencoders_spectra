import imp, sys
import itertools

import numpy as np
import astropy.units as un
import astropy.coordinates as coord
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, scale
from scipy.cluster.hierarchy import linkage, to_tree
from astropy.table import Table, join
from getopt import getopt

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import *
imp.load_source('helper2', '../tSNE_test/cannon3_functions.py')
from helper2 import *
# input data
galah_data_dir = '/home/klemen/data4_mount/'

galah_cannon = Table.read(galah_data_dir+'sobject_iraf_iDR2_171103_cannon.fits')
abund_cols = get_abundance_cols3(galah_cannon.colnames)
set = '_cannon'

# galah_cannon = Table.read(galah_data_dir+'galah_abund_ANN_SME3.0.1_stacked_median_ext0.fits')
# abund_cols = get_abundance_colsann(galah_cannon.colnames)
# set = '_ann'

list_objects = np.array([160106001601089,160106001601282,160106001601267,160106001601259,160106001601230,160923003701263,160106001601301,160106001601153,160106001601097,160106001601078,160106001601129,160923003701001,160109002501320,160923003701164,170830004001231,170830005101052,170828004401029,170829002901275,170828003901384,170829002901175,170829002901128,170829002901108,170829003901336,170830004001244,170829002901052,170830004001205,170830004001363,170830004001040,170829003401291,170829003401169,170830005101301,170830005101223,160401002101397,160401002101323,170102001901023,160401002101180,160401002101122,160401002101357,160401002101067,160401002101138,170828002701351,170829002401184,170828002701137,170828002701095,170829001901293,170829001901382,170829001901022,170828002701287,170829002401291,170829001901252,170828002701097,170829001901008,])
idx_set = np.in1d(galah_cannon['sobject_id'], list_objects)
galah_cannon = galah_cannon[idx_set]

for col in abund_cols:
    idx_ok = np.logical_and(galah_cannon['flag_'+col]==0,galah_cannon['flag_cannon']==0)
    plt.scatter(galah_cannon['Teff'+set][idx_ok], galah_cannon[col][idx_ok]-galah_cannon['Feh'+set][idx_ok])
    plt.xlabel('Teff'+set)
    plt.ylabel(col+' - Feh'+set)
    #plt.show()
    plt.savefig(col+'_teff.png', drpi=300)
    plt.close()


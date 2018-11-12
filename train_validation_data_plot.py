from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt


data_dir = '/data4/cotar/'
iso_dir = data_dir+'isochrones/padova_Gaia_DR2_Solar/'
t = Table.read(data_dir+'GALAH_iDR3_ts_DR2.fits')
o = Table.read(data_dir+'GALAH_iDR3_OpenClusters.fits')
g = Table.read(data_dir+'GALAH_iDR3_GlobularClusters.fits')
i = Table.read(iso_dir+'isochrones_all.fits')

plt.scatter(t['teff'], t['logg'], s=0.7,lw=0,label='train',c='black',alpha=0.5)
plt.scatter(o['teff'], o['logg'], s=0.7,lw=0,label='OC',c='blue',alpha=0.5)
plt.scatter(g['teff'], g['logg'], s=0.7,lw=0,label='GC',c='red',alpha=0.5)

for mh in np.unique(i['MHini']):
    i_mh = i[i['MHini']==mh]
    plt.plot(i_mh['teff'], i_mh['logg'], label='M/H {:.2f}'.format(mh))

plt.xlim(7300, 3500)
plt.ylim(5, 0)
plt.legend()
plt.tight_layout()
plt.savefig('sme_kiel.png', dpi=300)
plt.close()
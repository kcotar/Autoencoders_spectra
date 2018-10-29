
# # Inspect_GALAH: A tool to for GALAH spectra as well as SME/Cannon output
# 
# Author: Sven Buder

# Compatibility with Python 3
from __future__ import (absolute_import, division, print_function)

try:
    get_ipython().magic(u'matplotlib inline')
    get_ipython().magic(u"config InlineBackend.figure_format='retina'")
except:
    pass

# Basic packages
import numpy as np
import os
import sys

# Packages to work with FITS and (IDL) SME.out files
import astropy.io.fits as pyfits
import astropy.table as table
from scipy.io.idl import readsav

# Matplotlib and associated packages for plotting
import matplotlib.pyplot as plt
params = {'text.usetex': True, 'text.latex.preamble': [r'\usepackage{upgreek}', r'\usepackage{amsmath}'],'font.family' : 'lmodern','font.size' : 11}   
plt.rcParams.update(params)

# Package to save multiple PDF pages in one PDF
from matplotlib.backends.backend_pdf import PdfPages

# First be sure you are in the right directory and where your spectra and SME/Cannon files are:

# Now we can start with the major routines to save the spectra and SME/Cannon files in a dictionary strucuture: 

# # Create 'fits' class and fill with necessary data and enable plot routines

def get_fits_data(directory='default',sobject_id=140708006401203):
    """
    INPUT:
    
    directory  = directory where the 4 CCD fits files are
    sobject_id = sobject_id
    
    OUTPUT:
    
    'spectrum'
    
    This is a dictionary structure, i.e. you can access the wavelength entry via sme['wave']. Keywords are:
    
    directory  : same as input
    sobject_id : same as sobject_id
    wave_raw   : IRAF wavelength of raw flux
    sob_raw    : IRAF raw flux for CCD
    uob_raw    : IRAF raw flux error
    wave_norm  : IRAF wavelength of GUESS normalised flux
    sob_norm   : IRAF GUESS normalised flux
    uob_norm   : IRAF GUESS normalised flux error
    as well as ccd_*CCD* in front of the major keywords for CCD Nr. *CCD* (e.g. ccd_*CCD*_wave_raw)

    """
    
    spectrum = {}
    spectrum['directory'] = directory
    spectrum['sobject_id'] = sobject_id

    for each in range(1,4+1):

        if directory == 'default':
            fits_path = 'SPECTRA/dr5.2/'+str(sobject_id)[0:6]+'/standard/com/'+str(sobject_id)+str(each)+'.fits'
        else:
            fits_path = directory+''+str(sobject_id)[0:6]+'/standard/com/'+str(sobject_id)+str(each)+'.fits'
            #door = pyfits.open(self.directory+str(sobject_id)+str(each)+'.fits')
		
	if not os.path.isfile(fits_path):
		return
	door = pyfits.open(fits_path)

        ws=door[0].header["CRVAL1"]
        inc=door[0].header["CDELT1"]
        nax=door[0].header["NAXIS1"]
        spectrum['ccd'+str(each)+'_wave_raw']=(np.arange(0,nax) + 1)*inc+ws
        ws=door[4].header["CRVAL1"]
        inc=door[4].header["CDELT1"]
        nax=door[4].header["NAXIS1"]
        spectrum['ccd'+str(each)+'_wave_norm']=(np.arange(0,nax) + 1)*inc+ws

        spectrum['ccd'+str(each)+'_sob_raw'] = door[0].data
        spectrum['ccd'+str(each)+'_sob_norm'] = door[4].data

        spectrum['ccd'+str(each)+'_uob_raw'] = door[0].data*door[1].data
        spectrum['ccd'+str(each)+'_uob_norm'] = door[4].data*door[1].data[len(door[1].data)-len(door[4].data):]

    spectrum['wave_raw'] = np.concatenate([spectrum['ccd'+str(ccd)+'_wave_raw'] for ccd in range(1,4+1)])
    spectrum['wave_norm'] = np.concatenate([spectrum['ccd'+str(ccd)+'_wave_norm'] for ccd in range(1,4+1)])
    spectrum['sob_raw'] = np.concatenate([spectrum['ccd'+str(ccd)+'_sob_raw'] for ccd in range(1,4+1)])
    spectrum['sob_norm'] = np.concatenate([spectrum['ccd'+str(ccd)+'_sob_norm'] for ccd in range(1,4+1)])
    spectrum['uob_raw'] = np.concatenate([spectrum['ccd'+str(ccd)+'_uob_raw'] for ccd in range(1,4+1)])
    spectrum['uob_norm'] = np.concatenate([spectrum['ccd'+str(ccd)+'_uob_norm'] for ccd in range(1,4+1)])

    return spectrum

class fits_class(object):
    """ A class with all FITS entries and necessary routines """
    
    def __init__(self, sobject_id=140708006401203, directory='default'):
        self.sobject_id = sobject_id
        self.directory = directory
        
        spectrum = get_fits_data(self.directory,self.sobject_id)
	if spectrum is not None:        
		self.wave_raw  = spectrum['wave_raw']
		self.ccd1_wave_raw  = spectrum['ccd1_wave_raw']
		self.ccd2_wave_raw  = spectrum['ccd2_wave_raw']
		self.ccd3_wave_raw  = spectrum['ccd3_wave_raw']
		self.ccd4_wave_raw  = spectrum['ccd4_wave_raw']
		self.wave_norm = spectrum['wave_norm']
		self.ccd1_wave_norm = spectrum['ccd1_wave_norm']
		self.ccd2_wave_norm = spectrum['ccd2_wave_norm']
		self.ccd3_wave_norm = spectrum['ccd3_wave_norm']
		self.ccd4_wave_norm = spectrum['ccd4_wave_norm']
		self.sob_raw   = spectrum['sob_raw']
		self.ccd1_sob_raw   = spectrum['ccd1_sob_raw']
		self.ccd2_sob_raw   = spectrum['ccd2_sob_raw']
		self.ccd3_sob_raw   = spectrum['ccd3_sob_raw']
		self.ccd4_sob_raw   = spectrum['ccd4_sob_raw']
		self.sob_norm  = spectrum['sob_norm']
		self.ccd1_sob_norm  = spectrum['ccd1_sob_norm']
		self.ccd2_sob_norm  = spectrum['ccd2_sob_norm']
		self.ccd3_sob_norm  = spectrum['ccd3_sob_norm']
		self.ccd4_sob_norm  = spectrum['ccd4_sob_norm']
		self.uob_raw   = spectrum['uob_raw']
		self.ccd1_uob_raw   = spectrum['ccd1_uob_raw']
		self.ccd2_uob_raw   = spectrum['ccd2_uob_raw']
		self.ccd3_uob_raw   = spectrum['ccd3_uob_raw']
		self.ccd4_uob_raw   = spectrum['ccd4_uob_raw']
		self.uob_norm  = spectrum['uob_norm']
		self.ccd1_uob_norm  = spectrum['ccd1_uob_norm']
		self.ccd2_uob_norm  = spectrum['ccd2_uob_norm']
		self.ccd3_uob_norm  = spectrum['ccd3_uob_norm']
		self.ccd4_uob_norm  = spectrum['ccd4_uob_norm']
        
    def plot_norm_spectrum_on4axes(self,
        include_errors=True,
        ylim=(-0.1,1.1),
        xlim_ccd4='raw',
        savefig='PLOTS_4CCDs',
        continuum=True,
        help_lines=[(r'$\mathrm{H_\alpha}$',6562.7970),(r'$\mathrm{H_\beta}$',4861.3230),(r'$\mathrm{Li}$',6707.7635)]
        ):
        """
        INPUT:
        
        include_errors : Includes grey area sob-uob to sob+uob in background
        ylim           : (lower_yaxis,upper_yacis), default: (-0.1,1.1)
        savefig        : will save plot in directory passed with 'savefig'
        continuum=True : continuum line at 1.0
        help_lines     : will add vertical lines with (name,lambda)
        
        OUTPUT:
        
        4 axes plot of normalised spectrum
        
        """
        
        kwargs_plot = dict(color='black', linewidth=0.5)
        kwargs_fill_between = dict(color='grey', linewidth=0.5)
        f,(ax1,ax2,ax3,ax4) = plt.subplots(4,figsize=(15,10),sharey=True)
        sob1 = ax1.plot(self.ccd1_wave_norm,self.ccd1_sob_norm,label='Spectrum',**kwargs_plot)
        ax2.plot(self.ccd2_wave_norm,self.ccd2_sob_norm,**kwargs_plot)
        ax3.plot(self.ccd3_wave_norm,self.ccd3_sob_norm,**kwargs_plot)
        ax4.plot(self.ccd4_wave_norm,self.ccd4_sob_norm,**kwargs_plot)
        if include_errors==True:
            fill1 = ax1.fill_between(self.ccd1_wave_norm,self.ccd1_sob_norm-self.ccd1_uob_norm,self.ccd1_sob_norm+self.ccd1_uob_norm,label='Error',**kwargs_fill_between)
            ax2.fill_between(self.ccd2_wave_norm,self.ccd2_sob_norm-self.ccd2_uob_norm,self.ccd2_sob_norm+self.ccd2_uob_norm,**kwargs_fill_between)
            ax3.fill_between(self.ccd3_wave_norm,self.ccd3_sob_norm-self.ccd3_uob_norm,self.ccd3_sob_norm+self.ccd3_uob_norm,**kwargs_fill_between)
            ax4.fill_between(self.ccd4_wave_norm,self.ccd4_sob_norm-self.ccd4_uob_norm,self.ccd4_sob_norm+self.ccd4_uob_norm,**kwargs_fill_between)
        ax1.legend(loc='lower left')
        ax4.set_xlabel(r'$\mathrm{Wavelength~[\AA]}$')
        ax1.set_ylabel(r'$\mathrm{Flux~[norm]}$')
        ax2.set_ylabel(r'$\mathrm{Flux~[norm]}$')
        ax3.set_ylabel(r'$\mathrm{Flux~[norm]}$')
        ax4.set_ylabel(r'$\mathrm{Flux~[norm]}$')
	ax1.set_xlim((4710,4910))
	ax2.set_xlim((5640,5890))
	ax3.set_xlim((6460,6750))
	ax4.set_xlim((7580,7890))
        ax4.set_ylim(ylim)
        # if xlim_ccd4=='raw':
        #     ax4.set_xlim(self.ccd4_wave_raw[0],self.ccd4_wave_raw[-1])

        help_line_kwargs = dict(color='red',linestyle = 'dotted')
        help_line_text_kwargs = dict(fontsize=15, ha='left')
        if help_lines!=False:
            for (each_line_name,each_line_wave) in help_lines:
                if (each_line_wave > self.ccd1_wave_norm[0]) & (each_line_wave < self.ccd1_wave_norm[-1]):
                    ax1.axvline(each_line_wave,**help_line_kwargs)
                    ax1.text(each_line_wave,0.0,each_line_name,**help_line_text_kwargs)
                if (each_line_wave > self.ccd2_wave_norm[0]) & (each_line_wave < self.ccd2_wave_norm[-1]):
                    ax2.axvline(each_line_wave,**help_line_kwargs)
                    ax2.text(each_line_wave,0.0,each_line_name,**help_line_text_kwargs)
                if (each_line_wave > self.ccd3_wave_norm[0]) & (each_line_wave < self.ccd3_wave_norm[-1]):
                    ax3.axvline(each_line_wave,**help_line_kwargs)
                    ax3.text(each_line_wave,0.0,each_line_name,**help_line_text_kwargs)
                if (each_line_wave > self.ccd4_wave_norm[0]) & (each_line_wave < self.ccd4_wave_norm[-1]):
                    ax4.axvline(each_line_wavev,**help_line_kwargs)
                    ax4.text(each_line_wave,0.0,each_line_name,**help_line_text_kwargs)

        cont_kwargs = dict(lw = 0.5, color='blue', linestyle='dashed')
        if continuum==True:
            ax1.axhline(1.0,**cont_kwargs)
            ax2.axhline(1.0,**cont_kwargs)
            ax3.axhline(1.0,**cont_kwargs)
            ax4.axhline(1.0,**cont_kwargs)
        
        plt.tight_layout()

        if savefig!=False:
            plt.savefig(savefig+'/'+str(self.sobject_id)+'_norm.png', dpi=220)
	plt.close()
        
    def plot_raw_spectrum_on4axes(self,
        include_errors=True,
        ylim1=False,
        ylim2=False,
        ylim3=False,
        ylim4=False,
        savefig='PLOTS_4CCDs_0',
        help_lines=[(r'$\mathrm{H_\alpha}$',6562.7970),(r'$\mathrm{H_\beta}$',4861.3230),(r'$\mathrm{Li}$',6707.7635)]
        ):
        """
        INPUT:
        
        include_errors : Includes grey area sob-uob to sob+uob in background
        ylim1-ylim4    : (lower_yaxis,upper_yacis), default: False, i.e. autoadjust
        savefig        : will save plot in directory passed with 'savefig'
        help_lines     : will add vertical lines at given wavelengths
        
        OUTPUT:
        
        4 axes plot of raw spectrum
        
        """
        
        kwargs_plot = dict(color='black', linewidth=0.5)
        kwargs_fill_between = dict(color='grey', linewidth=0.5)
        f,(ax1,ax2,ax3,ax4) = plt.subplots(4,figsize=(15,10))
        sob1 = ax1.plot(self.ccd1_wave_raw,self.ccd1_sob_raw,label='Spectrum',**kwargs_plot)
        ax2.plot(self.ccd2_wave_raw,self.ccd2_sob_raw,**kwargs_plot)
        ax3.plot(self.ccd3_wave_raw,self.ccd3_sob_raw,**kwargs_plot)
        ax4.plot(self.ccd4_wave_raw,self.ccd4_sob_raw,**kwargs_plot)
        if include_errors==True:
            fill1 = ax1.fill_between(self.ccd1_wave_raw,self.ccd1_sob_raw-self.ccd1_uob_raw,self.ccd1_sob_raw+self.ccd1_uob_raw,label='Error',**kwargs_fill_between)
            ax2.fill_between(self.ccd2_wave_raw,self.ccd2_sob_raw-self.ccd2_uob_raw,self.ccd2_sob_raw+self.ccd2_uob_raw,**kwargs_fill_between)
            ax3.fill_between(self.ccd3_wave_raw,self.ccd3_sob_raw-self.ccd3_uob_raw,self.ccd3_sob_raw+self.ccd3_uob_raw,**kwargs_fill_between)
            ax4.fill_between(self.ccd4_wave_raw,self.ccd4_sob_raw-self.ccd4_uob_raw,self.ccd4_sob_raw+self.ccd4_uob_raw,**kwargs_fill_between)
        ax1.legend(loc='lower left')
        ax4.set_xlabel(r'$\mathrm{Wavelength~[\AA]}$')
        ax1.set_ylabel(r'$\mathrm{Flux~[raw]}$')
        ax2.set_ylabel(r'$\mathrm{Flux~[raw]}$')
        ax3.set_ylabel(r'$\mathrm{Flux~[raw]}$')
        ax4.set_ylabel(r'$\mathrm{Flux~[raw]}$')
        if ylim1 != False:
            ax1.set_ylim(ylim1)
        if ylim2 != False:
            ax2.set_ylim(ylim2)
        if ylim3 != False:
            ax3.set_ylim(ylim3)
        if ylim4 != False:
            ax4.set_ylim(ylim4)
        
        help_line_kwargs = dict(color='red',linestyle = 'dotted')
        help_line_text_kwargs = dict(fontsize=15,va='bottom',ha='left')
        if help_lines!=False:
            for (each_line_name,each_line_wave) in help_lines:
                if (each_line_wave > self.ccd1_wave_raw[0]) & (each_line_wave < self.ccd1_wave_raw[-1]):
                    ax1.axvline(each_line_wave,**help_line_kwargs)
                    xmin1, xmax1 = ax1.get_xlim()
                    ax1.text((each_line_wave-xmin1)/(xmax1-xmin1),0.0,each_line_name,transform=ax1.transAxes,**help_line_text_kwargs)
                if (each_line_wave > self.ccd2_wave_raw[0]) & (each_line_wave < self.ccd2_wave_raw[-1]):
                    ax2.axvline(each_line_wave,**help_line_kwargs)
                    xmin2, xmax2 = ax2.get_xlim()
                    ax2.text((each_line_wave-xmin2)/(xmax2-xmin2),0.0,each_line_name,transform=ax2.transAxes,**help_line_text_kwargs)
                if (each_line_wave > self.ccd3_wave_raw[0]) & (each_line_wave < self.ccd3_wave_raw[-1]):
                    ax3.axvline(each_line_wave,**help_line_kwargs)
                    xmin3, xmax3 = ax3.get_xlim()
                    ax3.text((each_line_wave-xmin3)/(xmax3-xmin3),0.0,each_line_name,transform=ax3.transAxes,**help_line_text_kwargs)
                if (each_line_wave > self.ccd4_wave_raw[0]) & (each_line_wave < self.ccd4_wave_raw[-1]):
                    ax4.axvline(each_line_wavev,**help_line_kwargs)
                    xmin4, xmax4 = ax4.get_xlim()
                    ax4.text((each_line_wave-xmin4)/(xmax4-xmin4),0.0,each_line_name,transform=ax4.transAxes,**help_line_text_kwargs)
        ax4.axvline(self.ccd4_wave_norm[0],**help_line_kwargs)
        xmin4, xmax4 = ax4.get_xlim()
        ax4.text((self.ccd4_wave_norm[0]-xmin4)/(xmax4-xmin4),0.0,'Norm. Spectrum',transform=ax4.transAxes,**help_line_text_kwargs)
        plt.tight_layout()
        
        if savefig!=False:
            plt.savefig(savefig+'/'+str(self.sobject_id)+'_red.pdf')
	plt.close()
            
    def plot_HbHaLi(self,
        include_errors=True,
        ylimHb=False,
        ylimHa=False,
        ylimLi=False,
        windowHb=15,
        windowHa=15,
        windowLi=15,
        continuum=True,
        savefig='PLOTS_HbHaLi'
        ):
        """
        INPUT:
        
        include_errors : Includes grey area sob-uob to sob+uob in background
        ylim1-ylim3    : (lower_yaxis,upper_yacis), default: False, i.e. autoadjust
        windowHb/Ha/Li : Window size in Angstrom around the 3 lines, default: 10 
        savefig        : will save plot in directory passed with 'savefig'
        continuum=True : Continuum line at 1.0
        help_lines     : will add vertical lines at given wavelengths
        
        OUTPUT:
        
        4 axes plot of raw spectrum
        
        """
        
        kwargs_plot = dict(color='black', linewidth=0.5)
        kwargs_fill_between = dict(color='grey', linewidth=0.5)
        if (ylimHb == False) & (ylimHa == False) & (ylimLi==False):
            f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5),sharey=True)
            ax1.set_ylim(-0.1,1.1)
        else:
            f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))
        sob1 = ax1.plot(self.ccd1_wave_norm,self.ccd1_sob_norm,label='Spectrum',**kwargs_plot)
        ax2.plot(self.ccd3_wave_norm,self.ccd3_sob_norm,**kwargs_plot)
        ax3.plot(self.ccd3_wave_norm,self.ccd3_sob_norm,**kwargs_plot)
        if include_errors==True:
            fill1 = ax1.fill_between(self.ccd1_wave_norm,self.ccd1_sob_norm-self.ccd1_uob_norm,self.ccd1_sob_norm+self.ccd1_uob_norm,label='Error',**kwargs_fill_between)
            ax2.fill_between(self.ccd3_wave_norm,self.ccd3_sob_norm-self.ccd3_uob_norm,self.ccd3_sob_norm+self.ccd3_uob_norm,**kwargs_fill_between)
            ax3.fill_between(self.ccd3_wave_norm,self.ccd3_sob_norm-self.ccd3_uob_norm,self.ccd3_sob_norm+self.ccd3_uob_norm,**kwargs_fill_between)
        ax1.legend(loc='lower left')
        ax1.set_xlabel(r'$\mathrm{Wavelength~[\AA]}$')
        ax2.set_xlabel(r'$\mathrm{Wavelength~[\AA]}$')
        ax3.set_xlabel(r'$\mathrm{Wavelength~[\AA]}$')
        ax1.set_ylabel(r'$\mathrm{Flux~[raw]}$')
        if ylimHb != False:
            ax1.set_ylim(ylimHb)
        if ylimHa != False:
            ax2.set_ylim(ylimHa)
        if ylimLi != False:
            ax3.set_ylim(ylimLi)
        ax1.set_xlim(4861.3230-0.5*windowHb,4861.3230+0.5*windowHb)
        ax2.set_xlim(6562.7970-0.5*windowHa,6562.7970+0.5*windowHa)
        ax3.set_xlim(6707.7635-0.5*windowLi,6707.7635+0.5*windowLi)
        
        help_line_kwargs = dict(color='red',linestyle = 'dotted')
        help_line_text_kwargs = dict(fontsize=15,va='bottom',ha='left')

        ax1.axvline(4861.3230,**help_line_kwargs)
        ax1.set_title(r'$\mathrm{H_\beta}$')
        ax2.axvline(6562.7970,**help_line_kwargs)
        ax2.set_title(r'$\mathrm{H_\alpha}$')
        ax3.axvline(6707.7635,**help_line_kwargs)
        ax3.set_title(r'$\mathrm{Li}$')
        
        cont_kwargs = dict(lw = 0.5, color='blue', linestyle='dashed')
        if continuum==True:
            ax1.axhline(1.0,**cont_kwargs)
            ax2.axhline(1.0,**cont_kwargs)
            ax3.axhline(1.0,**cont_kwargs)
        plt.tight_layout()
        
        if savefig!=False:
            plt.savefig(savefig+'/'+str(self.sobject_id)+'_HaHbLi.pdf')
	plt.close()


def get_sme_data(directory='OUTPUT',field='bmstar',sobject_id=140708006401203,setup='DR2',mode='all',show_na=False):
    """
    INPUT:
    
    directory  = directory where the file is
    field      = field that was designated by WG4 when running SME
    sobject_id = sobject_id
    setup      = DR2 by default used by WG4 to initialise SME segments & line masks
    mode       = Instructions in 'mode_DR2' part of https://github.com/svenbuder/GALAH_BusyDays2017/blob/master/README.md
    show_na    = Show which modes could not be loaded, if mode=='all'
    
    OUTPUT:
    
    'sme'
    
    This is a dictionary structure, i.e. you can access the wavelength entry via sme['wave']. Keywords are:
    
    field, sobject_id, setup, mode : same as input

    Depending on what you use for 'mode', different arrays will be created with variable 'mode':    
    *mode*_wave : SME wavelength grid of given *mode*
    *mode*_sob  : SME norm. spectrum of given *mode* on 'wave' grid
    *mode*_uob  : SME error of sob of given *mode* on 'wave' grid
    *mode*_smod : SME's synthetic spectrum of given *mode* on 'wave' grid

    """
    
    sme = {}
    sme['field'] = field
    sme['sobject_id'] = sobject_id
    sme['setup'] = setup

    if mode == 'all':

        mode = [
            'Sp',
            'Li', 'C',
            'O', 'O7772', 'O7774', 'O7775', 'Na', 'Na4752', 'Na5683', 'Na5688',
            'Mg', 'Mg4730', 'Mg5711', 'Mg7692','Al', 'Al6696', 'Al6699', 'Al7835', 'Al7836',
            'Si', 'Si5666', 'Si5690', 'Si5793', 'Si6722', 'K', 'K5802',
            'Ca', 'Ca5857', 'Ca5862', 'Ca5868', 'Ca6494', 'Ca6500', 'Ca6501', 'Ca6509',
            'Sc', 'Ti', 'Fe', 'V', 'Cr', 'Mn', 'Co', 'Ni','Cu', 'Cu5700', 'Cu5782',
            'Zn', 'Zn4722', 'Zn4811', 'Rb', 'Sr', 'Y', 'Y4855', 'Y4884', 'Y5663', 'Y5729',
            'Zr', 'Mo', 'Ru', 'Ba', 'Ba5854', 'Ba6497', 'La', 'Ce', 'Nd', 'Nd4811',
            'Sm', 'Eu', 'Eu5819', 'Eu6645'
            ]

    sme['mode'] = mode

    for each_mode in mode:
        try:
            sme_out = readsav(directory+'/'+field+'_'+str(sobject_id)+'_'+setup+'_'+each_mode+'_SME.out')
            sme_out = sme_out.sme[0]

            sme[each_mode+'_wave']  = sme_out.wave
            sme[each_mode+'_sob']   = sme_out.sob
            sme[each_mode+'_uob']   = sme_out.uob
            sme[each_mode+'_smod']  = sme_out.smod
            sme[each_mode+'_mob']   = sme_out.mob
            sme[each_mode+'_wran']  = sme_out.wran
            sme[each_mode+'_wind']  = sme_out.wind
            sme[each_mode+'_nseg']  = sme_out.nseg
            if len(np.where(sme_out.ab_free == 1)[0]) > 0:
                sme[each_mode+'_abund'] = sme_out.abund[sme_out.ab_free == 1][0]
                sme[each_mode+'_ab_free'] = np.where(sme_out.ab_free == 1)[0][0]
            
            if (each_mode == 'Sp') | (len(mode) == 1) | ('Sp' not in mode):
                sme['teff']  = sme_out.teff
                sme['logg']  = sme_out.grav
                sme['feh']   = sme_out.feh
                sme['vsini'] = sme_out.vsini
                
        except:
            if show_na | (len(mode)==1):
                print('Mode '+each_mode+' not available')
            else:
                pass
        
    return sme;

class sme_class(object):
    """ The SME output class including plotting routines """
    
    def __init__(self, field='bmstar', sobject_id=140708006401203, directory='OUTPUT', mode='all'):
        self.field      = field
        self.sobject_id = sobject_id
        self.directory  = directory
        self.setup      = 'DR2'
        self.mode       = mode
        
        sme = get_sme_data(directory=self.directory,field=self.field,sobject_id=self.sobject_id,setup=self.setup,mode=self.mode,show_na=False)
        self.sme        = sme
        
        self.solar_el   = np.array([
             "H",  "He",  "Li",  "Be",   "B",   "C",   "N",   "O",   "F",  "Ne",
            "Na",  "Mg",  "Al",  "Si",   "P",   "S",  "Cl",  "Ar",   "K",  "Ca", 
            "Sc",  "Ti",   "V",  "Cr",  "Mn",  "Fe",  "Co",  "Ni",  "Cu",  "Zn",       
            "Ga",  "Ge",  "As",  "Se",  "Br",  "Kr",  "Rb",  "Sr",   "Y",  "Zr",  
            "Nb",  "Mo",  "Tc",  "Ru",  "Rh",  "Pd",  "Ag",  "Cd",  "In",  "Sn", 
            "Sb",  "Te",   "I",  "Xe",  "Cs",  "Ba",  "La",  "Ce",  "Pr",  "Nd", 
            "Pm",  "Sm",  "Eu",  "Gd",  "Tb",  "Dy",  "Ho",  "Er",  "Tm",  "Yb",     
            "Pm",  "Sm",  "Eu",  "Gd",  "Tb",  "Dy",  "Ho",  "Er",  "Tm",  "Yb",     
            "Tl",  "Pb",  "Bi",  "Po",  "At",  "Rn",  "Fr",  "Ra",  "Ac",  "Th",
            "Pa",   "U",  "Np",  "Pu",  "Am",  "Cm",  "Bk",  "Cs",  "Es", "TiO"
        ])
            
        solar_ab   = np.array([
            12.00, 10.93,  1.05,  1.38,  2.70,  8.39,  7.78,  8.66,  4.56,  7.84,
             6.17,  7.53,  6.37,  7.51,  5.36,  7.14,  5.50,  6.18,  5.08,  6.31,
             3.17,  4.90,  4.00,  5.64,  5.39,  7.45,  4.92,  6.23,  4.21,  4.60,
             2.88,  3.58,  2.29,  3.33,  2.56,  3.25,  2.60,  2.92,  2.21,  2.58,
             1.42,  1.92, -8.00,  1.84,  1.12,  1.66,  0.94,  1.77,  1.60,  2.00,
             1.00,  2.19,  1.51,  2.24,  1.07,  2.17,  1.13,  1.70,  0.58,  1.45,
            -8.00,  1.00,  0.52,  1.11,  0.28,  1.14,  0.51,  0.93,  0.00,  1.08,
             0.06,  0.88, -0.17,  1.11,  0.23,  1.25,  1.38,  1.64,  1.01,  1.13,
             0.90,  2.00,  0.65, -8.00, -8.00, -8.00, -8.00, -8.00, -8.00,  0.06,
            -8.00, -0.52, -8.00, -8.00, -8.00, -8.00, -8.00, -8.00, -8.00
        ])
        eonh = 10.0**solar_ab
        renorm = np.sum(eonh)
        eontot = eonh / renorm
        
        solar_ab = np.log10(eontot)
        solar_ab[0] = eonh[0]/renorm
        
        self.solar_ab = solar_ab
        
    def plot_mode(self,
        mode,
        ylim=(-0.1,1.1),
        include_errors=True,
        savefig='SME_PLOTS',
        show_plots=True,
        continuum=True
        ):

        """
        INPUT:
        
        mode           : Line or element you want to plot
        ylim           : (lower limit, upper limit) of yaxis
        include_errors : Includes grey area sob-uob to sob+uob in background
        savefig        : will save plot in directory passed with 'savefig'
        show_plots     : Decide if you want to see the plot in this application
        continuum=True : Continuum line at 1.0
        
        OUTPUT:
        
        Plot SME synthesis over observed spectrum, including line mask
        
        """
        
        kwargs_sob = dict(color='black', linewidth=1)
        kwargs_smod = dict(color='red', linewidth=2)
        kwargs_fill_between = dict(color='grey')

        if mode == 'all':
            gothrough_mode = self.mode
        else:
            gothrough_mode = mode

        for mode in [gothrough_mode]:
            
            try:
                file_exists = self.sme[mode+'_nseg']

                if savefig!=False:
                    pdf_pages = PdfPages(savefig+'/'+str(self.sobject_id)+'_'+mode+'.pdf')

                # Looping through each element
                for each_segment in range(self.sme[mode+'_nseg']):

                    if each_segment%2 == 0:
                        f = plt.figure(figsize=(15,10))

                    ax1 = plt.subplot(2,1,1+each_segment%2)

                    # Because wave, sob, smod, and mob come in 1D arrays, we have to match segments wavelength with wave
                    if self.sme[mode+'_nseg'] == 1:
                        in_segment = np.where((self.sme[mode+'_wave'] >= self.sme[mode+'_wran'][0]) & (self.sme[mode+'_wave'] <= self.sme[mode+'_wran'][1]))[0]
                    else:
                        in_segment = np.where((self.sme[mode+'_wave'] >= self.sme[mode+'_wran'][each_segment,0]) & (self.sme[mode+'_wave'] <= self.sme[mode+'_wran'][each_segment,1]))[0]                 

                    # Plot chi2 optimisation / line mask (defined as sme.mob == 1)
                    if self.sme[mode+'_nseg'] == 1:
                        mask_in_segment = np.where((self.sme[mode+'_mob'] == 1) & (self.sme[mode+'_wave'] >= self.sme[mode+'_wran'][0]) & (self.sme[mode+'_wave'] <= self.sme[mode+'_wran'][1]))[0]
                    else:
                        mask_in_segment = np.where((self.sme[mode+'_mob'] == 1) & (self.sme[mode+'_wave'] >= self.sme[mode+'_wran'][each_segment,0]) & (self.sme[mode+'_wave'] <= self.sme[mode+'_wran'][each_segment,1]))[0]
                    mask_kwargs = dict(color = 'yellow')
                    if len(mask_in_segment) > 0:
                        if mask_in_segment[-1] - mask_in_segment[0] == len(mask_in_segment) - 1:
                            ax1.axvspan(self.sme[mode+'_wave'][mask_in_segment[0]],self.sme[mode+'_wave'][mask_in_segment[-1]],**mask_kwargs)
                        else:
                            for each,each_mis in enumerate(mask_in_segment):
                                if each == 0:
                                    current_mask = [each_mis]
                                elif mask_in_segment[each]-mask_in_segment[each-1] > 1:
                                    current_mask = np.array(current_mask)
                                    ax1.axvspan(self.sme[mode+'_wave'][current_mask[0]],self.sme[mode+'_wave'][current_mask[-1]],**mask_kwargs)
                                    current_mask = [each_mis]
                                elif each_mis == mask_in_segment[-1]:
                                    ax1.axvspan(self.sme[mode+'_wave'][current_mask[0]],self.sme[mode+'_wave'][current_mask[-1]],**mask_kwargs)
                                else:
                                    current_mask.append(each_mis)
                    else:
                        print('No mask!')

                    # Now we can plot sob and smod
                    sob1 = ax1.plot(self.sme[mode+'_wave'][in_segment],self.sme[mode+'_sob'][in_segment],label='Spectrum',**kwargs_sob)
                    if include_errors==True:
                        fill1 = ax1.fill_between(self.sme[mode+'_wave'][in_segment],self.sme[mode+'_sob'][in_segment]-self.sme[mode+'_uob'][in_segment],self.sme[mode+'_sob'][in_segment]+self.sme[mode+'_uob'][in_segment],label='Error',**kwargs_fill_between)
                    smod1 = ax1.plot(self.sme[mode+'_wave'][in_segment],self.sme[mode+'_smod'][in_segment],label='Spectrum',**kwargs_smod)

                    # Plot continuum
                    cont_kwargs = dict(lw = 0.5, color='blue', linestyle='dashed')
                    if continuum==True:
                        ax1.axhline(1.0,label='Flux = 1.0',**cont_kwargs)

                    ax1.set_xlabel(r'$\mathrm{Wavelength~[\AA]}$')
                    ax1.set_ylabel(r'$\mathrm{Flux~[raw]}$')
                    ax1.set_ylim(ylim)
                    ax1.legend(loc = 'lower left')

                    if mode != 'Sp':
                        free_abund = self.sme[mode+'_abund']-self.solar_ab[self.sme[mode+'_ab_free']]
                        title_text = r'$\mathrm{sobject\_id:}~'+str(self.sobject_id)+r'\mathrm{~Atmosphere~values:~} T_\mathrm{eff} = '+str(round(self.sme['teff']))+'\,\mathrm{K}, \log g = '+str(0.01*round(100*self.sme['logg']))+'\,\mathrm{dex,~[Fe/H] = }'+str(0.01*round(100*self.sme['feh']))+'\,\mathrm{dex}, v_{\sin i}= '+str(0.01*round(100*self.sme['vsini']))+r'\,\mathrm{km/s,~['+mode+r'/Fe]} = '+str(0.01*round(100*free_abund))+'$'           
                    else:
                        title_text = r'$\mathrm{sobject\_id:}~'+str(self.sobject_id)+r'\mathrm{~Atmosphere~values:~} T_\mathrm{eff} = '+str(round(self.sme['teff']))+'\,\mathrm{K}, \log g = '+str(0.01*round(100*self.sme['logg']))+'\,\mathrm{dex,~[Fe/H] = }'+str(0.01*round(100*self.sme['feh']))+'\,\mathrm{dex}, v_{\sin i}= '+str(0.01*round(100*self.sme['vsini']))+r'\,\mathrm{km/s}$'           

                    ax1.set_title(title_text)
                    plt.tight_layout()

                    if (savefig!=False) & ((each_segment%2 == 1) | (each_segment == range(self.sme[mode+'_nseg'])[-1])):
                        pdf_pages.savefig(f)
                        
                if savefig!=False:
                    pdf_pages.close()
                    
                if show_plots!=True:
                    plt.close()
            except:
                print('Run for '+mode+' not found')





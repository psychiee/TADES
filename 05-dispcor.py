#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 08:17:43 2018

@author: wskang
"""
import sys, os
from glob import glob
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting 
import astropy.constants as Const
from speclib import x_helio, read_params, speccomb, speccrop
GREEKS = ['alf', 'bet', 'gam', 'del', 'eps', 'zet', 'eta', 'tet',
          'iot', 'kap', 'lam', 'mu.', 'nu.', 'ksi', 'omi', 'pi.',
          'rho', 'sig', 'tau', 'ups', 'phi', 'khi', 'psi', 'ome']

par = read_params()
os.chdir(par['WORKDIR'])

ECID_FILE = par['EIDFILE']
FITTING_ORDER = np.array(par['EIDORDER'].split(','), int)
CONORDER = int(par['CONORDER'])
CONUPREJ = float(par['CONUPREJ'])
CONLOREJ = float(par['CONLOREJ'])

#print sys.argv
if len(sys.argv) == 2:
    SPECLIST = []
    filename = sys.argv[1]
    # READ the list of files 
    if filename.startswith('@'):
        SPECLIST += list(np.genfromtxt(filename[1:], dtype='S'))
    else:
        SPECLIST += glob(filename)
else:
    SPECLIST = [] 
    for f in np.genfromtxt('obj.list', dtype='U').flatten():
        tmp = f.split('.')
        SPECLIST.append('w'+tmp[0]+'.ec.fits')

#=============================================================
# READ the wavelength data from ecidentify 
dat = np.genfromtxt(ECID_FILE)
eap, eord = np.array(dat[:,0], int), np.array(dat[:,1], int)
ewv, epix, edx, eflag = dat[:,2], dat[:,4], dat[:,5], dat[:,6]
eapset = list(set(eap))
eapset.sort()
NAP = len(eapset)

# READ pixel position, order, wavelength for wavelength calibration 
mpix, mord, mlam = [], [], []
for j in range(NAP):
    vv, = np.where(eap == eapset[j])
    mpix += list(epix[vv])
    mord += list(eord[vv])
    mlam += list(ewv[vv])
mpix, mord, mlam = np.array(mpix), np.array(mord), np.array(mlam)
# MAKE fitting model with chebychev 2D 
p_init = models.Chebyshev2D(FITTING_ORDER[0],FITTING_ORDER[1])
f = fitting.LinearLSQFitter()
p = f(p_init, mpix, mord, mlam)
dlam = mlam - p(mpix, mord)

# PLOT the fitting results 
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, num=88, figsize=(6,10))
ax1.plot(mpix, dlam, 'ro', alpha=0.2)
ax2.plot(mord, dlam, 'bo', alpha=0.2)
ax3.plot(mlam, dlam, 'go', alpha=0.2)
# DO sigma-clipping 
N, NMAX = 0, 15    
while(N < NMAX):
    N = N + 1
    dmed, dsig = np.median(dlam), np.std(dlam)
    vv, = np.where((dlam < dmed+dsig*3) & (dlam > dmed-dsig*3))
    if len(vv) == len(dlam): break
    if len(vv) < 150: break
    mpix, mord, mlam = mpix[vv], mord[vv], mlam[vv]
    p = f(p_init, mpix, mord, mlam)
    dlam = mlam - p(mpix, mord)
davg, dsig = np.mean(dlam), np.std(dlam)
ax1.plot(mpix, dlam, 'r+', ms=8, label=f'{davg:.5f}$\pm${dsig:.5f}')
ax2.plot(mord, dlam, 'b+', ms=8)
ax3.plot(mlam, dlam, 'g+', ms=8)
ax1.legend()
ax1.grid()
ax2.grid()   
ax3.grid()
fig.savefig('dispcor-result')

# LOOP for stars 
for INPUT_SPEC in SPECLIST:    
    # READ the raw spectrum 
    hdu = fits.open(INPUT_SPEC)[0]
    xhdr = hdu.header 
    xspec = hdu.data 
    
    xwv = np.zeros_like(xspec)
    
    OBJECT = xhdr.get('OBJECT')
    JD = xhdr.get('JD')
    DATEOBS = xhdr.get('DATE-OBS').replace('-','').replace(':','')
    EXPTIME = '%is' % (xhdr.get('EXPTIME'),)
    RA = xhdr.get('RA')
    DEC = xhdr.get('DEC')
    print(RA, DEC)
    # CHECK object name for Greek
    if (OBJECT[0:3].lower() in GREEKS) & (len(OBJECT) == 6):
        _OBJECT = OBJECT[0:3].lower()+' '+OBJECT[3:]
        OBJECT = OBJECT[0:3].lower()+OBJECT[3:].capitalize()
    else:
        _OBJECT = OBJECT
        
    print ('### DISPCOR ###')
    print ('OBJECT=', OBJECT)
    print ('TIME=', DATEOBS)
    print ('JD=', JD)
    try: 
        c = SkyCoord.from_name(_OBJECT)
        RA, DEC = c.ra.value, c.dec.value
        HV = x_helio(RA, DEC, jd=JD)[0]
        print ('RA, Dec= %.2f, %.2f' % (RA, DEC))
        print ('HV= %.5f' % (HV,))
    except:
        print ('******** NAME PARSING ERROR ********')
        HV = 0.0
        #continue
        
    OUTPUT = OBJECT+'-'+DATEOBS+'-'+EXPTIME
    
    # APPLY the wavelength solution into the pixel 
    xpix, xord = [], []
    for j in range(NAP):
        vv, = np.where(eap == eapset[j])
        xpix.append(np.arange(xspec.shape[1])+1)
        xord.append(np.zeros(xspec.shape[1])+eord[vv[0]])
    xwv = p(xpix, xord)*(1.0 - HV/Const.c.value*1000.0)
    
    # SAVE multispec without no correction 
    xwvm, xspecm, xapm = speccrop(xwv, xspec)
    # WRITE to the file
    foutm = open(OUTPUT+'-multi.lpx', 'w')
    for ap, wv, spec in zip(xapm, xwvm, xspecm):
        for i in range(len(wv)):
            foutm.write('%12.4f %4i %10.8f \n' % (wv[i], ap[i], spec[i]))
    foutm.close()
    
    # SAVE 1d spec with continuum correction    
    xwv1, xspec1, xap1 = speccomb(xwv, xspec, order=CONORDER, sig=[CONUPREJ, CONLOREJ])
    # WRITE to the file 
    fout1 = open(OUTPUT+'-1d.lpx', 'w')
    for k in range(xwv1.shape[0]):
        fout1.write('%12.4f %10.8f %4i \n' % (xwv1[k], xspec1[k], xap1[k]))
    fout1.close()
    
    # PLOT 1d spectrum     
    x, y, ap = xwv1, xspec1, xap1
    # with 6 rows
    NROW = 7
    fig, axs = plt.subplots(num=98, nrows=NROW, figsize=(18,24))
    x1, x2 = min(x), max(x)
    y1, y2 = max([np.nanmin(y), 0]), np.percentile(y,99.9)
    dx = (x2-x1)/NROW
    for i in range(NROW):
        x_start, x_end = x1+dx*i, x1+dx*(i+1)
        cond = (x > x_start) & (x < x_end)
        apr, xr, yr = ap[cond], x[cond], y[cond]
        axs[i].plot(xr, yr, 'b-', lw=1)
        for j in list(set(apr)):
            vv, = np.where((apr == j))
            apcut = np.max(xr[vv])
            axs[i].plot([apcut, apcut], [y1,y2], 'r-', lw=2, alpha=0.5)
            axs[i].text(apcut, y2+(y2-y1)*0.03, int(j), color='r', fontsize=12)
        axs[i].set_xlim(x_start, x_end)
        axs[i].grid()
        axs[i].set_ylim(y1, y2)
    fig.suptitle(f'1D Full Spectrum\n{OUTPUT:s}\n (JD={JD:.6f}, HV={HV:.4f})', fontsize=25)
    fig.savefig(OUTPUT+'-1d.pdf')
    fig.clf()
    
    # PLOT multi spectrum for each aperture 
    with PdfPages(OUTPUT+'-multi.pdf') as pdf:
        for x, y, ap in zip(xwvm, xspecm, xapm):
            y1, y2 = max([min(y),0]), max(y)
            fig = plt.figure(figsize=(12,5))
            ax = fig.add_subplot(111)
            ax.plot(x, y, 'b-')
            yfit = xspec1[(xwv1 >= min(x)) & (xwv1 <= max(x))]
            ax.plot(x, y/yfit, 'r-', lw=3, alpha=0.5)
            ax.set_title(f'Aperture RAW - AP{ap[0]:02.0f}\n{OUTPUT} (JD={JD:.6f}, HV={HV:.4f})')
            ax.grid()
            ax.set_ylim(y1-(y2-y1)*0.1, y2+(y2-y1)*0.1)
            pdf.savefig(fig)
            plt.close(fig)
                    
    plt.close('all')


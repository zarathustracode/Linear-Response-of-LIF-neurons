#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt

#import time
#from tqdm import tqdm, tqdm_gui, tqdm_notebook

from tqdm import tqdm_notebook as tqdm
from scipy.optimize import curve_fit

from ioLIF import *


r0, r1, fh, rho, cov = frt(18, 5., -60., -40., -60., 20., 2.,0.1, 0.314, 1e-05)

print(r0*1000., r1*1000.)

plt.figure()

time = np.arange(0., 10001., 1.)

rate = r0 + np.abs(r1)*np.cos(0.01 *time +np.angle(r1))

plt.plot(time, 1000.*rate)
plt.xlabel("t[ms]")
plt.ylabel("r(t)")

plt.ylim([0,20])



fl = 0.0001/1000
fh = 1000/1000
nfpos = 10000
fpos = np.exp(np.linspace(np.log(fl), np.log(fh), nfpos))
#fpos = np.linspace(fl, fh, nfpos)

f = np.hstack((-fpos[::-1],0,fpos))
nf = len(f)
fHz = f*1000
w = 2.*np.pi*f

dw = ( np.append(np.diff(w),w[nf-1]-w[nf-2]) + np.append(w[1]-w[0],np.diff(w)) )/2.
dw0= dw[w==0]
dw0 = float(dw0)


rhat = []
fhat = []
rhohat = []
covhat = []


for ww in tqdm(w):
    r0, r1, fh, rho, cov = frt(15., 4.0, -60., -40., -60., 20., 2.,0.1, ww, dw0)
    rhat.append(r1)
    fhat.append(fh)
    rhohat.append(rho)
    covhat.append(cov)
    
print("DONE", r0*1000.)
    


rhat = np.array(rhat)
fhat = np.array(fhat)
rhohat = np.array(rhohat)
covhat = np.array(covhat)

powerr = np.abs(rhat)
powerh = np.abs(fhat)
powerrho = np.abs(rhohat)
powercov = np.abs(covhat )
    
timex, xfilter = GetTheFilter(dw, w, rhat, t1=-10., t2=200, dt=1.0)

timeh, hfilter = GetTheFilter(dw, w, fhat, t1=-10, t2=200, dt=1.0)

timer, rhofilter = GetTheFilter(dw, w, rhohat, t1=-10, t2=300, dt=1.0)

timec, covfilter = GetTheFilter(dw, w, covhat, t1=-5, t2=5, dt=0.1)




plt.figure(figsize=(16,8))

plt.subplot(241)

plt.semilogx(fHz,powerr)

plt.xlabel("Frequency [Hz]")
plt.ylabel("Rate Power [Hz]")

plt.subplot(242)

#plt.plot(time, rhofilter)
plt.semilogx(fHz,powerh)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Hazard Power [Hz]")

plt.subplot(243)

#plt.plot(time, rhofilter)
plt.semilogx(fHz,powerrho)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Renewal Power [Hz]")

plt.subplot(244)

#plt.plot(time, covfilter)
plt.semilogx(fHz,powercov)

plt.xlabel("Frequency [Hz]")
plt.ylabel("Covariance Power [Hz]")

plt.subplot(245)

plt.plot(timex, 1000.*(r0+np.real(xfilter)))
plt.xlabel("Time [ms]")
plt.ylabel("Rate [Hz]")
#plt.xlim([-10,60])

plt.subplot(246)

plt.plot(timeh, 1000.*np.real(hfilter))
plt.xlabel("Time [ms]")
plt.ylabel("Hazard rate [Hz]")

plt.subplot(247)

plt.plot(timer, 1000.*np.real(rhofilter))
plt.xlabel("Time [ms]")
plt.ylabel("Renewal rate [Hz]")

plt.subplot(248)

plt.plot(timec, 1000.*np.real(covfilter))

#plt.xlim([-10,10])
plt.xlabel("Time [ms]")
plt.ylabel("Covariance [Hz]")

plt.tight_layout()

plt.savefig("Respose functions.pdf")

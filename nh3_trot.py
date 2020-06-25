# --------------- NH3 spectrum generator and Trot calculation -------------------
#    Calculating T_rot from NH3 (1,1) and (2,2) lines
#    Created on Fri Jan 5 14:02:20 2016
#    Latest editing: June 10, 2020
#    @author: Zhiyuan Ren, Shen Wang 

import matplotlib.pyplot as plt
import numpy as np
import pyspeckit
from astropy import units as u
from pyspeckit.spectrum.models import ammonia_constants, ammonia, ammonia_hf
from pyspeckit.spectrum.models.ammonia_constants import freq_dict
from pyspeckit.spectrum.units import SpectroscopicAxis, SpectroscopicAxes
from decimal import Decimal
from scipy import stats
from astropy.modeling import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
import nh3_hfgr as hfgr

# %% -------- physical parameters ---------
Tex = 28.0
Tex_ini = 28
N_tot = 1.5e15
tau_1 = N_tot * 9.1288e-15 * np.exp(-23.279 / Tex) / (0.7604 + 0.0519 * Tex + 0.00094361 * Tex ** 2)
tau_2 = 2 / 1.26 * 0.6 * tau_1 * np.exp(-40.99 / Tex)

# %% -------- spectral parameters ---------
dv = dv_theory = 1.5
cw = cw1 = cw2 = 0.1
vs_theory = np.arange(-30.0, 30.0, cw)
vs_cal = vs_theory - 0.0
rms = 0.3

T11_thy = hfgr.T11_theory(vs_theory, dv_theory, tau_1, Tex=Tex, osg1=1, osg2=1, mg=1, isg1=1, isg2=1)
T22_thy = hfgr.T22_theory(vs_theory, dv_theory, tau_2, Tex=Tex, osg1=1, osg2=1, mg=1, isg1=1, isg2=1)
Tb_11 = T11_thy + ((np.random.randn(len(vs_cal)) - 0.0) * rms)
Tb_22 = T22_thy + ((np.random.randn(len(vs_cal)) - 0.0) * rms)

# %% ------------- spectra sampling ------------------
Trot = hfgr.trot_fun(vs_theory, Tb_11, vs_theory, Tb_22)
print('Trot=',Trot)

# %% -----------  single noise spectra plot: -------------
T11 = T11_thy + ((np.random.randn(len(vs_cal)) - 0.0) * 0.3)
T22 = T22_thy + ((np.random.randn(len(vs_cal)) - 0.0) * 0.3)

T11_obs = T11 * 0.6
T11_fit = T11_thy * 0.6
T11_bsl = T11 * 0 + 7
T22_obs = T22 * 0.6
T22_fit = T22_thy * 0.6
T22_bsl = T22 * 0

fig = plt.figure(figsize=(7.5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.fill_between(vs_cal, T11_obs+T11_bsl, T11_bsl, color='skyblue')
ax.plot(vs_cal, T11_obs + T11_bsl, c='black', drawstyle='steps', linewidth=1)
ax.plot(vs_cal, T11_fit + T11_bsl, c='red', linewidth=1)
ax.fill_between(vs_cal, T22_obs, T22_bsl, color='skyblue')
ax.plot(vs_cal, T22_obs + T22_bsl, c='black', drawstyle='steps', linewidth=1)
ax.plot(vs_cal, T22_fit + T22_bsl, c='red', linewidth=1)
ax.set_xlabel('Velocity (km/s)')
ax.set_ylabel('Intensity (K)')
ax.axis([-32, 32, -1, 29])
plt.savefig('figure.pdf', format='pdf')
plt.show()


#%% thermal motion:
Trot = 30;
dv = np.sqrt(kb * Trot / 28 * m0 / 2.33)
np.sqrt(dv)

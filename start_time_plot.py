import math
import numpy as np
from scipy.io import loadmat
import scipy.signal as signal
from matplotlib import pyplot as plt
import myModules.PRK as prk
import myModules.plot as myplot



#######################################
file_params = 'data/kinetics_params.mat'
file_observations = 'data/2016-10-19_CR worth_600mm 0,1s.TKA'
tstep_observation = 0.1 # seconds
control_rod_height = 600 # mm
reactivities_pcm = np.array([
        [200, 11], # [mm, pcm]
        [400, 52],
        [600, 112],
        [800, 154],
        [1000, 165],
        ])
stdev_reactivity_dollars = 0.008
stdev_transition = 1e-2
#######################################


                                           
params = loadmat(file_params)
crocus = prk.PointReactor(params['LAMBDA_MEAN'].flatten()[1:], params['BETA_MEAN'].flatten()[1:], params['GEN_TIME_MEAN'].flatten()[0])
crocus.set_include_params(True)
uncertainties = prk.Fuel(params['LAMBDA_STD'].flatten()[1:], params['BETA_STD'].flatten()[1:], params['GEN_TIME_STD'].flatten()[0])

reactivity_unitless = + 1.0e-5 * np.interp(control_rod_height, reactivities_pcm[:,0], reactivities_pcm[:,1])
stdev_reactivity_unitless = crocus.dollars_to_unitless(stdev_reactivity_dollars)

observations = np.loadtxt(file_observations)
observations = observations[:np.argmax(observations)]
observations = observations[(observations <= 0).nonzero()[0][-1]+1:1000]
times = np.arange(len(observations)) * tstep_observation

fig = plt.figure(figsize=[myplot.figure_width, myplot.figure_width*0.5])
ax = plt.gca()
ax.plot(times, observations, 'b', label='Experimental observations')

ax.set(xlabel=r'Time [s]', ylabel=r'neutron count (per $\Delta t$)', title=r"Estimation of the time of reactivity insertion.")
#ax.xlabel(r'$t' + myplot.unit('s') + r'$')
#plt.ylabel(r'neutron count (per $\Delta t$)')

b,a = signal.butter(5, 0.05)
observations_smoothed = signal.filtfilt(b,a, observations)
ax.plot(times, observations_smoothed, 'y', linewidth=3, label='Smoothed signal')

max_resting = max(observations_smoothed[:min(50, math.ceil(5/tstep_observation))])

ind_start = np.asarray(observations_smoothed >= 2*max_resting).nonzero()[0][0]
ax.axvline(ind_start * tstep_observation, color='k', label='Approximate time of reactivity insertion', ymax = 0.7)

myplot.legend(ax)
fig.savefig('time_of_rod_drop.pdf')

#tup = (1, 3, "spam")
#print(repr(tup))
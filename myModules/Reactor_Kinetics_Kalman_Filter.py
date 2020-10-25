# -*- coding: utf-8 -*-
"""
Author: Felix Grimberg

Runs with Python 3.7.3.final.0 and pykalman 0.9.5, installed using Anaconda3
         (conda version : 4.6.14
    conda-build version : 3.17.8
         python version : 3.7.3.final.0)
on Windows 10.

Next:
    Try EKF without observations
    Add final estimates/uncertainties?

200116-1800:
    In EKF.filter, multiply stdev_transition_dep by state_pred[:nvars] BEFORE squaring, rather than after.
    
    This reflects the multiplication of state_end[:nvars] with (1 + noise) in PointReactor.rod_drop: We add a term of variance var[noise] * state_end**2.
    
    Check successful
    
191003-1050:
    Replaced np.dot by @ in EKF.filter (for COV update equation). That changed the behavior. Previous behavior was incorrect.
    Check successful.
"""
# %% Set-up

import math

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import traceback
#import scipy.stats

import pykalman

import myModules.config as config
import myModules.PRK as prk
import myModules.plot as myplot
import myModules.util as util
import myModules.extended_kalman_filter as ekf

config.include_reactivity = True
config.usemask = False
config.stdev_initial_factor = 0.5
config.stdev_transition_dep = 1e-3
config.use_EKF = True
config.use_UKF = True

np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float': lambda x: '0' if x == 0 else '{:.6e}'.format(x)})


# %% Initialize kalman filters
file_params = 'data/kinetics_params.mat'
file_observations = 'data/2016-10-19_CR worth_600mm 0,1s.TKA'
tstep_observation = 0.1 # seconds
keep_every = 1
tstep_filter = tstep_observation * keep_every
last_observation_time = 300 # s
control_rod_height = 600 # mm
reactivities_pcm = np.array([
        [200, 11], # [mm, pcm]
        [400, 52],
        [600, 112],
        [800, 154],
        [1000, 165],
        ])
stdev_reactivity_dollars = 0.008
#######################################


                                           
params = loadmat(file_params)
crocus = prk.PointReactor(params['LAMBDA_MEAN'].flatten()[1:], params['BETA_MEAN'].flatten()[1:], params['GEN_TIME_MEAN'].flatten()[0])
crocus.set_include_params(True)
uncertainties = prk.Fuel(params['LAMBDA_STD'].flatten()[1:], params['BETA_STD'].flatten()[1:], params['GEN_TIME_STD'].flatten()[0])

reactivity_unitless = + 1.0e-5 * np.interp(control_rod_height, reactivities_pcm[:,0], reactivities_pcm[:,1])
stdev_reactivity_unitless = crocus.dollars_to_unitless(stdev_reactivity_dollars)

observations = util.extract_observations(file_observations, tstep_observation)[:math.ceil(last_observation_time / tstep_observation):keep_every]
times = np.arange(len(observations)) * tstep_filter

def f_observation(state, noise_std_gaussian):
    """
    Given the true state of the system and a realisation of the standard Gaussian distribution, this function returns a noisy observation.
    The observed quantity is a neutron count, which follows a Poisson-distribution whose mean is proportional to the true neutron flux.
    It would be possible to build a function whose output is truly Poisson-distributed with parameter state[0] by returning scipy.stats.poisson.ppf(scipy.stats.norm.cdf(noise_std_gaussian), state[0]).
    However, the UKF will approximate f_observation with a Gaussian distribution anyway. It is hence futile to construct a truly Poisson-distributed output. Instead, the output of this function follows a Gaussian distribution with mean and variance state[0], which approximates the Poisson-distribution well for means > 15, except in the tails.
    
    INPUTS:
        state: The true state of the reactor.
        noise_std_gaussian: A realisation of the univariate standard Gaussian distribution.
        
    OUTPUTS:
        observation: A realisation of the univariate Gaussian distribution with mean and variance state[0]. observation is calculated from the inputs in a purely deterministic fashion.
    """
    if state[0] <= 0:
        # The neutron count will be 0 if the neutron population is 0.
        return 0
    observation = state[0] + math.sqrt(state[0]) * noise_std_gaussian
    # The neutron count is a non-negative number.
    return max([observation, 0]) #1e-4])
# Noise generated from N(0, COV_observation) will be passed to f_observation as noise_std_gaussian --> COV_observation must be 1.
COV_observation = np.ones((1,1))

def f_transition(state, noise_transition):
    """
    Given the true state of the system and a realisation of the (multivariate Gaussian) transition noise distribution, this function returns the next true state following the noisy PRK model. It makes use of the PointReactor.rod_drop method.
    
    INPUTS:
        state: True state of the system.
        noise_transition: Random vector drawn from the transition noise distribution.
        
    OUTPUTS:
        state_new: True state of the system at the next time step. Note: state_new is computed from the inputs in an entirely deterministic manner.
    """
    reactivity = None
    if not config.include_reactivity:
        reactivity = reactivity_unitless
    return crocus.rod_drop(
            (0, tstep_filter),
            initial_state=state,
            noise=noise_transition,
            store_time_steps=False,
            reactivity_unitless=reactivity
            )

COV_transition = np.diag(np.full(crocus.state_dims, (config.stdev_transition_dep * keep_every)**2 ))

# The initial state is unknown --> huge covariance
state_initial = crocus.stationary_PRKE_solution(15, #observations[0],
                                                reactivity=reactivity_unitless)
COV_initial = np.diag( np.power(
        np.concatenate((
                state_initial[:crocus.nvars] * config.stdev_initial_factor,
                uncertainties.param_values(stdev_reactivity_unitless)
                )),
        2) )


UKF = pykalman.UnscentedKalmanFilter(
        transition_functions = f_transition,
        observation_functions = f_observation,
        transition_covariance = COV_transition,
        observation_covariance = COV_observation,
        initial_state_mean = state_initial,
        initial_state_covariance = COV_initial,
        )

EKF = ekf.EKF(crocus, tstep_filter)


# %% Apply the kalman filters

# Select how many of the observations will be treated:
test_length = len(observations)
observations = observations[:test_length]
times = times[:test_length]

# Create a new figure
fig, axes_list = plt.subplots(nrows=3, figsize=[myplot.figure_width, myplot.figure_width*1.4], gridspec_kw={'hspace':0.35})
ax = axes_list[0]
ax_err = axes_list[1]
ax_SE = axes_list[2]
if np.max(observations) < 350:
    neutron_plot = ax.plot
    err_plot = ax_err.plot
    ax_err.set_ylim(bottom=0, top=np.sqrt(np.max(observations)) * 1.35)
    SE_plot = ax_SE.plot
else:
    neutron_plot = ax.semilogy
    err_plot = ax_err.semilogy
    SE_plot = ax_SE.plot

# Plot the measured neutron count and its uncertainty:
neutron_plot(times, observations, myplot.Label_UKF.fmt_str['observation'], label=r'$z^{(i)}$')
err_plot(times, np.sqrt(observations), color=myplot.Label_UKF.fmt_str['observation'][0], label=r'$\sigma_{observation}$')

# Compute and plot the neutron count as precited:
if not config.usemask:
    end_state, sol_theory = crocus.rod_drop((0, times[-1]), reactivity_unitless=reactivity_unitless, initial_neutron_population=observations[0])
    neutron_plot(sol_theory.times, sol_theory.neutron_populations(), myplot.Label_UKF.fmt_str['model'], label=r'$n_{PRK}(t)$')

exceptions = []

if config.usemask and (config.use_EKF or config.use_UKF):
    # Apply the Kalman filter:
    print('\n\nApplying the Unscented Kalman Filter (masked observations)---------------------------')
    t0 = time.time()
    observations_masked = np.ma.masked_where(True, observations)
    try:
        states_noupd, COVs_noupd = UKF.filter(observations_masked)
        print("Kalman filter was successfully applied to {0} time steps in {1} seconds.".format(test_length, time.time() - t0))
    except:
        raise
        exceptions.append(sys.exc_info())
        states_UKF, COVs_UKF = (None, None)
    else:
        neutron_plot(times, states_noupd[:,0], myplot.Label_UKF.fmt_str['model'], label=r'$n_{PRK}(t)$')
        err_plot(times, np.sqrt(COVs_noupd[:,0,0]), myplot.Label_UKF.fmt_str['model'], label=myplot.Label_UKF.make('sigma', 'PRK'))
        SE_plot(times, np.absolute(states_noupd[:,0] - observations) / observations, myplot.Label_UKF.fmt_str['model'], label=myplot.squared_errorize_RKKF('n_{PRK}(t^{(i)})'))
        

# Apply the Extended Kalman filter:
if config.use_EKF:
    
    print('\n\nApplying the Extended Kalman Filter ---------------------------')
    t0 = time.time()
    if not config.include_reactivity:
        raise ValueError('This part (EKF without reactivity estimation) is not coded.')
    try:
        states_EKF, COVs_EKF = EKF.filter(observations, state_initial, COV_initial)
        print("Extended Kalman filter was successfully applied to {0} time steps in {1} seconds.".format(test_length, time.time() - t0))
    except:
        exceptions.append(sys.exc_info())
        states_EKF, COVs_EKF = (None, None)
    else:
        neutron_plot(times, states_EKF[:,0], myplot.Label_UKF.fmt_str['EKF'], label=r'$n_{EKF}(t)$', linewidth=2.5)
        err_plot(times, np.sqrt(COVs_EKF[:,0,0]), myplot.Label_UKF.fmt_str['EKF'], label=myplot.Label_UKF.make('sigma', 'EKF'))
        SE_plot(times, np.absolute(states_EKF[:,0] - observations) / observations, myplot.Label_UKF.fmt_str['EKF'], label=myplot.squared_errorize_RKKF('n_{EKF}(t^{(i)})'))

# Apply the Unscented Kalman filter:
if config.use_UKF:

    print('\n\nApplying the Unscented Kalman Filter ---------------------------')
    t0 = time.time()
    try:
        states_UKF, COVs_UKF = UKF.filter(observations)
        print("Unscented Kalman filter was successfully applied to {0} time steps in {1} seconds.".format(test_length, time.time() - t0))
    except:
        exceptions.append(sys.exc_info())
        states_UKF, COVs_UKF = (None, None)
    else:
        neutron_plot(times, states_UKF[:,0], myplot.Label_UKF.fmt_str['UKF'], label=r'$n_{UKF}(t)$', linewidth=2.5)
        err_plot(times, np.sqrt(COVs_UKF[:,0,0]), myplot.Label_UKF.fmt_str['UKF'], label=myplot.Label_UKF.make('sigma', 'UKF'))
        SE_plot(times, np.absolute(states_UKF[:,0] - observations) / observations, myplot.Label_UKF.fmt_str['UKF'], label=myplot.squared_errorize_RKKF('n_{UKF}(t^{(i)})'))


print("\n\n\n{} exceptions occured in the call to KF.filter_update and UKF.filter:\n".format(len(exceptions)))
for exc in exceptions:
    traceback.print_tb(exc[2]) # Traceback object
    print(repr(exc[0]) + ":    " + repr(exc[1])) # Error class and message

# Add legend, labels, annotations, title; show and save fig
ax.set(xlabel="Time [s]", ylabel=r'Neutron count [1 per $\Delta t$]', title=r"Neutron count vs. its various estimates")
myplot.legend(ax)
myplot.annotate_RKKF(ax, config.stdev_initial_factor, config.stdev_transition_dep)
#myplot.annotate_PRK(ax, tstep_observation, reactivity_unitless)#, loc_y=0.7, loc_x=0.1)
ax_err.set(xlabel="Time [s]", ylabel=r'Standard deviation [1 per $\Delta t$]', title=r"Estimated uncertainty of the estimates of the neutron count")
myplot.legend(ax_err)
myplot.annotate_RKKF(ax_err, config.stdev_initial_factor, config.stdev_transition_dep, loc_x=0.27)
#ax_SE.set_ylim(bottom=1e-2)
ax_SE.set(xlabel="Time [s]", ylabel=r'Relative error [untiless]', title=r"Relative error of the estimates of the neutron count")
myplot.legend(ax_SE)
myplot.annotate_RKKF(ax_SE, config.stdev_initial_factor, config.stdev_transition_dep, loc_x=0.29)

plt.show()
fig.savefig('neutron_population.pdf', bbox_inches='tight')

# %% Plot parameter estimates

t0 = time.time()

# Create a new figure
nrows = 3
nfigs = math.ceil((crocus.fuel.ngroups() + 2) / nrows)
figs, axes_leftcol, axes_rightcol = [], [], []
for fig_num in range(nfigs):
    # Create a figure
    fig, axes_list = plt.subplots(nrows=nrows, ncols=2, figsize=[myplot.figure_width, myplot.figure_width * 1.414], gridspec_kw={'hspace': 0.8, 'top': 0.84, 'wspace': 0.25})
    fig.suptitle(u'Estimation of independent variables with UKF and EKF ({fig_num}/{nfigs})\n(${sigmas[0]}$, ${sigmas[1]}$)'.format(fig_num=fig_num+1, nfigs=nfigs, sigmas=myplot.annot_strs_RKKF(config.stdev_initial_factor, config.stdev_transition_dep)), fontsize=22)
    if fig_num == 0 and config.include_reactivity:
        # Replace the top two axes by one wider axis:
        for ax in axes_list[0,:]:
            ax.remove()
        ax_top = fig.add_subplot(axes_list[1,1].get_gridspec()[0,:])
        axes_list = axes_list[1:, :]
    #Sore the axes and the figure
    figs.append(fig)
    axes_leftcol = np.concatenate((axes_leftcol, axes_list[:,0]))
    axes_rightcol = np.concatenate((axes_rightcol, axes_list[:,1]))

# Remove unused axes:
axes_leftcol = util.remove_excess_axes(axes_leftcol, crocus.fuel.ngroups() + 1)
axes_rightcol = util.remove_excess_axes(axes_rightcol, crocus.fuel.ngroups() + 1)
# Plot estimates of the parameters:
axes_list_shuffle = np.concatenate((
        # For the beta_i :
        axes_leftcol[1:],
        # For the lambda_i :
        axes_rightcol[1:],
        # For Lambda :
        axes_rightcol[0:1]
        ))
if config.include_reactivity:
    axes_list_shuffle = np.insert(axes_list_shuffle, 0, ax_top)

for ax, parname, index, prior, prior_stdev in zip(
        axes_list_shuffle,
        crocus.fuel.param_names(),
        range(crocus.nvars, crocus.state_dims),
        crocus.fuel.param_values(reactivity_unitless),
        uncertainties.param_values(stdev_reactivity_unitless)
        ):
    myplot.plot_param(ax, parname, times, states_UKF, states_EKF, prior, prior_stdev, index=index, COVs_UKF=COVs_UKF, COVs_EKF=COVs_EKF)
#######################################
#    if index > 9:
#        break
myplot.plot_param(axes_leftcol[0], 'beta', times, states_UKF[:,crocus.slc_beta_l].sum(axis=1), states_EKF[:,crocus.slc_beta_l].sum(axis=1), params['BETA_MEAN'].flatten()[0], params['BETA_STD'].flatten()[0])

## Write out the final value for each parameter:
#ax = axes_list2[0,0]
#ax.axis('off')
#ax.set_title(r'\underline{Final estimates of the indep. variables}', loc='right', pad=12)
#def my2str(x):
#    tmp_str = str(x * 1e5)
#    words = tmp_str.split('.')
#    if len(words[0]) >= 5:
#        return words[0]
#    return tmp_str[:6]
#def complete_line(ind, state):
#    beta_l = state[crocus.slc_beta_l]
#    lambda_l = state[crocus.slc_lambda_l]
#    return r' & \beta_{0} = {1} & \lambda_{0} = {2} {3}\\ '.format(ind + 1, my2str(beta_l[ind]), my2str(lambda_l[ind]), myplot.unit(r's^{-1}'))
#def make_eqnarray(state):
#    if config.include_reactivity:
#        rhostr = r'\rho = ' + my2str(state[crocus.ind_rho])
#    else:
#        rhostr = ''
#    first_column = [r'\begin{eqnarray*} \textbf{All values \hspace{4mm}}',
#                    r'\mathbf{\times 10^{-5}} \hspace{7mm}',
#                    '',
#                    rhostr,
#                    r'\beta = ' + my2str(state[crocus.slc_beta_l].sum()),
#                    r'\Lambda = ' + my2str(state[crocus.ind_Lambda]) + myplot.unit('s')
#                    ]
#    if len(first_column) < crocus.fuel.ngroups():
#        first_column[len(first_column):crocus.fuel.ngroups()] = [''] * (crocus.fuel.ngroups() - len(first_column))
#    s = ''
#    for ind in range(crocus.fuel.ngroups()):
#        s = s + first_column[ind] + complete_line(ind, state)
#    s = s + r'\end{eqnarray*}'
#    return s
#
#txt = make_eqnarray(states_UKF[-1,:])
#ax.text(0.4, 0.5, txt, transform=ax.transAxes, horizontalalignment='center', verticalalignment='center')#, bbox={'edgecolor': 'k', 'facecolor': 'w'})

for fig_num in range(nfigs):
    figs[fig_num].savefig('UKF_indep_{}.pdf'.format(fig_num), bbox_inches='tight')
    plt.show(fig)

print('---- Made figure UKF_parameters.pdf in {0} s'.format(time.time() - t0))

#with open('independent_variables_final.tex', 'w') as f:
#    f.write(repr(config.config_dict()))
#    f.write('Means:\n' + txt + '\n')
#    f.write('Uncertainties (standard deviations):\n' + make_eqnarray(np.sqrt(np.diag(COVs_UKF[-1,:,:]))))

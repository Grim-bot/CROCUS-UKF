# -*- coding: utf-8 -*-
"""
Author: Felix Grimberg

Runs with Python 3.7.3.final.0 and pykalman 0.9.5, installed using Anaconda3
         (conda version : 4.6.14
    conda-build version : 3.17.8
         python version : 3.7.3.final.0)
on Windows 10.
A working LaTeX installation is required (for the matplotlib usetex option in myModules.plot). Runs with MikTeX 2.9, installed using protext.

Next:
    WHEN VARYING THE TIME STEP, ADAPT STDEV_TRANSITION_DEP
    Replace polycollection with BW-readable alternative?
    Vary the time step!

200207-1129:
    Mostly alter the neutron plots.
    Check successful.

200130-2024:
    BUGFIX to EKF: Start with an update step instead of a prediction step.
    After all, the initial guess concerns the state at the time of the initial measurement.
    Changes only to EKF.filter (UKF was already good thanks to pykalman)
    Also switch plot colors.
    
    Check successful.

200126-1635:
    Vary the time step (incl. refactoring, part 3)
    Wrap the application of the Kalman filters in a class so it's easy to repeat.
    

200123-1844:
    Make the plots BW-readable.
    Check successful.

200123-1701:
    Refactoring, part 2: Separate the storage of KF results from the actual plotting. This and part 1 make it easier to alter the plots.
    Check successful.

200123-1201:
    Refactoring, part 1: Create function to plot the estimates, uncertainty estimates, and relative errors of the neutron population (code smell: duplicated code).
    Check successful.

200122-1338:
    Add option to solve rod drop analytically (using matrix exponentiation). Note that that is slower than the dopri5 solver.
    Check successful.

200116-1800:
    In EKF.filter, multiply stdev_transition_dep by state_pred[:nvars] BEFORE squaring, rather than after.
    
    This reflects the multiplication of state_end[:nvars] with (1 + noise) in PointReactor.rod_drop: We add a term of variance var[noise] * state_end**2.
    
    Check successful.
    
191003-1050:
    Replaced np.dot by @ in EKF.filter (for COV update equation). That changed the behavior. Previous behavior was incorrect.
    Check successful.
"""
# %% Set-up

import math, os

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

def set_config():
    config.include_reactivity = True
    config.usemask = False
    config.stdev_initial_factor = 0.5
    config.stdev_transition_dep = 1e-3
    config.use_EKF = True
    config.use_UKF = True
    config.integration = config.Integration.SOLVER
    
    np.set_printoptions(precision=3)
    np.set_printoptions(formatter={'float': lambda x: '0' if x == 0 else '{:.6e}'.format(x)})



# %% Initialize kalman filters

class Simulation:
    file_params = os.path.join('data', 'kinetics_params.mat')
    file_observations = os.path.join('data', '2016-10-19_CR worth_600mm 0,1s.TKA')
    tstep_observation = 0.1 # seconds
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
                                           
    params = loadmat(os.path.join(os.getcwd(), file_params))
    crocus = prk.PointReactor(params['LAMBDA_MEAN'].flatten()[1:], params['BETA_MEAN'].flatten()[1:], params['GEN_TIME_MEAN'].flatten()[0])
    crocus.set_include_params(True)
    uncertainties = prk.Fuel(params['LAMBDA_STD'].flatten()[1:], params['BETA_STD'].flatten()[1:], params['GEN_TIME_STD'].flatten()[0])

    reactivity_unitless = + 1.0e-5 * np.interp(control_rod_height, reactivities_pcm[:,0], reactivities_pcm[:,1])
    stdev_reactivity_unitless = crocus.dollars_to_unitless(stdev_reactivity_dollars)
    # Noise generated from N(0, COV_observation) will be passed to f_observation as noise_std_gaussian --> COV_observation must be 1.
    COV_observation = np.ones((1,1))

    # The initial state is unknown --> huge covariance
    state_initial = crocus.stationary_PRKE_solution(15, #observations[0],
                                                    reactivity=reactivity_unitless)
    COV_initial = np.diag( np.power(
            np.concatenate((
                    state_initial[:crocus.nvars] * config.stdev_initial_factor,
                    uncertainties.param_values(stdev_reactivity_unitless)
                    )),
            2) )

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
        
    def __init__(self, keep_every):
        self.tstep = Simulation.tstep_observation * keep_every
        self.observations = util.extract_observations(
                Simulation.file_observations,
                Simulation.tstep_observation
                )[:math.ceil(
                        Simulation.last_observation_time
                        / Simulation.tstep_observation
                        ):keep_every]
        self.times = np.arange(len(self.observations)) * self.tstep

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
                reactivity = Simulation.reactivity_unitless
            return Simulation.crocus.rod_drop(
                    (0, self.tstep),
                    initial_state=state,
                    noise=noise_transition,
                    store_time_steps=False,
                    reactivity_unitless=reactivity
                    )
    
        COV_transition = np.diag(np.full(
                Simulation.crocus.state_dims,
                (config.stdev_transition_dep * keep_every)**2
                ))


        self.UKF = pykalman.UnscentedKalmanFilter(
                transition_functions = f_transition,
                observation_functions = Simulation.f_observation,
                transition_covariance = COV_transition,
                observation_covariance = Simulation.COV_observation,
                initial_state_mean = Simulation.state_initial,
                initial_state_covariance = Simulation.COV_initial,
                )
        self.EKF = ekf.EKF(Simulation.crocus, self.tstep)

# %% Apply the kalman filters

class NeutronPopulationResults:
    def __init__(self, times, observations):        
        # Store the times and observations
        self.times = times
        self.observations = observations
        self.method_results = [self.MethodResult(myplot.MethodType.observation, observations, observations)]
    
    def add(self, method_type, states, COVs):
        self.method_results.append(self.MethodResult(method_type, states[:,0], COVs[:,0,0]))

    class MethodResult:
        def __init__(self, method_type, means, variances):
            self.method_type = method_type
            self.means = means
            self.variances = variances


def run_simulations(simulations):
    exceptions = []
    
    for simulation in simulations:
        simulation.neutron_population_results = NeutronPopulationResults(
                simulation.times, simulation.observations
                )
    
        # Apply the Extended Kalman filter:
        if config.use_EKF:
            
            print('\n\nApplying the Extended Kalman Filter ---------------------------')
            t0 = time.time()
            if not config.include_reactivity:
                raise ValueError(
                'This part (EKF without reactivity estimation) is not implemented.'
                )
            try:
                simulation.states_EKF, simulation.COVs_EKF = simulation.EKF.filter(
                        simulation.observations,
                        Simulation.state_initial,
                        Simulation.COV_initial
                        )
                print("Extended Kalman filter was successfully applied to {0} time steps in {1} seconds.".format(len(simulation.observations), time.time() - t0))
            except:
                print("Exception occurred during EKF execution with tstep = {}".format(simulation.EKF.tstep))
                exceptions.append(sys.exc_info())
                simulation.states_EKF, simulation.COVs_EKF = (None, None)
            else:
                simulation.neutron_population_results.add(myplot.MethodType.EKF, simulation.states_EKF, simulation.COVs_EKF)
    
        # Apply the Unscented Kalman filter:
        if config.use_UKF:
        
            print('\n\nApplying the Unscented Kalman Filter ---------------------------')
            t0 = time.time()
            try:
                simulation.states_UKF, simulation.COVs_UKF = simulation.UKF.filter(simulation.observations)
                print("Unscented Kalman filter was successfully applied to {0} time steps in {1} seconds.".format(len(simulation.observations), time.time() - t0))
            except:
                print("Exception occurred during UKF execution with tstep = {}".format(simulation.EKF.tstep))
                exceptions.append(sys.exc_info())
                simulation.states_UKF, simulation.COVs_UKF = (None, None)
            else:
                simulation.neutron_population_results.add(myplot.MethodType.UKF, simulation.states_UKF, simulation.COVs_UKF)
    return exceptions

    # %% Plot estimates and estimated uncertainty of neutron population
def plot_neutron_population(simulation):
    neutron_population_plot = myplot.NeutronPopulationPlot(
            simulation.neutron_population_results
            )
    return neutron_population_plot.plot_n_shaded()
    
    # %% Plot parameter estimates
def plot_parameters(simulation):
    print("-"*10 + "tstep: {:.2f}".format(simulation.tstep) + "-"*10)

    # Create a new figure
    nrows = 3
    nfigs = math.ceil((Simulation.crocus.fuel.ngroups() + 2) / nrows)
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
    axes_leftcol = util.remove_excess_axes(axes_leftcol, Simulation.crocus.fuel.ngroups() + 1)
    axes_rightcol = util.remove_excess_axes(axes_rightcol, Simulation.crocus.fuel.ngroups() + 1)
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
            Simulation.crocus.fuel.param_names(),
            range(Simulation.crocus.nvars, Simulation.crocus.state_dims),
            Simulation.crocus.fuel.param_values(Simulation.reactivity_unitless),
            Simulation.uncertainties.param_values(
                    Simulation.stdev_reactivity_unitless
                    )
            ):
        myplot.plot_param(ax, parname, simulation.times, simulation.states_UKF, simulation.states_EKF, prior, prior_stdev, index=index, COVs_UKF=simulation.COVs_UKF, COVs_EKF=simulation.COVs_EKF)
    #######################################
    #    if index > 9:
    #        break
    myplot.plot_param(axes_leftcol[0], 'beta', simulation.times, simulation.states_UKF[:,simulation.crocus.slc_beta_l].sum(axis=1), simulation.states_EKF[:,simulation.crocus.slc_beta_l].sum(axis=1), Simulation.params['BETA_MEAN'].flatten()[0], Simulation.params['BETA_STD'].flatten()[0])
    
    for fig_num in range(nfigs):
        figs[fig_num].savefig('UKF_indep_tstep{:.1f}_{}.pdf'.format(simulation.tstep, fig_num), bbox_inches='tight')
        
    return figs

#%% Define main():
    
def main(time_spacings=np.logspace(2, 0, num=5).astype(np.int8)):
    set_config()

    simulations = [Simulation(keep_every) for keep_every in time_spacings]
    
    exceptions = run_simulations(simulations)
    
    print("\n\n\n{} exceptions occured in the call to KF.filter_update and UKF.filter:\n".format(len(exceptions)))
    for exc in exceptions:
        traceback.print_tb(exc[2]) # Traceback object
        print(repr(exc[0]) + ":    " + repr(exc[1])) # Error class and message
    
    figs = {}
    for simulation in simulations:
        figs[simulation] = {
                'neutron_population_plot': plot_neutron_population(simulation),
                'parameter_plots':         plot_parameters(simulation)
                }
    return exceptions, figs

#%% Run main():

if __name__ == "__main__":
    exceptions, figures = main(time_spacings=[100, 1])    
    for k, v in figures.items():
        print("\n" + "-"*15 + "  tstep: {:.2f}  ".format(k.tstep) + "-"*15)
        plt.show(v['neutron_population_plot'])
        for fig in v['parameter_plots']:
            plt.show(fig)
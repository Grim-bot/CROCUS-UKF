# -*- coding: utf-8 -*-
"""
Author: Felix Grimberg
Version: 2019-10-03

Version: 2019-05-10 (May 10th, 2019)
The Point Reactor Kinetics Equations are implemented here.
"""

import os, warnings
import numpy as np
if (__name__ == '__main__'):
    import util
    import config
else:
    import myModules.util as util
    import myModules.config as config
# import math
# import warnings
import scipy.integrate as sp_int
from scipy.linalg import expm

def check_config():
    print(config.config_dict())

class Precursor_group:
    
    def __init__(self, decay_constant, fractional_yield):
        self.decay_constant = decay_constant
        self.fractional_yield = fractional_yield

class Fuel:
    """
    Instances of this class store information about a nuclear fuel, such as the decay constants and fractional yields of a given number of precursor groups.
    The class also provides various methods for accessing this information.
    """
    
    def __init__(self, decay_constants, fractional_yields, mean_generation_time):
        """
        Class constructor for the Fuel class.
        
        INPUT:
            decay_constants:   iterable (e.g., list) of decay constants (unit: [1/s]).
            fractional_yields: iterable (e.g., list) of fractional yields (unitless).
            mean_generation_time: Mean neutron generation time (unit: [s]).
            
        """        
        
        self.mean_generation_time = mean_generation_time
        # Store information containing the precursor groups as a list of
        # Precursor_group objects:
        if len(decay_constants) != len(fractional_yields):
            error_msg = "Could not construct a Fuel object due to a size mismatch of the arguments 'decay_constants' and fractional_yields'.\n"
            error_msg += "decay_constants is of length " + repr(len(decay_constants)) + ", while fractional_yields is of length " + repr(len(fractional_yields)) + ".\n"
            error_msg += "decay_constants = " + repr(decay_constants) + "\n"
            error_msg += "fractional_yields = " + repr(fractional_yields)
            raise ValueError(error_msg)
        self.groups = [Precursor_group(l, b) for l, b in zip(decay_constants, fractional_yields)]
        
    def ngroups(self):
        """
        Returns the number of precursor groups stored with the fuel.
        """
        return len(self.groups)
        
    def decay_constants(self, group_nr=None):
        """
        Access the decay constants of the precursor groups.
        
        INPUT:
            group_nr (optional): The index or slice corresponding to the precursor group(-s) of interest. If it is not specified, then the decay constants of all of the precursor groups are returned.
        OUTPUT:
            The decay constant of the group_nr-th precursor group if group_nr is an integer, or otherwise a Numpy N-d array of precursor group decay constants.
        """
        
        if isinstance(group_nr, int):
            return self.groups[group_nr].decay_constant
        if group_nr is None:
            group_nr = slice(self.ngroups())
        return np.array([group.decay_constant for group in self.groups[group_nr]])
    
    def lambda_l(self, group_nr=None):
        """
        Shorthand for the decay_constants method.
        """
        return self.decay_constants(group_nr)

    def fractional_yields(self, group_nr=None):
        """
        Access the fractional yields of the precursor groups.
        
        INPUT:
            group_nr (optional): The index or slice corresponding to the precursor group(-s) of interest. If it is not specified, then the fractional yields of all of the precursor groups are returned.
        OUTPUT:
            The decay constant of the group_nr-th precursor group if group_nr is an integer, or otherwise a Numpy N-d array of the precursor groups' fractional yields.
        """
        if isinstance(group_nr, int):
            return self.groups[group_nr].fractional_yield
        if group_nr is None:
            group_nr = slice(self.ngroups())
        return np.array([group.fractional_yield for group in self.groups[group_nr]])
    
    def beta_l(self, group_nr=None):
        """
        Shorthand for the fractional_yields method.
        """
        return self.fractional_yields(group_nr)
    
    def delayed_neutron_fraction(self):
        """
        Compute the delayed neutron fraction (sum of fractional yields).
        """
        return self.fractional_yields().sum()
    def beta(self):
        """
        Shorthand for the delayed_neutron_fraction method.
        """
        return self.delayed_neutron_fraction()
    
    def param_names(self):
        names = []
        if config.include_reactivity:
            names.append('rho')
        for base_name in ('beta_', 'lambda_'):
            names.extend([base_name + str(i+1) for i in range(self.ngroups())])
        names.append('Lambda')
        return names
    
    def param_values(self, reactivity=None):
        vals = []
        if config.include_reactivity:
            vals.append(reactivity)
        vals.extend(self.beta_l())
        vals.extend(self.lambda_l())
        vals.append(self.mean_generation_time)
        return np.array(vals)
    
class Reactor_IO:
    """
    This base class defines functions for dealing with reactivity.
    """
            
    def unitless_to_dollars(self, rho):
        return rho / self.fuel.beta()
    
    def dollars_to_unitless(self, rho):
        return rho * self.fuel.beta()
        
    def reactivity_from_user_input(self, dollars, unitless):
        """
        Checks that one of the input arguments is None, and returns a unitless reactivity from the other.
        
        INPUT:
            dollars:  None or reactivity in dollar units.
            unitless: None or reactivity in absolute units.
            
        OUTPUT:
            reactivity: Reactivity in absolute units.
        """
        
        # Check how many input arguments were specified:
        util.assert_number_specified(locals(), expected_nums=1, levels_up=2)
        
        if dollars is None:
            return unitless
        else:
            return self.dollars_to_unitless(dollars)
        
    def initial_state_from_user_input(self, initial_state=None, initial_neutron_population=None, reactivity=None):
        """
        Checks that one of the input arguments is None, and returns an initial state from the other.
        If the initial neutron population is given, then the stationary solution to the Point Reactor Kinetics Equations is returned, to be used as an initial state.
        
        INPUT:
            initial_state:  None or a (n_groups + 1)-by-1 Numpy array containing the neutron population and the concentration of each precursor group.
            initial_neutron_population: None or the numerical value of the initial neutron population.
            
        OUTPUT:
            state: Either initial_state, or the stationary solution to the Point Reactor Kinetics Equations for the neutron population given by initial_neutron_population.
        """
            
        # Check how many input arguments were specified:
        util.assert_number_specified({'initial_state': initial_state, 'initial_neutron_population': initial_neutron_population}, expected_nums=1, levels_up=2)
        
        if initial_state is None:
            if not (self.include_params and config.include_reactivity):
                reactivity=None
            return self.stationary_PRKE_solution(initial_neutron_population, reactivity)
        else:
            if len(initial_state) != self.state_dims:
                raise ValueError("The shape of 'initial_state' argument is not consistent with the number of precursor groups of the fuel. The fuel has " + repr(self.fuel.ngroups()) + " precursor groups, so the state (neutron population and precursor concentrations) must be of length " + repr(self.state_dims) + ". The specified initial_state is of length " + repr(len(initial_state)) + ".")
            return initial_state
    

class PointReactor(Reactor_IO):
    """
    """
    
    def __init__(self, *args):
        """
        Class constructor for the Reactor class.
        
        INPUT:
            args: The arguments necessary for instantiation of the Fuel class. Check the docstring of the Fuel class for the latest information.
        """
        
        self.fuel = Fuel(*args)
        self.nvars = self.fuel.ngroups() + 1
        self.state_dims = self.nvars
        self.include_params = False
        
        self.nparams = None
        self.ind_rho = None
        self.slc_beta_l = None
        self.slc_lambda_l = None
        self.ind_Lambda = None
    
    def set_include_params(self, new_state):
        """
        Allows setting self.include_params to True or False.
        Adapts the state_dims attribute.
        
        INPUT:
            new_state: Bool to be used as the new state.
            
        No output.
        """
        if not isinstance(new_state, bool):
            raise RuntimeWarning("PointReactor.set_include_params takes only a boolean as input. The input was " + repr(new_state) + ". It will be converted to boolean explicitly.")
            print(type(new_state))
        
        self.include_params = bool(new_state)
        if self.include_params:
            self.state_dims = self.nvars + 1 + self.fuel.ngroups() * 2 + int(config.include_reactivity)
            self.nparams = self.state_dims - self.nvars
            if config.include_reactivity:
                self.ind_rho = self.nvars
            else:
                self.ind_rho = None
            ind_beta_l = self.nvars + int(config.include_reactivity)
            ind_lambda_l = ind_beta_l + self.fuel.ngroups()
            self.ind_Lambda = ind_lambda_l + self.fuel.ngroups()
            self.slc_beta_l = slice(ind_beta_l, ind_lambda_l)
            self.slc_lambda_l = slice(ind_lambda_l, self.ind_Lambda)
        else:
            self.state_dims = self.nvars
            self.ind_rho = None
            self.slc_beta_l = None
            self.slc_lambda_l = None
            self.ind_Lambda = None
        return
    
    def PRKE_matrix(self, reactivity=None, noise=None, state=None, verbose=False):
        """
        Constructs the matrix A s.t. dx/dt = A @ x + s, where:            
            x = [ neutron population,
                  concentration of precursors of first group,
                  .
                  .
                  .
                  concentration of precursors of last group]
            s = [ rate of neutrons added to the system by an external source,
                  0,
                  .
                  .
                  .
                  0]
        
        This corresponds to the Point Reactor Kinetics Equations:
            dn/dt   = (rho - beta) / LAMBDA * n(t) + sum(lambda_l * c_i(t))
            dc_i/dt = beta_l / LAMBDA * n(t) - lambda_l * c_i(t)
        Where:
            n(t)     : neutron population.
            c_i(t)   : concentration of precursors of i-th group.
            rho      : reactivity.
            beta_l   : fractional yield of precursors of i-th group.
            beta     : delayed neutron fraction. beta = sum(beta_l)
            LAMBDA   : mean neutron generation time.
            lambda_l : average decay constant of precursors of i-th group.
            
        INPUT:
            reactivity: The reactivity is assumed to be constant and imposed onto the reactor.
            noise (optional): If given, this 1-d array should contain noise terms which are  added, in order, to the parameters listed below. If self.include_params is True and len(noise) = self.state_dims, then the first self.nvars items are ignored.
                    parameters:
                the reactivity, rho
                each precursor group's fractional yield, beta_l
                each precursor group's decay constant, lambda_l
                the mean_generation time, Lambda
            state (optional): If self.include_params, then the parameters can also be inferred from a specified state (which includes the parameters).
            
        OUTPUT:
            A : N-d array of shape (ngroups + 1, ngroups + 1) s.t. dx/dt = A * x + s
        """
        if state is not None and not self.include_params:
            state = None
            raise RuntimeWarning("A state was specified in the call to PointReactor.PRKE_matrix despite self.include_params being False. The specified state will be ignored.")
        
        test_dict = {"reactivity":reactivity}
        if config.include_reactivity:
            test_dict['state'] = state
        util.assert_number_specified(test_dict)
        
        if state is None:
            beta_l = self.fuel.fractional_yields()
            lambda_l = self.fuel.decay_constants()
            Lambda = self.fuel.mean_generation_time
        else:
            if config.include_reactivity:
                reactivity = state[self.ind_rho]
            beta_l = state[self.slc_beta_l]
            lambda_l = state[self.slc_lambda_l]
            Lambda = state[self.ind_Lambda]
        
        if noise is not None:
            noise = np.array(noise).flatten()
            if self.include_params and (len(noise) == self.state_dims):
                # If include_params is True, then the noise may also contain terms for the variables, which are ignored by this function.
                noise = noise[self.nvars:]
            if len(noise) != self.nparams:
                raise ValueError("Size of 'noise' argument in call to PointReactor.PRKE_matrix() does not match the number of groups. See method docstring.")            
            
            if verbose:
                print("reactivity (before/after):\n", reactivity)
            
            if config.include_reactivity:
                reactivity += noise[self.ind_rho - self.nvars]
            beta_l += noise[util.shift_slice(self.slc_beta_l, - self.nvars)]
            lambda_l += noise[util.shift_slice(self.slc_lambda_l, - self.nvars)]
            Lambda += noise[self.ind_Lambda - self.nvars]
            
            if verbose:
                print(reactivity)
                print("beta_l (before/after):\n", self.fuel.beta_l())
                print(beta_l)
                print("lambda_l (before/after):\n", self.fuel.lambda_l())
                print(lambda_l)
                print("Lambda (before/after):\n", self.fuel.mean_generation_time)
                print(Lambda)

        beta = beta_l.sum()
        if verbose:
            print("beta (before/after):\n", self.fuel.beta())
            print(beta)
        
        first_column = np.insert(beta_l, 0, reactivity - beta).reshape((self.nvars, 1)) / Lambda
        other_columns = np.vstack((lambda_l, -1 * np.diag(lambda_l)))
        return np.hstack((first_column, other_columns))
        
    def stationary_PRKE_solution(self, neutron_population, reactivity=None):
        """
        Compute the stationary solution of the Point Reactor Kinetics Equations for a given neutron population (reactivity = 0 and constant neutron and precursor concentrations).
        
        INPUT:
            neutron_population: The stationary neutron population. This must be specified, because the PRK equations are linearly dependent in the case where reactivity = 0 and the neutron and precursor concentrations are constant.
            reactivity (optional): This input will be discarded unless self.include_params is True. If self.include_params is true, then the reactivity is part of the state and may therefore (optionally) be specified (unitless). Still, the computed state corresponds to the solution of the PRKE for a reactivity of 0.
            
        OUTPUT:
            x_stationary: Numpy 1-D array of length ngroups + 1 with          
            x = [ neutron population,
                  concentration of precursors of first group,
                  .
                  .
                  .
                  concentration of precursors of last group]
        """
        
        precursor_concentrations = (neutron_population / self.fuel.mean_generation_time) * self.fuel.fractional_yields() / self.fuel.decay_constants()
        state = np.insert(
                precursor_concentrations, 0, neutron_population
                )
        if self.include_params:
            if reactivity is None:
                reactivity = 0
            if config.include_reactivity:
                state = np.concatenate((
                        state, 
                        [reactivity],
                        self.fuel.beta_l(),
                        self.fuel.lambda_l(),
                        [self.fuel.mean_generation_time]
                        ))
            else:
                state = np.concatenate((
                        state, 
                        self.fuel.beta_l(),
                        self.fuel.lambda_l(),
                        [self.fuel.mean_generation_time]
                        ))
        elif reactivity is not None:
            raise RuntimeWarning("The stationary solution to the PRKE is precisely the solution for a reactivity of 0. Despite that, a reactivity was specified in the call to stationary_PRKE_solution for a reactor with include_params = False, where the reactivity is not included in the state.")
        return state
            
    def rod_drop(self, times, reactivity_dollars=None, reactivity_unitless=None, initial_state=None, initial_neutron_population=None, rtol=1e-5, solution=None, noise=None, store_time_steps=False):
        """
        Perform a simulation of a step reactivity insertion (positive or negative) 
        
        INPUT:
            times:                      Tuple made up of (start_time, end_time).
            reactivity_dollars:         The reactivity must be specified, either in dollars or unitless, unless it is included in initial_state.
            reactivity_unitless:        Alternative to reactivity_dollars.
            initial_state:              Needs not be specified, as it can be computed from the initial neutron population. The state of the reactor (neutron population and concentration of each precursor group) before the rod drop.
            initial_neutron_population: Do not specify this if initial_state is specified! Corresponds to the neutron population before the rod drop. If this argument is given, the reactor (i.e., the concentrations of the precursor groups) is assumed to be at equilibrium state before the rod drop.
            rtol:                       Relative tolerance for the solution.
            solution:                   Solution object in which to store the computed solution at each time step.
            noise (optional):           If given, this noise will be added to the reactor/fuel parameters. Cf. docstring of "PRKE_matrix" method for more detailed information.
            store_time_steps:           If set to True (default), then the solution will be stored for each time step.
        """
        
        start_time, end_time = times
        mat = None
        reactivity = None
        if initial_state is None or not self.include_params or not config.include_reactivity:
            reactivity = self.reactivity_from_user_input(reactivity_dollars, reactivity_unitless)
        if initial_state is None or not self.include_params:
            mat = self.PRKE_matrix(reactivity=reactivity, noise=noise)
            
        initial_state = self.initial_state_from_user_input(initial_state, initial_neutron_population, reactivity)
        
        if mat is None:
            if noise is not None:
                if len(noise) != self.state_dims:
                    raise ValueError('The dimensions of the noise specified to PRK.rod_drop() does not match self.state_dims!')
            mat = self.PRKE_matrix(state=initial_state, reactivity=reactivity)
        
        if config.integration is config.Integration.SOLVER:
            dstate_dt = lambda t, state: mat @ state
            solver = sp_int.ode(dstate_dt)
            solver.set_initial_value(initial_state[:self.nvars], t=start_time)
            solver.set_integrator('dopri5', rtol=rtol, nsteps=1e9) # 4-5 stages explicit RK method with adaptative time step.
            if store_time_steps:
                if solution is None:
                    solution = Solution(self)
                solver.set_solout(solution)
            
            end_state = solver.integrate(end_time)
        elif config.integration is config.Integration.ANALYTICAL:
            end_state = expm(mat * (end_time - start_time)) @ initial_state[:self.nvars]
        else:
            raise TypeError("Integration method {} not supported")

        if self.include_params:
            initial_state[:self.nvars] = end_state
            end_state = initial_state
        
        if (noise is not None) and (len(noise) == self.state_dims):
            end_state[:self.nvars] = end_state[:self.nvars] * (1 + noise[:self.nvars])
        
        if config.integration is config.Integration.SOLVER:
            if solver.successful():
                if store_time_steps:
                    return end_state, solution
                return end_state
            else:
                if solution is None:
                    return None
                return self.rod_drop((solution.times[-1], end_time), initial_state=end_state, solution=solution, store_time_steps=store_time_steps, reactivity_unitless=reactivity_unitless, reactivity_dollars=reactivity_dollars)
        else:
            return end_state

class Solution:
    """
    Class for storing the solution of the simulation at each time step.
    """
    
    def __init__(self, reactor):
        """
        Construct an empty Solution object. The reactor (a PointReactor Object) is used to set the shape of the state.
        """
        self.times = []
        self.states = []
    
    def __call__(self, time, state):
        """
        Store a time and state.
        """
        self.times.append(time)
        self.states.append(state.copy())
    
    def at(self, time):
        """
        Get the state at a given time. More accurately:
        Return the (time, state) pair closest to the specified time.
        """
        time, index = util.takeClosest(self.times, time)
        return time, self.states[index]
    
    def neutron_populations(self, times=None, time_interval=None, index=None):
        """
        Get the neutron population at any time or for any interval. The basic function call neutron_populations() returns a Numpy array containing the neutron population at all time steps.
        If times are specified, the nearest time for which a solution exists is chosen instead.
        
        INPUT:
            times: Float or sequence of floats specifying a certain time of the experiment.
            time_interval: Tuple of floats specifying the times between which the evolution of the neutron population is requested (time_interval[0] is included, while time_interval[1] is excluded).
            index: Index (int) or slice (slice object) to be used directly in self.states.
            
        OUTPUT:
            neutron_populations: 1-D numpy array containing the neutron population at the requested times.
        """
        
        util.assert_number_specified(locals(), expected_nums=[0,1])
        
        if times is not None:
            try:
                index = [util.takeClosest(self.times, t)[1] for t in times]
            except TypeError:
                index = util.takeClosest(self.times, times)[1]
        elif time_interval is not None:
            index_start = util.takeClosest(self.times, time_interval[0])[1]
            index_stop = util.takeClosest(self.times, time_interval[1])[1]
            index = slice(index_start, index_stop)
        elif index is None:
            index = slice(0, len(self.times))
        
        return np.array(self.states)[index,0]

if (__name__ == '__main__'):
    
    import matplotlib.pyplot as plt
    import plot # For the LaTeX-related parameters
    import copy
    
    
    def label_dollars(reactivity, dollars, sign=''):
        return r'$\rho = {2} {1} \cdot \beta$ {{\normalsize ($= {2} {0:.5f}$)}}'.format(reactivity, dollars, sign)
    def label_beta(ind):
        return r'$\beta_{0} = 1.2 \cdot \beta_{{{0},0}}$'.format(ind+1)
    
    # Store the properties of the CROCUS experimental reactor at EPFL:
    decay_constants = [0.0133535, 0.0326123, 0.121058, 0.305665, 0.861038, 2.89202]
    fractional_yields = [0.000238149, 0.00126101, 0.00122866, 0.00283823, 0.00126319, 0.000525235]
    # u235 = Fuel(decay_constants, fractional_yields, mean_generation_time=2.0e-5) #seconds
    crocus = PointReactor(decay_constants, fractional_yields, 4.68678e-05)
    
    end_time = 40 # seconds
    dollars = 0.15
    reactivity = crocus.fuel.beta() * dollars # unitless
    suptitle_str = r"Neutron population after reactivity insertion"
    
    fig, axes_list = plt.subplots(nrows=1, ncols=1, figsize=[plot.figure_width, plot.figure_width * 11/16], gridspec_kw={'top': 9/11})
    
    try:
        final_state, rod_drop = crocus.rod_drop(
                (0, end_time),
                reactivity_unitless=reactivity,
                initial_neutron_population=1
                )
    except TypeError as e:
        warnings.warn("Could not integrate the PRK equations (probably stiff).")
        print(e)
    else:
        plt.gca().plot(
                rod_drop.times,
                rod_drop.neutron_populations(),
                'm',
                label=label_dollars(reactivity, dollars, '+')
                )
        print(rod_drop.at(0.3))
    
    try:
        final_state, rod_drop = crocus.rod_drop(
                (0, end_time),
                reactivity_unitless=-reactivity,
                initial_neutron_population=1
                )
    except TypeError as e:
        warnings.warn("Could not integrate the PRK equations (probably stiff).")
        print(e)
    else:
        plt.gca().plot(
                rod_drop.times,
                rod_drop.neutron_populations(),
                'b--',
                label=label_dollars(reactivity, dollars, '-')
                )
    
    dollars = 1.001
    reactivity = crocus.fuel.beta() * dollars # unitless
    
    try:
        final_state, rod_drop = crocus.rod_drop(
                (0, end_time),
                reactivity_unitless=-reactivity,
                initial_neutron_population=1
                )
    except TypeError as e:
        warnings.warn("Could not integrate the PRK equations (probably stiff).")
        print(e)
    else:
        plt.gca().plot(
                rod_drop.times,
                rod_drop.neutron_populations(),
                'b--',
                label=label_dollars(reactivity, dollars, '-')
                )
    
    print(rod_drop.states[-1][0])
    plt.gca().axhline(y=1, xmin=0, xmax=end_time, c='k')
    plt.gca().axhline(y=0, xmin=0, xmax=end_time, c='k')
    fig.suptitle(suptitle_str)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\frac{n(t)}{n(t=0)}$")
    plt.gca().legend(bbox_to_anchor=(0,1, 1, 0), loc='lower left', mode='expand', ncol=2, frameon=True, framealpha=1, edgecolor='k')#, labelspacing=0.1)
    
    plt.savefig(fname=os.path.join('figures','PRK_transients.pdf'), bbox_inches='tight')
    plt.show()
    
    fig, axes_list = plt.subplots(nrows=1, ncols=1, figsize=[plot.figure_width, plot.figure_width * 10.5/16], gridspec_kw={'top': 9/10.5})
    
    final_state, rod_drop = crocus.rod_drop((0, end_time), reactivity_unitless=reactivity, initial_neutron_population=1)
    plt.gca().semilogy(rod_drop.times, rod_drop.neutron_populations(), 'r', label=label_dollars(reactivity, dollars))
    
    plt.gca().axhline(y=1, xmin=0, xmax=end_time, c='k')
    fig.suptitle(suptitle_str)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\frac{n(t)}{n(t=0)}$")
    plt.gca().legend(bbox_to_anchor=(0,1, 1, 0), loc='lower left', mode='expand', ncol=1, frameon=True, framealpha=1, edgecolor='k')#, labelspacing=0.1)
    
    plt.savefig(fname=os.path.join('figures','PRK_transients_b.pdf'), bbox_inches='tight')
    plt.show()
    
    
    end_time = 250 # seconds
    reactivity = 0.00112 # unitless
    
    stylz = (':', '--', '-.')
    colorz = ('g', 'r', 'm', 'y')
    markercolorz = ('g', 'r', 'm', 'y', 'b', 'c')
#    colorz = markercolorz
    markerz = ('v', '^', 'X', 'o', 'P', '*')
    other_markerz = ('1', '2', '3', '4', '+', 'x')
    markeveryz = (1500, 1700, 2000, 2200, 2500, 2800)
    
#    print("solution at t={0}s:\n{1}".format(end_time, final_state))
#    print("times:\n" + repr(rod_drop_1.times))
#    t, y = rod_drop_1.at(0.1)
#    print("solution at t=" + repr(t) + "s:\n" + repr(y))
    
    #######################################
    fig, axes_list = plt.subplots(nrows=1, ncols=1, figsize=[plot.figure_width, plot.figure_width * 11.5/16], gridspec_kw={'top': 9/11.5})
    
    label_first_line = r'$\rho = \rho_0$, $\beta_1 = \beta_{{1,0}}$, $\cdots$, $\beta_{0} = \beta_{{{0},0}}$'.format(crocus.fuel.ngroups())
    final_state, rod_drop = crocus.rod_drop((0, end_time), reactivity_unitless=reactivity, initial_neutron_population=1)
    plt.gca().plot(rod_drop.times, rod_drop.neutron_populations(), 'k', label=label_first_line)
    
    reactivity_2 = reactivity / 1.2
    final_state, rod_drop = crocus.rod_drop((0, end_time), reactivity_unitless=reactivity_2, initial_neutron_population=1)
    plt.plot(rod_drop.times, rod_drop.neutron_populations(), 'c', label=r'$\rho = 0.833 \cdot \rho_0$')
    
    final_state, rod_drop = crocus.rod_drop((0, end_time), reactivity_unitless=-reactivity, initial_neutron_population=1)
    plt.gca().plot(rod_drop.times, rod_drop.neutron_populations(), 'b', label=r'$\rho = - \rho_0$')
    
    for ind in range(crocus.fuel.ngroups()):
        crocus2 = copy.deepcopy(crocus)
        crocus2.fuel.groups[ind].fractional_yield = crocus.fuel.groups[ind].fractional_yield * 1.2
        final_state, rod_drop = crocus2.rod_drop((0, end_time), reactivity_unitless=reactivity, initial_neutron_population=1)
        plt.gca().plot(rod_drop.times, rod_drop.neutron_populations(), label=label_beta(ind), linestyle=stylz[ind%len(stylz)], color=colorz[ind%len(colorz)], marker=markerz[ind], markersize=12, markevery=markeveryz[ind%len(markeveryz)], markeredgecolor=markercolorz[ind%len(markercolorz)], markerfacecolor=markercolorz[ind%len(markercolorz)])    #, markeredgewidth=2
    
    fig.suptitle(suptitle_str)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\frac{n(t)}{n(t=0)}$")
    plt.gca().legend(bbox_to_anchor=(0,1, 1, 0), loc='lower left', mode='expand', ncol=3, frameon=True, framealpha=1, edgecolor='k')#, labelspacing=0.1)
    
    plt.savefig(fname=os.path.join('figures','PRK_neutron_population.pdf'), bbox_inches='tight')
    plt.show()
    
    
    ###############################
    reac_difference = 0.0002
    fig, axes_list = plt.subplots(nrows=1, ncols=1, figsize=[plot.figure_width, plot.figure_width * 12.5/16], gridspec_kw={'top': 9/12.5})
    
    final_state, rod_drop = crocus.rod_drop((0, end_time), reactivity_unitless=reactivity, initial_neutron_population=1)
    plt.gca().plot(rod_drop.times, rod_drop.neutron_populations(), 'k', label=r'$\rho = {0}$'.format(reactivity))
    
    reactivity_2 = reactivity - reac_difference
    final_state, rod_drop = crocus.rod_drop((0, end_time), reactivity_unitless=reactivity_2, initial_neutron_population=1)
    plt.plot(rod_drop.times, rod_drop.neutron_populations(), 'c', label=r'$\rho = {0:.6f}$ ($-{1}$)'.format(reactivity_2, reac_difference))
    
    final_state, rod_drop = crocus.rod_drop((0, end_time), reactivity_unitless=-reactivity, initial_neutron_population=1)
    plt.gca().plot(rod_drop.times, rod_drop.neutron_populations(), 'b', label=r'$\rho = -{0}$'.format(reactivity))
    
    for ind in range(crocus.fuel.ngroups()):
        crocus2 = copy.deepcopy(crocus)
        crocus2.fuel.groups[ind].fractional_yield = crocus.fuel.groups[ind].fractional_yield + reac_difference
        final_state, rod_drop = crocus2.rod_drop((0, end_time), reactivity_unitless=reactivity, initial_neutron_population=1)
        plt.gca().plot(rod_drop.times, rod_drop.neutron_populations(), label=r'$\rho = {0}$ ($\beta_{1} + {2}$)'.format(reactivity, ind+1, reac_difference), linestyle=stylz[ind%len(stylz)], color=colorz[ind%len(colorz)], marker=markerz[ind], markersize=12, markevery=markeveryz[ind%len(markeveryz)], markeredgecolor=markercolorz[ind%len(markercolorz)], markerfacecolor=markercolorz[ind%len(markercolorz)])    #, markeredgewidth=2
    
    fig.suptitle(suptitle_str)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\frac{n(t)}{n(t=0)}$")
    plt.gca().legend(bbox_to_anchor=(0,1, 1, 0), loc='lower left', mode='expand', ncol=2, frameon=True, framealpha=1, edgecolor='k')#, labelspacing=0.1)
    
    plt.savefig(fname=os.path.join('figures','PRK_neutron_population_b.pdf'), bbox_inches='tight')
    plt.show()
    
    
    #######################################
    fig, axes_list = plt.subplots(nrows=1, ncols=1, figsize=[plot.figure_width, plot.figure_width * 11/16], gridspec_kw={'top': 9/11})
    
    final_state, rod_drop = crocus.rod_drop((0, end_time), reactivity_unitless=reactivity, initial_neutron_population=1)
    plt.gca().plot(rod_drop.times, rod_drop.neutron_populations(), 'k', label=label_first_line)
    
    reactivity_2 = reactivity / 1.2
    final_state, rod_drop = crocus.rod_drop((0, end_time), reactivity_unitless=reactivity_2, initial_neutron_population=1)
    plt.plot(rod_drop.times, rod_drop.neutron_populations(), 'c', label=r'$\rho = 0.833 \cdot \rho_0$')
    

    crocus2 = copy.deepcopy(crocus)
    ind = 1
    crocus2.fuel.groups[ind].fractional_yield = crocus.fuel.groups[ind].fractional_yield * 1.2
    final_state, rod_drop = crocus2.rod_drop((0, end_time), reactivity_unitless=reactivity, initial_neutron_population=1)
    plt.gca().plot(rod_drop.times, rod_drop.neutron_populations(), label=label_beta(ind), linestyle=stylz[ind%len(stylz)], color=colorz[ind%len(colorz)], marker=markerz[ind], markersize=12, markevery=markeveryz[ind%len(markeveryz)], markeredgecolor=markercolorz[ind%len(markercolorz)], markerfacecolor=markercolorz[ind%len(markercolorz)])    #, markeredgewidth=2
    
    crocus2 = copy.deepcopy(crocus)
    crocus2.fuel.groups[ind].fractional_yield = crocus.fuel.groups[ind].fractional_yield / 1.2
    final_state, rod_drop = crocus2.rod_drop((0, end_time), reactivity_unitless=reactivity_2, initial_neutron_population=1)
    label_str = label_beta(ind)
    # r'$\rho = \frac{\rho_0}{1.2}$'
    ind += 1
    plt.gca().plot(rod_drop.times, rod_drop.neutron_populations(), label=label_str, linestyle=stylz[ind%len(stylz)], color=colorz[ind%len(colorz)], marker=markerz[ind], markersize=12, markevery=markeveryz[ind%len(markeveryz)], markeredgecolor=markercolorz[ind%len(markercolorz)], markerfacecolor=markercolorz[ind%len(markercolorz)])    #, markeredgewidth=2
    
    fig.suptitle(suptitle_str)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\frac{n(t)}{n(t=0)}$")
    plt.gca().legend(bbox_to_anchor=(0,1, 1, 0), loc='lower left', mode='expand', ncol=2, frameon=True, framealpha=1, edgecolor='k')#, labelspacing=0.1)
    
    plt.savefig(fname=os.path.join('figures','PRK_neutron_population_c.pdf'), bbox_inches='tight')
    plt.show()
    
    
#    print("initial neutron population:\n" + repr(rod_drop_1.neutron_populations(times=0)))
#    print("last three neutron populations:\n" + repr(rod_drop_1.neutron_populations(index=slice(-4,-1))))

#    for count in range(2):
#        crocus.PRKE_matrix(reactivity=0.0005, noise=np.full((15,), 10)) #, verbose=True)
    
#    crocus.set_include_params(True)
#    print(crocus.param_names(), crocus.param_values(0.001))
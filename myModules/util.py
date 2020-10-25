# -*- coding: utf-8 -*-
"""
Author: Felix Grimberg
Version: 2019-10-03

Auxiliary functions for more or less mathematical tasks.
"""

import numpy as np
import scipy.signal as signal
import inspect, math
from bisect import bisect_left


def remove_excess_axes(l_in, len_desired):
    if len(l_in) < len_desired:
        raise ValueError('List is shorter than desired length!')
    while len(l_in) > len_desired:
        l_in[-1].remove()
        l_in = l_in[:-1]
    return l_in

def shift_slice(slc, shift):
    start = slc.start
    if start is None:
        start = 0
    stop = slc.stop
    step = slc.step
    return slice(start+shift, stop+shift, step)


def extract_observations(fname, tstep_observation, positive_reactivity=True):
    """
    Estimates the index at which the rod drop took place. In a first step, the observations are smoothed. The time of rod drop is estimated conservatively (i.e., minimizing the probability that the calculated time of rod drop lies before the actual rod drop): It is taken as the time when the smoothed signal first surpasses its own maximum of the first 50 observations (or the first 5s, if the time step is >= 0.1).
    
    INPUTS:
        observations: A portion of the recorded neutron counts, which 
    """
    # Read neutron counts from file:
    observations = np.loadtxt(fname)
    # Stop at maximum (the control rods are inserted back after reaching the maximum):
    if positive_reactivity:
        observations = observations[:np.argmax(observations)]
        # Remove the beginning of the measurement because 0 counts are a real problem:
        observations = observations[(observations <= 0).nonzero()[0][-1]+1:]
    else:
        last_min = len(observations) - np.argmin(observations[::-1])
        observations = observations[:last_min]
    
    # Smooth the first 1000 observations:
    b,a = signal.butter(5, 0.05)
    observations_smoothed = signal.filtfilt(b,a, observations[:1000])
    if positive_reactivity:
        max_resting = max(observations_smoothed[:min(50, math.ceil(5/tstep_observation))])
        ind_start = np.asarray(observations_smoothed >= 2*max_resting).nonzero()[0][0]
    else:
        print('WARNING: Time of rod drop is estimated very badly for this case!')
        ind_start = np.argmax(observations_smoothed)
    print(ind_start)
    return observations[ind_start:]

def fix_SPD(mat):
    """
    Performs eigenvalue decomposition of the matrix mat and replaces all negative eigenvalues with a small positive number before returning the rebuilt matrix.
    This ensures that the returned matrix is SPD.
    
    INPUTS:
        mat: Matrix to be fixed.
        
    OUTPUTS:
        mat: Fixed matrix.
    """
   
    vals, vecs_right = np.linalg.eig(mat)
    
    replacement = 1e-18
    
    minimal, total= (np.min(vals), np.sum(vals <= 0))
    
    vals = np.maximum(vals, replacement)
    
    mat2 = vecs_right @ np.diag(vals) @ vecs_right.T
    
    return mat2, minimal, total

def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber, and corresponding index.

    If two numbers are equally close, return the smaller of the two.
    
    INPUT:
        myList: sequence of sorted numbers.
        myNumber: sought-for value.
        
    OUTPUT:
        value: Value in myList that is nearest to myNumber.
        index: Index of value in myList.        
    
    Copied from https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0], pos
    if pos == len(myList):
        return myList[-1], -1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after, pos
    else:
       return before, pos - 1

def only_x1(inputs):
    """only_x1 returns a Numpy 1D-array containing the "top left" value of each ND-array in inputs."""
    if np.array(inputs).ndim == 2:
        return np.array(inputs)[:,0]
    else:
        return np.array(inputs)[:,0,0]

def cov2corr(COV):
    """Given a covariance matrix COV, returns the corresponding correlation matrix CORR."""
    if (COV==0).all():
        return COV
    temp = np.diag(np.diag(COV)**-0.5)
    return temp @ COV @ temp

def assert_number_specified(args_dict, expected_nums=[1], levels_up=1, keep_self=False):
    """
    Typical function call: assert_number_specified(locals())
    Checks how many of the values in args_dict are None. If args_dict contains an item with the key 'self', this item is excluded from this check unless keep_self is True.
    If the number of values that are not None is different from 1, then a ValueError is raised. The error message informs the user of the number of expected arguments, the number of specified arguments, the names of the possible arguments, and the name of the function which called assert_number_specified.
    
    INPUT:
        args_dict: This dictionary contains the names and values of the arguments to be checked. It should typically be the locals() dict of the calling function.
        expected_nums (optional): This sequence of ints or single int corresponds to the number of arguments that are expected to be different from None.
        levels_up (optional): The name of the function which is levels_up levels above assert_number_specified in the call hierarchy will be printed out with the error message. This should, ideally, be the function called directly by the user.
        keep_self (optional): If args_dict contains an item with the key 'self', this item is excluded from any considerations unless keep_self is True.
        
    OUTPUT:
        None. Will raise a ValueError if the number of specified arguments does not correspond to the expected number.
    """
    
    # Place expected_nums in a list if it is a single value:
    try:
        iter(expected_nums)
    except TypeError:
        expected_nums = [expected_nums]
            
    
    if not keep_self:
        args_dict.pop("self", None)
    num_specified = np.sum([v is not None for v in args_dict.values()])
    if all([num_specified != num for num in expected_nums]):
        caller = inspect.stack()[levels_up].function
        arg_names = "'" + "' and '".join(args_dict.keys()) + "'"
        raise ValueError("Specify exactly " + repr(expected_nums) + " of " + arg_names + "! " + repr(num_specified) + " of them were specified during call to " + caller + ".")

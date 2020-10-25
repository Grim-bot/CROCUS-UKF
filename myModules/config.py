"""
Author: Felix Grimberg
Version: 2019-10-03
"""
import enum

class Integration(enum.Enum):
    SOLVER = enum.auto()
    ANALYTICAL = enum.auto()
    
if not 'integration' in globals():
    integration = Integration.SOLVER

if not 'include_reactivity' in globals():
    include_reactivity = True

if not 'usemask' in globals():
    usemask = True

if not 'use_EKF' in globals():
    use_EKF = True

if not 'use_UKF' in globals():
    use_UKF = True
    
if not 'stdev_initial_factor' in globals():
    stdev_initial_factor = 0.5

if not 'stdev_transition_dep' in globals():
    stdev_transition_dep = 1e-3


def config_dict():
    return {'include_reactivity': include_reactivity,
            'usemask': usemask,
            'use_EKF': use_EKF,
            'use_UKF': use_UKF,
            'stdev_initial_factor': stdev_initial_factor,
            'stdev_transition_dep': stdev_transition_dep,
            'integration': integration
            }
    

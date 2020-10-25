# -*- coding: utf-8 -*-
"""The introductory problem of a moving car is implemented here, using the pykalman module. Plotting is done using matplotlib, and of course the numpy package is used.
"""
import math, os
import numpy as np
import matplotlib.pyplot as plt
import pykalman
import myModules.plot as myplot
import myModules.util as ut


###############################################################################
# %%  Parameters to specify:  ###########################

dt = 1          # time step                                         [s]
num_steps = 13  # number of time steps to compute (including t=0)
stdev_acc = 0.3         # standard deviation of the acceleration    [m/s^2]
stdev_observation = 2   # standard deviation of the observations    [m]

###############################################################################
# %%  Problem specification:  ###########################

# Values for a Swedish SJ X2 train:
mass = 73 + 5 * 47 # metric tons, 1 locomotive and 5 coaches
max_force = 160 # kN --> max_acceleration = 0.5 m/s^2

angle = 0.25*math.pi #np.random.uniform(0,2*math.pi)
direction = np.array([math.sin(angle),
                         math.cos(angle)])
    
# Initialize position, velocity:
initial_state = np.zeros(4)
# State transition matrix: (state vector: transpose(x1, x2, v1, v2))
transition_matrix = np.eye(4) + dt * np.eye(4,k=2)
# Control input matrix: (control vector: transpose(a1, a2))
control_matrix = np.vstack((0.5 * dt**2 * np.eye(2), dt * np.eye(2)))
# Process noise covariance:
q = np.hstack((0.5 * dt**2 * direction, dt * direction))
COV_transition = stdev_acc**2 * np.outer(q,q)
# So far, COV_transition was the covariance of a process noise entirely caused by a perturbed acceleration. In the next line, the variances are slightly modified arbitrarily and the covariances are reduced, allowing more irregular variations.
COV_transition = COV_transition * [[1.15, 0.7, 0.75, 0.73],
         [0.7, 1.1, 0.73, 0.76],
         [0.75, 0.73, 1, 0.95],
         [0.73, 0.76, 0.95, 1.015]]

# Observation matrix:
observation_matrix = np.eye(3, M=4)
observation_matrix[2,2:] = 0.5 * np.ones((1,2))
# Observation noise covariance:
COV_observation = np.eye(3)
#COV_observation[2,2] = 1.5
COV_observation = COV_observation * stdev_observation**2
# State transition offset: The pykalman package combines the control matrix and the control input into the state transition offset, which is a vector of same dimension as the state. It is simply the product of the control input matrix and the control input, resulting in the equation:
# x_(t+1) = transition_matrix @ x_(t) + transition_offset + noise
# In the case of the train, the control input would be a certain acceleration/decelration profile, for which we pick a sine curve. We are assuming straight tracks.
transition_offsets = np.array([control_matrix @ (direction * max_force / mass * factor) for factor in np.sin(np.linspace(0, 2*math.pi, num_steps, endpoint = False)[1:])])

# Save the problem description in a KalmanFilter object:
kf = pykalman.KalmanFilter(
        transition_matrices=transition_matrix, observation_matrices=observation_matrix, transition_covariance=COV_transition, observation_covariance=COV_observation, initial_state_mean=initial_state,
        observation_offsets=np.zeros(3), initial_state_covariance=np.zeros((4,4)),
        transition_offsets=transition_offsets
        )

###############################################################################
# %%  COMPUTATION:  #####################################

# Generate a sequence of true (hidden) states and observations
states_true, observations = kf.sample(num_steps)
# Estimate the state from the observations
states_filtered, COVs_filtered = kf.filter(observations)

# Compute the result of the prediction step at each time step, together with the covariance matrix of this estimate:
states_pred_2_to_n = np.einsum('ij,kj->ki',transition_matrix, states_filtered[:-1]) + transition_offsets
states_pred = np.concatenate((np.zeros((1,4)), states_pred_2_to_n))
COVs_pred = np.concatenate(([np.zeros((4,4))], [transition_matrix @ COV_f @ transition_matrix.T + COV_transition for COV_f in COVs_filtered[:-1]]))

# Compute a set of purely model-based estimates (for comparison):
states_model = [initial_state]
COVs_model = [np.zeros((4,4))]
for transition_offset in transition_offsets:
    states_model.append(transition_matrix @ states_model[-1] + transition_offset)
    COVs_model.append(transition_matrix @ COVs_model[-1] @ transition_matrix.T + COV_transition)


###############################################################################
# %%  Visualization of the results:  ####################

time = np.linspace(0.,dt*num_steps, num=num_steps)

# Plot the true distance and its various predictions:
# TRUE VALUE
x1_true = ut.only_x1(states_true)
fmt_true = "k-"
# MODEL
x1_model = ut.only_x1(states_model)
stdev_model = np.sqrt(ut.only_x1(COVs_model))
fmt_model = "c--"
# KALMAN - prediction step
x1_pred = ut.only_x1(states_pred)
stdev_pred = ut.only_x1(COVs_pred)
fmt_pred = "bv"
# KALMAN - update step
x1_filtered = ut.only_x1(states_filtered)
stdev_filtered = ut.only_x1(COVs_filtered)
fmt_filtered = "g-."
# OBSERVATION
x1_observation = ut.only_x1(observations)
color_observation = "m"
# We want to plot the evolution of the Kalman-filtered prediction through a sequence of prediction and update steps. To do that, we must first compute the error of the predictions made before the update step. Then, we also need a ND-array of times where each time is contained twice in sequence.
# Construct a ND-array which contains each time twice:
two_step_time = np.empty((num_steps)*2)
two_step_time[0::2] = time
two_step_time[1::2] = time
# Construct the series of predictions (model prediction - update - model prediction - etc)
x1_two_step = np.empty((num_steps)*2)
x1_two_step[0::2] = x1_pred
x1_two_step[1::2] = x1_filtered
stdev_two_step = np.empty((num_steps)*2)
stdev_two_step[0::2] = stdev_pred
stdev_two_step[1::2] = stdev_filtered
fmt_two_step = "b--"



# Create a new figure
fig, axes_list = plt.subplots(nrows=3, figsize = [myplot.figure_width, myplot.figure_width*1.4], gridspec_kw={'hspace':0.35})
ax = axes_list[0]

# PLOT VALUES
ax.plot(time, x1_true, fmt_true)
# Plot the various estimates with errorbars.
ax.plot(time, x1_observation, color_observation)
ax.plot(time, x1_model, fmt_model[0])
ax.plot(time, x1_pred, fmt_pred)
ax.plot(time, x1_filtered, fmt_filtered)
ax.plot(two_step_time, x1_two_step, fmt_two_step)
#ax.plot(time[1:], x1_observation[1:], color_observation+",")

# Labels, legend, annotation
ax.set(xlabel="Time [s]", ylabel=r"$x_1$ [m]", title=r"$x_1(t)$ vs. its various estimates")
myplot.legend(ax, [r'$x_1(t)$', r'$z_1^{(i)}$', r'$\hat x_1^{(i)}$', r'$\hat x_1^{(i|i-1)}$', r'$\hat x_1^{(i|i)}$'], ncol=1)
myplot.annotate(ax, dt, stdev_acc, stdev_observation, loc_x=0.22, loc_y=0.64) 

    
# Plot the errors of the predictions
# COMPUTE ERRORS
error_observation = abs(x1_observation - x1_true)
error_model = abs(x1_model - x1_true)
error_filtered = abs(x1_filtered - x1_true)
error_pred = abs(x1_pred - x1_true)
error_two_step = np.empty((num_steps)*2)
error_two_step[0::2] = error_pred
error_two_step[1::2] = error_filtered


# Plot standard deviations
ax = axes_list[1]
ax.axhline(y=stdev_observation, color=color_observation, label=r'$\sigma_{obs}$')
ax.plot(time, stdev_model, fmt_model, label=r'$\sigma^{(i)}$')
ax.plot(time, stdev_pred, fmt_pred, label=r'$\sigma^{(i|i-1)}$')
ax.plot(time, stdev_filtered, fmt_filtered, label=r'$\sigma^{(i|i)}$')
ax.plot(two_step_time, stdev_two_step, fmt_two_step)
ax.set(xlabel="Time [s]", ylabel=r"Standard deviation [m]", title=r"Uncertainty of the estimates of $x_1(t)$")
myplot.legend(ax, ncol=2)



# PLOT
ax = axes_list[2]
ax.plot(
        time, error_observation, color_observation,
        time, error_model, fmt_model,
        time, error_pred, fmt_pred,
        time, error_filtered, fmt_filtered,
        two_step_time, error_two_step, fmt_two_step
        )
#ax.grid()

# Labels, legend, annotation
ax.set(xlabel="Time [s]", ylabel="Error [m]", title=r"Absolute error of the estimates of $x_1(t)$")
myplot.legend(ax, myplot.errorize([r'$z_1^{(i)}$', r'$\hat x_1^{(i)}$', r'$\hat x_1^{(i|i-1)}$', r'$\hat x_1^{(i|i)}$']), ncol=2)
plt.savefig(os.path.join('figures','distance_plot.pdf'), bbox_inches='tight')


###############################################################################
# %% PLOT THE COVARIANCE MATRICES ################

fig_COVs = myplot.plot_matrices(COVs_filtered[1:], dt, non_neg=False, title=r'Visualization of $\mathbf{\Sigma}_{\hat \mathbf{x} \hat \mathbf{x}}^{(i|i)}$ for $i = 1,\dots,n-1$', include_zero=False)
fig_COVs.savefig(os.path.join('figures','Covariance.pdf'), bbox_inches='tight')

CORRs_filtered = [ut.cov2corr(COV) for COV in COVs_filtered]
fig_CORRs = myplot.plot_matrices(CORRs_filtered[1:], dt, non_neg=False, title='Matrix of correlation coefficients of $\mathbf{\hat x}^{(i|i)}$ for $i = 1,\dots,n-1$', include_zero=False)
fig_CORRs.savefig(os.path.join('figures','Correlation_coeffs.pdf'), bbox_inches='tight')
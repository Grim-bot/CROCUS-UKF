# -*- coding: utf-8 -*-
"""
Author: Felix Grimberg
Version: 2019-10-03

Auxiliary functions used for plotting with matplotlib and latex.
"""

import math
from distutils.spawn import find_executable
import numpy as np
import enum
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
if (__name__ == '__main__'):
    import config
else:
    import myModules.config as config
    
if not find_executable('latex'):
    raise ImportError("A working Latex installation is required (unrelated to Python installation)!")

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif', size=14, weight='bold')
matplotlib.rc('errorbar', capsize=8)
matplotlib.rc('axes.formatter', limits=[0,20], useoffset=False)

figure_width = 9.7  # Approximate width in inches that a figure should have on an A4 page according to me (not set in stone!).

bbox_propdict = {'edgecolor':'k', 'facecolor':'w', 'boxstyle':'round,pad=1.0,rounding_size=0.2'}


plot_kwargs = {'markersize':8, 'mfc':'w', 'mew':2}

fmt_str = {'UKF': 'mv-',
           'EKF': 'co-',
           'PRK': 'y--',
           'observation': 'k'}

@enum.unique
class MethodType(enum.Enum):
    UKF = r'$n_{UKF}(t)$'
    EKF = r'$n_{EKF}(t)$'
    PRK = r'$n_{PRK}(t)$'
    observation = r'$z^{(i)}$'
    
    def get_fmt(self):
        return fmt_str[self.name]
    def get_color(self):
        return self.get_fmt()[0]

class NeutronPopulationPlot:
    n_sigmas_shade = 10
    def __init__(self, neutron_pop_results):
        self.times = neutron_pop_results.times
        self.observations = neutron_pop_results.observations
        self.method_lines = [
                self.MethodLine(method_result) for method_result
                in neutron_pop_results.method_results
                ]
        
    class MethodLine:
        def __init__(self, method_result):
            self.method_type = method_result.method_type
            self.means = method_result.means
            self.variances = method_result.variances
            
        def linewidth(self):
            return 2 if ((self.method_type.name in ['EKF', 'UKF'])) else 0.5
    
        def label(self):
            return self.method_type.value
        
        def fmt(self):
            return self.method_type.get_fmt()
        
        def color(self):
            return self.method_type.get_color()
        
        def std_label(self):
            return Label_UKF.make('sigma', self.method_type.name)
        
        def SE_label(self):
            return squared_errorize_RKKF(
                    'n_{' + self.method_type.name + '}(t^{(i)})'
                    )
    
    def plot_n_shaded(self):
        """Create a plot featuring the estimates of the neutron population
        over time, where the uncertainty of each estimate is visualized as
        a shaded area around the plot line.
        """
        fig = plt.figure(figsize=[figure_width, figure_width])
        ax = plt.gca()
        plt.yscale('log')
        
        num_markers=4
        for method_line in self.method_lines:
            lower = np.maximum(
                    1,
                    method_line.means - self.n_sigmas_shade / 2 * np.sqrt(method_line.variances)
                    )
            upper = method_line.means + self.n_sigmas_shade / 2 * np.sqrt(method_line.variances)
            ax.fill_between(
                    self.times, upper, lower,
                    facecolor=method_line.color(), alpha=0.2
                    )
            ax.semilogy(
                    self.times, method_line.means,
                    method_line.fmt(),
                    label=method_line.label(),
                    lw=method_line.linewidth(),
                    markevery=len(self.times) // num_markers,
                    **plot_kwargs
                    )
            num_markers += 3
            
        # Add legend, labels, annotations, title; show and save fig
        ax.set(xlabel="Time [s]", ylabel=r'Neutron count [1 per $\Delta t$]', title=r"Neutron count vs. estimates of the neutron population")
        legend(ax)
        tstep = self.times[1] - self.times[0]
        annotate_RKKF(ax, config.stdev_initial_factor, config.stdev_transition_dep, tstep=tstep)
        
        fig.savefig('neutron_population_tstep{:.1f}.pdf'.format(tstep), bbox_inches='tight')
        return fig
    
    def plot_n_sd_erel(self):
        """Create a figure with 3 subplots containing respectively:
            -The estimates of the neutron population over time,
            -The estimated uncertainty of the neutron population estimates, and
            -The relative difference between the estimate of the neutron
            population and the measured neutron count."""
        # Create a new figure
        fig, axes_list = plt.subplots(
                nrows=3,
                figsize=[figure_width, figure_width*1.4],
                gridspec_kw={'hspace':0.35}
                )
        ax = axes_list[0]
        ax_std = axes_list[1]
        ax_SE = axes_list[2]
        
        # Choose between semilog-plot and regular plot
        if np.max(self.observations) < 350:
            neutron_plot = ax.plot
            std_plot = ax_std.plot
            ax_std.set_ylim(bottom=0, top=np.sqrt(np.max(self.observations)) * 1.35)
            SE_plot = ax_SE.plot
        else:
            neutron_plot = ax.semilogy
            std_plot = ax_std.semilogy
            SE_plot = ax_SE.plot
        
        # Plot lines for each method result:
        num_markers = 4
        for method_line in self.method_lines:
            neutron_plot(
                    self.times, method_line.means,
                    method_line.fmt(),
                    label=method_line.label(),
                    lw=method_line.linewidth(),
                    markevery=len(self.times) // num_markers,
                    **plot_kwargs
                    )
            std_plot(
                    self.times, np.sqrt(method_line.variances),
                    method_line.fmt(),
                    label=method_line.std_label(),
                    lw=method_line.linewidth(),
                    markevery=len(self.times) // num_markers,
                    **plot_kwargs
                    )
            if method_line.method_type is not MethodType.observation:
                SE_plot(
                        self.times,
                        np.absolute(
                                method_line.means - self.observations
                                ) / self.observations,
                        method_line.fmt(),
                        label=method_line.SE_label(),
                        lw=0.2,
                        markevery=len(self.times) // num_markers,
                        **plot_kwargs
                        )
            num_markers += 3
        
        # Add legend, labels, annotations, title; show and save fig
        ax.set(xlabel="Time [s]", ylabel=r'Neutron count [1 per $\Delta t$]', title=r"Neutron count vs. its various estimates")
        legend(ax)
        annotate_RKKF(ax, config.stdev_initial_factor, config.stdev_transition_dep)
        ax_std.set(xlabel="Time [s]", ylabel=r'Standard deviation [1 per $\Delta t$]', title=r"Estimated uncertainty of the estimates of the neutron count")
        legend(ax_std)
        annotate_RKKF(ax_std, config.stdev_initial_factor, config.stdev_transition_dep, loc_x=0.27)
        ax_SE.set(xlabel="Time [s]", ylabel=r'Relative error [untiless]', title=r"Relative error of the estimates of the neutron count")
        legend(ax_SE)
        annotate_RKKF(ax_SE, config.stdev_initial_factor, config.stdev_transition_dep, loc_x=0.29)
        
        tstep = self.times[1] - self.times[0]
        plt.show(block=False)
        fig.savefig('neutron_population_tstep{:.1f}.pdf'.format(tstep), bbox_inches='tight')



def plot_param(ax, parname, times, states_UKF, states_EKF, prior, prior_stdev, COVs_UKF=None, COVs_EKF=None, index=None):
    num_markers = 7
    # Initialize lists of lines and labels for the legend:
    lines, lines_std = [], []
    labels, labels_std = [], []
    # Plot the prior estimate and its standard deviation:
    line = ax.axhline(prior, color='k')
    lines.append(line)
    labels.append(Label_UKF.make(parname, 'initial'))
    
    line = ax.axhline(prior + prior_stdev, color='k', linestyle='--')
    lines_std.append(line)
    ax.axhline(prior - prior_stdev, color='k', linestyle='--')
    labels_std.append(
            Label_UKF.make(parname, 'initial_stdev')
            )
    
    y_min = [prior - prior_stdev]
    y_max = [prior + prior_stdev]
    slc = slice(len(times))
    slc_COV = slice(len(times))
    if index is not None:
        slc = (slc, index)
        slc_COV = (slc_COV, index, index)
    if states_UKF is not None:
        method_type = MethodType.UKF
        yerr = [0]
        if COVs_UKF is None:
            line, = ax.plot(
                    times, states_UKF[slc],
                    method_type.get_fmt(),
                    markevery= len(times) // num_markers,
                    **plot_kwargs
                    )
            lines.append(line)
            labels.append(Label_UKF.make(parname, method_type.name))
            
        else:
            yerr = np.sqrt(COVs_UKF[slc_COV])
            polycollection = ax.fill_between(times, states_UKF[slc] + yerr, states_UKF[slc] - yerr, facecolor=method_type.get_color(), alpha=0.2)#, edgecolor='#004400', linestyle='--', linewidth=2)
            line, = ax.plot(
                    times, states_UKF[slc],
                    method_type.get_fmt(),
                    markevery= len(times) // num_markers,
                    **plot_kwargs
                    )
            lines.append(line)
            labels.append(
                    Label_UKF.make(parname, method_type.name)
                    )
            lines_std.append(polycollection)
            labels_std.append(
                    Label_UKF.make(parname, method_type.name+'_stdev')
                    )
        y_min.append(states_UKF[slc][-1] - yerr[-1])
        y_max.append(states_UKF[slc][-1] + yerr[-1])
        num_markers += 2
        
    if states_EKF is not None:
        method_type = MethodType.EKF
        yerr = [0]
        if COVs_EKF is None:
            line, = ax.plot(
                    times, states_EKF[slc],
                    method_type.get_fmt(),
                    markevery= len(times) // num_markers,
                    **plot_kwargs
                    )
            lines.append(line)
            labels.append(Label_UKF.make(parname, method_type.name))
        else:
            yerr = np.sqrt(COVs_EKF[slc_COV])
            polycollection = ax.fill_between(times, states_EKF[slc] + yerr, states_EKF[slc] - yerr, facecolor=method_type.get_color(), alpha=0.2)#, edgecolor='#004400', linestyle='--', linewidth=2)
            line, = ax.plot(
                    times, states_EKF[slc],
                    method_type.get_fmt(),
                    label=Label_UKF.make(parname, method_type.name),
                    markevery= len(times) // num_markers,
                    **plot_kwargs
                    )
            lines.append(line)
            labels.append(
                    Label_UKF.make(parname, method_type.name)
                    )
            lines_std.append(polycollection)
            labels_std.append(
                    Label_UKF.make(parname, method_type.name+'_stdev')
                    )
        y_min.append(states_EKF[slc][-1] - yerr[-1])
        y_max.append(states_EKF[slc][-1] + yerr[-1])
    
    # Add proxy artists where needed to ensure that the legend has 3 rows:
    while len(lines_std) < len(lines):
        # Add empty dummy legend items
        lines_std.append(mlines.Line2D([], [], color='w'))
        labels_std.append(' ')
    lines = np.concatenate((lines, lines_std))
    labels = np.concatenate((labels, labels_std))
    
    bounding = [-0.03,1.06, 1.06, 0.2] # (x, y, width, height)
    if 'rho' in parname:
        bounding[0] = 0.1
        bounding[2] = 0.8
    ax.legend(lines, labels, bbox_to_anchor=bounding, loc='lower left', mode='expand', ncol=2, frameon=True, framealpha=1, edgecolor='k', labelspacing=0.1)
    ax.set(xlabel='Time [s]')
    y_min = min(y_min) - 0.5*prior_stdev
    y_max = max(y_max) + 0.5*prior_stdev
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom=max(0, bottom, y_min), top=min(top, y_max), emit=False)
    return

class Label_UKF:
    info_str = {'initial': r'',
                'initial_stdev': r' ($\mu \pm \sigma$)',
                'UKF_stdev': r' ($\mu \pm \sigma$)',
                'EKF_stdev': r' ($\mu \pm \sigma$)',
                'default': r'',}
    
    def make(parname, mode):
        if mode in Label_UKF.info_str:
            _info_str = Label_UKF.info_str[mode]
        else:
            _info_str = Label_UKF.info_str['default']
        if '_' in mode:
            mode = mode.split('_')[0]
        if '_' in parname:
            name, subscript = parname.split('_')
            symbol = r'$\{0}_{{{1},{2}}}$'.format(name, subscript, mode)
        else:
            symbol = r'$\{0}_{{{1}}}$'.format(parname, mode)
        return symbol + _info_str
#        return symbol

def legend(ax, label_list=None, ncol=1):
    if label_list is not None:
        ax.legend(label_list, loc='upper left', ncol=ncol, frameon=True, framealpha=1, edgecolor='k')
    else:
        ax.legend(loc='upper left', ncol=ncol, frameon=True, framealpha=1, edgecolor='k')

def errorize(label_list):
    return [''.join((r'$|', label.strip('$'), r' - x_1(t^{(i)})|$')) for label in label_list]

def squared_errorize_RKKF(label):
    return ''.join((r'$\frac{\small |', label.strip('$'), r' - z^{(i)}|}{\small z^{(i)}}$'))

def unit(str):
    return r"\hspace{1mm} \mathrm{"+str+"}"

def annot_strs_RKKF(stdev_initial_factor, stdev_transition_dep):
    str_init = r'\sigma_{{initial}} = {0}'.format(stdev_initial_factor)
    str_trans = r"\sigma_{process} = "
    if abs(stdev_transition_dep - 1e-3) < 1e-10:
        str_trans += r'10^{-3}'
    elif abs(stdev_transition_dep - 1e-5) < 1e-12:
        str_trans += r'10^{-5}'
    else:
        str_trans += r'{0}'.format(stdev_transition_dep)
    return str_init, str_trans

def annotate_RKKF(ax, stdev_initial_factor, stdev_transition_dep, loc_x=0.25, loc_y=0.74, tstep=None):
    """For Reactor Kinetics Kalman filter:
    Add a text box to the axes ax, containing information about the parameters:
        stdev_initial_factor
        stdev_transition_dep        
    """
    str_init, str_trans = annot_strs_RKKF(stdev_initial_factor, stdev_transition_dep)
    text = r"\begin{eqnarray*} " + str_init + r" \\ " + str_trans
    if tstep is not None:
        text += r" \\ " + r"\Delta t = {:.1f} ~ \mathrm{{s}}".format(tstep)
    text += r" \\ \end{eqnarray*}"
    ax.text(loc_x, loc_y, text, transform=ax.transAxes, bbox=bbox_propdict)
    

def annotate(ax, dt, stdev_acc, stdev_observation, loc_x=0.07, loc_y=0.5):
    """Add a text box to the axes ax, containing information about the main parameters: The length of the time step dt and the standard deviations chosen for the accelerration and the measurement error, respectively."""
    str_acc = r"\sigma_{acc} =&" + repr(stdev_acc)
    str_acc += unit(r"\frac{m}{s^2}")
    str_obs = r"\sigma_{obs} =&" + repr(stdev_observation) + unit("m")
    str_dt = r"\Delta t =&" + repr(dt) + unit("s")
    text = r"\begin{eqnarray*} " + str_dt + r" \\ " + str_acc + r" \\ " + str_obs + r" \\ \end{eqnarray*}"
    ax.text(loc_x, loc_y, text, transform=ax.transAxes, bbox=bbox_propdict)

def annotate_PRK(ax, dt, rho, loc_x=0.04, loc_y=0.45):
    """Add a text box to the axes ax, containing information about the main parameters: The length of the time step dt between observations, and the reactivity."""
    str_rho = r"\rho =&{0:.5f}".format(rho)
    str_dt = r"\Delta t =&" + repr(dt) + unit("s")
    text = r"\begin{eqnarray*} " + str_rho + r" \\ " + str_dt + r" \\ \end{eqnarray*}"
    ax.text(loc_x, loc_y, text, transform=ax.transAxes, bbox=bbox_propdict)
    
def plot_matrices(M_list, dt, title='Matrices', non_neg=False, include_zero=False):
    """Plot a sequence of matrices stored in the list M_list, using maptlotlib.imshow().
    
    ARGS:
        M_list:       List of numpy Nd-arrays to be plotted.
        dt:           Time step between matrices.
        title:        Title of the plot.
        non_neg:      If set to True (default), the matrices will be plotted
                      using the sequential colormap 'Greys', where white
                      corresponds to 0 and black corresponds to the maximum
                      value which occurs in the entire sequence. Negative
                      values are then colored in red.
                      Otherwise, they will be plotted using the diverging
                      colormap 'bwr_r', where white corresponds to 0, blue to
                      negative values and red to positive values. Both ends of
                      the colorscale will be determined by the maximum absolute
                      value which occurs in the entire sequence.
    """
     
    # We are interested in plotting each matrix in M_list. In order to keep the same color scale, we first need to find the maximum among all entries of all matrices M_list[i].
    M_max = max(np.max(np.abs(M)) for M in M_list)
    # Create a figure to plot all of the matrices in M_list with an aspect ratio of approximately 16c:9r .
    ncols = round(math.sqrt(len(M_list))*4/3)
    nrows = math.ceil(len(M_list)/ncols)
    
    # The first row is used for the title and should be half as high as the imshow plots.
    height_ratios = [2]*nrows
    height_ratios.insert(0,1)
    # The last column is used for the colorbar, which should be less wide than the imshow plots.
    width_ratios = [3]*ncols
    width_ratios.append(1)
    fig, axes_list = plt.subplots(ncols=ncols+1, nrows=nrows+1, figsize = [figure_width, figure_width*(nrows+1)/(ncols+1)], gridspec_kw={'width_ratios': width_ratios, 'height_ratios': height_ratios, 'hspace':0.35})
    for i, ax in enumerate(axes_list[1:,:-1].flat):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if not include_zero:
            i = i+1
        ax.set_xlabel(r"$t^{{({0})}} = {1}{2}$".format(i, i*dt, unit('s')))
    for i in range(0, nrows*ncols - len(M_list)):
        axes_list.flat[-2-i].axis('off')
    
    # Use the first row to plot the title
    ax_title = plt.subplot2grid((2*nrows+1, 1), (0,0))
    ax_title.axis('off')
    ax_title.text(0.5, 0, title, horizontalalignment='center', fontsize='x-large')
    # Use the last column to plot the colorbar:
    ax_colorbar = plt.subplot2grid((2*nrows+1, 3*ncols+1), (1, 3*ncols), rowspan=2*nrows)
    
    if non_neg:
        # Pick a sequential colormap to represent the purely non-negative matrix entries.
        M_cmap = matplotlib.cm.get_cmap(name='Greys')
        # Make sure that negative numbers are immediately apparent!
        M_cmap.set_under(color='r')
        M_min = 0
    else:
        # Pick a diverging colormap to represent positive and negative values.
        M_cmap = matplotlib.cm.get_cmap(name='bwr_r')
        M_min = - M_max
    for i, ax, M in zip(range(len(M_list)), axes_list[1:,:-1].flat, M_list):
        if i == 0:
            image = ax.imshow(M, vmin=M_min, vmax=M_max, cmap=M_cmap)
            plt.colorbar(image, cax=ax_colorbar)
        else:
            ax.imshow(M, vmin=M_min, vmax=M_max, cmap=M_cmap)
    plt.show(block=False)
    return fig

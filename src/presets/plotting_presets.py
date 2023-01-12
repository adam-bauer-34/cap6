"""CDDS @ UIUC plotting presets.

Adam Michael Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
9.12.2022
"""

import matplotlib.pyplot as plt

# plotting lists
color_list = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'] * 4
marker_list = ['o', 's', 'P', '*', '<', 'v', '>', 'p'] * 4
linestyle_list = ['solid'] * 8 + ['dashdot'] * 8 + ['dashed'] * 8 + ['dotted'] * 8

# dictionaries of params
cdds_params={'axes.linewidth': 3,
 'axes.axisbelow': False,
 'axes.edgecolor': 'grey',
 'axes.facecolor': 'None',
 'axes.grid': False,
 'axes.labelcolor': 'grey',
 'axes.spines.right': False,
 'axes.spines.top': False,
 'axes.titlesize': 20,
 'axes.labelsize': 20,
 'axes.titlelocation': 'left',
 'figure.facecolor': 'white',
 'figure.figsize': (18, 10),
 'lines.solid_capstyle': 'round',
 'lines.linewidth': 2.5,
 'patch.edgecolor': 'w',
 'patch.force_edgecolor': True,
 'text.color': 'grey',
 'legend.frameon': False,
 'xtick.bottom': True,
 'xtick.major.width': 3,
 'xtick.major.size': 6,
 'xtick.color': 'grey',
 'xtick.direction': 'out',
 'xtick.top': False,
 'ytick.color': 'grey',
 'ytick.direction': 'out',
 'ytick.left': True,
 'ytick.right': False,
 'ytick.color' : 'grey',
 'ytick.major.width': 3, 
 'ytick.major.size': 6,
 'axes.prop_cycle': plt.cycler(color=color_list, linestyle=linestyle_list),
 'font.size': 16,
 'font.family': 'serif'}

cdds_params_w_markers={'axes.linewidth': 3,
 'axes.axisbelow': False,
 'axes.edgecolor': 'grey',
 'axes.facecolor': 'None',
 'axes.grid': False,
 'axes.labelcolor': 'grey',
 'axes.spines.right': False,
 'axes.spines.top': False,
 'axes.titlesize': 20,
 'axes.labelsize': 20,
 'axes.titlelocation': 'left',     
 'figure.facecolor': 'white',
 'figure.figsize': (18, 10),
 'lines.solid_capstyle': 'round',
 'lines.linewidth': 2.5,
 'lines.markersize': 8, 
 'patch.edgecolor': 'w',
 'patch.force_edgecolor': True,
 'text.color': 'grey',
 'legend.frameon': False,
 'xtick.bottom': True,
 'xtick.major.width': 3,
 'xtick.major.size': 6,
 'xtick.color': 'grey',
 'xtick.direction': 'out',
 'xtick.top': False,
 'ytick.color': 'grey',
 'ytick.direction': 'out',
 'ytick.left': True,
 'ytick.right': False,
 'ytick.color' : 'grey',
 'ytick.major.width': 3, 
 'ytick.major.size': 6,
 'axes.prop_cycle': plt.cycler(color=color_list, linestyle=linestyle_list, marker=marker_list),
 'font.size': 16,
 'font.family': 'serif'}
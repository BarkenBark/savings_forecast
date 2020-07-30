import matplotlib.pyplot as plt
import numpy as np

def get_lims(ax):
    max_x = -np.inf
    min_x = np.inf
    for line in ax.get_lines():
        this_max_x = max(line.get_xdata())
        this_min_x = min(line.get_xdata())
        max_x = this_max_x if this_max_x > max_x else max_x
        min_x = this_min_x if this_min_x < min_x else min_x
    lims_x = [min_x, max_x]

    max_y = -np.inf
    min_y = np.inf
    for line in ax.get_lines():
        this_max_y = max(line.get_ydata())
        this_min_y = min(line.get_ydata())
        max_y = this_max_y if this_max_y > max_y else max_y
        min_y = this_min_y if this_min_y < min_y else min_y
    lims_y = [min_y, max_y]

    return lims_x, lims_y

def autoscale(ax, axis='both', factor=1.2):
    def center_delta(lims, factor):
        center = (lims[1]+lims[0])/2
        delta = factor*(lims[1]-lims[0])/2
        return center, delta

    lims_x, lims_y = get_lims(ax)
    if axis=='both' or axis=='x':
        lims = lims_x
        center, delta = center_delta(lims, factor)
        ax.set_xlim(center-delta, center+delta)
    if axis=='both' or axis=='y':
        lims = lims_y
        center, delta = center_delta(lims, factor)
        ax.set_ylim(center-delta, center+delta)
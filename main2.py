import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from plot_utils import autoscale, get_lims

cfg_init = {
    'value_init': 336000,
    'deposit_monthly': 10000,
    'return_annual_mean': (1+0.008027)**12-1, #0.04,
    'return_monthly_std': 0.054027,
    'standard_rate_annual_mean': 0.003,
    'n_months': 480,
    'annual_raise': 0.03,
    'months_of_interest': [60, 120, 240, 360],
    'n_trials': 500
}

def sample_monthly_return(return_monthly_mean, return_monthly_std=0):
    if return_monthly_std == 0:
        return return_monthly_mean
    else:
        return np.random.normal(return_monthly_mean, return_monthly_std)

def sample_monthly_deposit(month_idx, deposit_amount, annual_raise):
    return deposit_amount*(1+annual_raise)**(month_idx // 12)

def sample_standard_rate_annual(standard_rate_annual_mean):
    return standard_rate_annual_mean

def tax_isk(standard_rate, value_history, deposit_history, month_idx):
    tax_rate = max(0.0125, 0.01+standard_rate)
    return tax_rate*(value_history[month_idx-12]+value_history[month_idx-9]+value_history[month_idx-6]+value_history[month_idx-3] + sum(deposit_history[month_idx-12:month_idx])) / 4



# Calculate the new portfolio value as we enter a new month, expand histories
def get_new_portfolio_value(value_history, return_history, deposit_history, standard_rate_annual_mean):
    value_curr = value_history[-1] # v_i, value of portfolio at start of month i, before deposit
    return_curr = return_history[-1] # r_i, total return at the end of month i
    deposit_curr = deposit_history[-1] # d_i, deposited amount at the start of month i
    #assert len(value_history) == (len(return_history) + 1), f"{len(value_history)}, {len(return_history)}"
    #assert len(value_history) == (len(deposit_history) + 1)

    value_new = value_curr * (1 + return_curr)
    value_new = value_new + deposit_curr

    month_idx = len(value_history)-1
    if (month_idx%12 == 0) and (month_idx > 0):
        standard_rate = sample_standard_rate_annual(standard_rate_annual_mean)
        value_new = value_new - tax_isk(standard_rate, value_history, deposit_history, month_idx)

    return value_new

def simulate_trajectory(cfg):
#n_months, value_init, deposit_amount=0, return_annual_mean=0, standard_rate_annual_mean=0.0125, annual_raise=0):
    value_history = []
    deposit_history = []
    return_history = []

    # Pre-calculations
    return_monthly_mean = (1+cfg['return_annual_mean'])**(1/12)-1
    return_monthly_std = cfg['return_monthly_std']

    value_history.append(cfg['value_init'])
    for iMonth in range(cfg['n_months']):
        return_history.append(sample_monthly_return(return_monthly_mean, return_monthly_std))
        deposit_history.append(sample_monthly_deposit(iMonth, cfg['deposit_monthly'], cfg['annual_raise']))
        value_new = get_new_portfolio_value(value_history, return_history, deposit_history, cfg['standard_rate_annual_mean'])
        value_history.append(value_new)

    return value_history, return_history, deposit_history

class MainWindow:

    MAX_NO_SLIDERS = 20

    def __init__(self, cfg):
        self.cfg = cfg
        self.__initialize_figure(self.cfg)
        val, ret, dep = simulate_trajectory(self.cfg)
        self.__plot_trajectory(val, val, val, dep)
        plt.show()

    def __add_slider(self, key, vallims=None, valinit=None, valsteporder=None, title=None, **kwargs):

        def get_slider_lims(val):
            high = val*2
            low = val/2
            return low, high

        if valinit is None:
            valinit = self.cfg[key]

        if vallims is None:
            vallims = get_slider_lims(valinit)

        if valsteporder is None:
            valsteporder = -2
        valstep = 10**valsteporder

        if title is None: 
            title = key

        valfmt = "%d" if valsteporder > 0 else f"%1.{-valsteporder}f"

        if self.slider_counter >= self.MAX_NO_SLIDERS:
            raise Error("Too many sliders")
        ax_slider = self.fig.add_subplot(self.MAX_NO_SLIDERS, 2, self.slider_counter*2+1)
        self.slider_counter += 1
        self.sliders[key] = Slider(ax_slider, title, vallims[0], vallims[1], valinit=valinit, valstep=valstep, orientation='horizontal', valfmt=valfmt, **kwargs)
        on_changed = lambda x: self.__recalculate_and_draw(key, x)
        self.sliders[key].on_changed(on_changed)

    def __initialize_figure(self, cfg):

        def create_deposit_slider(ax, title='Monthly deposit', **kwargs):
            deposit_slider_lims = [0, 20000]
            deposit_slider_init = cfg['deposit_monthly']
            deposit_slider_step = 100
            return Slider(ax, title, deposit_slider_lims[0], deposit_slider_lims[1], valinit=deposit_slider_init, valstep=deposit_slider_step, **kwargs)

        def create_return_slider(ax, title, **kwargs):
            return_slider_lims = [-0.1, 0.2]
            return_slider_init = cfg['return_annual_mean']
            return_slider_step = 0.001
            return Slider(ax, title, return_slider_lims[0], return_slider_lims[1], valinit=return_slider_init, valstep=return_slider_step, **kwargs)

        self.fig = plt.figure(figsize=(16,9))
        self.ax_plot = self.fig.add_subplot(1,2,2)
        self.ax_plot.grid(axis='x', which='major')
        self.ax_plot.grid(axis='y', which='both')
        self.ax_plot.set_xlabel("Years from now")
        self.ax_plot.set_ylabel("Capital (kr)")
        self.ax_plot.set_xlim([0, (self.cfg['n_months']+1)/12])
        self.trajectory_line = None        
        self.trajectory_line_low = None        
        self.trajectory_line_high = None
        self.deposit_line = None

        self.sliders = {}
        self.slider_counter = 0
        # ax_deposit_slider = self.fig.add_subplot(self.MAX_NO_SLIDERS,2,1)
        # ax_return_slider = self.fig.add_subplot(self.MAX_NO_SLIDERS,2,3)
        # self.slider_counter = 2

        # self.sliders['deposit_monthly'] = create_deposit_slider(ax_deposit_slider, 'Monthly deposit (kr)', orientation='horizontal')
        # on_changed_deposit = lambda x: self.__recalculate_and_draw('deposit_monthly', x)
        # self.sliders['deposit_monthly'].on_changed(on_changed_deposit)

        # self.sliders['return_annual_mean'] = create_return_slider(ax_return_slider, 'Expected annual return', orientation='horizontal')
        # on_changed_return = lambda x: self.__recalculate_and_draw('return_annual_mean', x)
        # self.sliders['return_annual_mean'].on_changed(on_changed_return)
        self.__add_slider('deposit_monthly', title="Monthly deposit (kr)", vallims=[0,20000], valsteporder=2)
        self.__add_slider('return_annual_mean', title="Annual return", vallims=[-0.10, 0.20], valsteporder=-3)
        self.__add_slider('annual_raise', title="Annual raise", vallims=[0, 0.05], valsteporder=-3)
        self.__add_slider('value_init', title="Initial value (kr)", vallims=[0, 500000], valsteporder=4)
        self.__add_slider('standard_rate_annual_mean', title="Standard rate", vallims=[-0.03, 0.06], valsteporder=-3)

    def __update_cfg(self, key, val):
        if key not in self.cfg.keys():
            error_msg = f"{key} not in cfg"
            raise KeyError(error_msg)
        self.cfg[key] = val

    def __recalculate_and_draw(self, key, val):
        self.__update_cfg(key, val)
        vals = np.zeros((self.cfg['n_trials'], self.cfg['n_months']+1))
        for i in range(self.cfg['n_trials']):
            val, _, dep = simulate_trajectory(self.cfg)
            vals[i, :] = val
        val_mean = list(np.mean(vals, axis=0))
        val_95 = list(np.quantile(vals, 0.95, axis=0))
        val_05 = list(np.quantile(vals, 0.05, axis=0))
        print(val_05)
        self.__plot_trajectory(val_mean, val_05, val_95, dep)

    def __plot_trajectory(self, val, val_low, val_high, dep):
        if self.trajectory_line is not None:
            self.trajectory_line.set_ydata(val)
        else:
            self.trajectory_line = self.ax_plot.semilogy([i/12 for i in range(len(val))], val)[0]

        if self.trajectory_line_low is not None:
            self.trajectory_line_low.set_ydata(val_low)
        else:
            self.trajectory_line_low = self.ax_plot.semilogy([i/12 for i in range(len(val_low))], val_low)[0]

        if self.trajectory_line_high is not None:
            self.trajectory_line_high.set_ydata(val_high)
        else:
            self.trajectory_line_high = self.ax_plot.semilogy([i/12 for i in range(len(val_high))], val_high)[0]

        if self.deposit_line is not None:
            self.deposit_line.set_ydata(self.cfg['value_init']+np.cumsum(np.array(dep)))
        else:
            self.deposit_line = self.ax_plot.semilogy([i/12 for i in range(len(dep))], self.cfg['value_init']+np.cumsum(np.array(dep)))[0]

        # Update scale if necessary
        lims_y = self.ax_plot.get_ylim()
        if (max(val_high) > lims_y[1]*0.9) or (max(val_high) < lims_y[1] / 10) or (min(val_low) < lims_y[0]):
            print("Autoscaling")
            autoscale(self.ax_plot, axis='y', factor=2)


def main(cfg):
    main_window = MainWindow(cfg)

if __name__=="__main__":
    main(cfg_init)

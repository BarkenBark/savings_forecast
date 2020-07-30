import pickle
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

with open('kde.pickle', 'rb') as file:
        monthly_return_kde = pickle.load(file)




class Simulator:

    def __init__(self, cfg):
        self.cfg = cfg

    def set_cfg(self, cfg):
        self.cfg = cfg

    def update_cfg_item(self, key, val):
        if key not in self.cfg.keys():
            error_msg = f"{key} not in cfg"
            raise KeyError(error_msg)
        self.cfg[key] = val

    def simulate_trajectory(self):
    #n_months, value_init, deposit_amount=0, return_annual_mean=0, standard_rate_annual_mean=0.0125, annual_raise=0):
        value_history = []
        deposit_history = []
        return_history = []

        # Pre-calculations
        return_monthly_mean = (1+self.cfg['return_annual_mean'])**(1/12)-1
        return_monthly_std = self.cfg['return_monthly_std']

        value_history.append(self.cfg['value_init'])
        for iMonth in range(self.cfg['n_months']):
            return_history.append(self.__sample_monthly_return(return_monthly_mean, return_monthly_std))
            deposit_history.append(self.__sample_monthly_deposit(iMonth, self.cfg['deposit_monthly'], self.cfg['annual_raise']))
            value_new = self.__get_new_portfolio_value(value_history, return_history, deposit_history, self.cfg['standard_rate_annual_mean'])
            value_history.append(value_new)

        return value_history, return_history, deposit_history, self.cfg['value_init']

    def simulate_trajectories(self, n_trajectories):
        vals = np.zeros((n_trajectories, self.cfg['n_months']+1))
        for i in range(self.cfg['n_trials']):
                val, _, dep, _ = self.simulate_trajectory()
                vals[i, :] = val
        val_mean = list(np.mean(vals, axis=0))
        val_95 = list(np.quantile(vals, 0.95, axis=0))
        val_05 = list(np.quantile(vals, 0.05, axis=0))
        return val_mean, val_05, val_95, dep, self.cfg['value_init']

    # Calculate the new portfolio value as we enter a new month, expand histories
    def __get_new_portfolio_value(self, value_history, return_history, deposit_history, standard_rate_annual_mean):
        value_curr = value_history[-1] # v_i, value of portfolio at start of month i, before deposit
        return_curr = return_history[-1] # r_i, total return at the end of month i
        deposit_curr = deposit_history[-1] # d_i, deposited amount at the start of month i
        #assert len(value_history) == (len(return_history) + 1), f"{len(value_history)}, {len(return_history)}"
        #assert len(value_history) == (len(deposit_history) + 1)

        value_new = value_curr * (1 + return_curr)
        value_new = value_new + deposit_curr

        month_idx = len(value_history)-1
        if (month_idx%12 == 0) and (month_idx > 0):
            standard_rate = self.__sample_standard_rate_annual(standard_rate_annual_mean)
            value_new = value_new - self.__tax_isk(standard_rate, value_history, deposit_history, month_idx)

        return value_new

    def __sample_monthly_return(self, return_monthly_mean, return_monthly_std=0):
        return monthly_return_kde.resample(1)[0,0]
        if return_monthly_std == 0:
            return return_monthly_mean
        else:
            return np.random.normal(return_monthly_mean, return_monthly_std)

    def __sample_monthly_deposit(self, month_idx, deposit_amount, annual_raise):
        return deposit_amount*(1+annual_raise)**(month_idx // 12)

    def __sample_standard_rate_annual(self, standard_rate_annual_mean):
        return standard_rate_annual_mean

    def __tax_isk(self, standard_rate, value_history, deposit_history, month_idx):
        tax_rate = max(0.0125, 0.01+standard_rate)
        return tax_rate*(value_history[month_idx-12]+value_history[month_idx-9]+value_history[month_idx-6]+value_history[month_idx-3] + sum(deposit_history[month_idx-12:month_idx])) / 4


class MainWindow:

    MAX_NO_SLIDERS = 20

    def __init__(self, cfg):
        self.simulator = Simulator(cfg)
        self.n_trials = cfg['n_trials']
        self.__initialize_figure(cfg['n_months'])
        val, ret, dep, val_init = self.simulator.simulate_trajectory()
        self.__plot_trajectory(val, val, val, dep, val_init)
        plt.show()

    def __add_slider(self, key, vallims=None, valinit=None, valsteporder=None, title=None, **kwargs):

        def get_slider_lims(val):
            high = val*2
            low = val/2
            return low, high

        if valinit is None:
            valinit = self.simulator.cfg[key]

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

    def __initialize_figure(self, n_months):

        self.fig = plt.figure(figsize=(16,9))

        self.ax_plot = self.fig.add_subplot(1,2,2)
        self.ax_plot.grid(axis='x', which='major')
        self.ax_plot.grid(axis='y', which='both')
        self.ax_plot.set_xlabel("Years from now")
        self.ax_plot.set_ylabel("Capital (kr)")
        self.ax_plot.set_xlim([0, (n_months+1)/12])
        
        self.trajectory_line = None
        self.trajectory_line_low = None
        self.trajectory_line_high = None
        self.deposit_line = None

        self.sliders = {}
        self.slider_counter = 0

        self.__add_slider('deposit_monthly', title="Monthly deposit (kr)", vallims=[0,20000], valsteporder=2)
        self.__add_slider('return_annual_mean', title="Annual return", vallims=[-0.10, 0.20], valsteporder=-3)
        self.__add_slider('annual_raise', title="Annual raise", vallims=[0, 0.05], valsteporder=-3)
        self.__add_slider('value_init', title="Initial value (kr)", vallims=[0, 500000], valsteporder=4)
        self.__add_slider('standard_rate_annual_mean', title="Standard rate", vallims=[-0.03, 0.06], valsteporder=-3)

    def __recalculate_and_draw(self, key, val):
        self.simulator.update_cfg_item(key, val)
        val_mean, val_05, val_95, dep, val_init = self.simulator.simulate_trajectories(self.n_trials)
        self.__plot_trajectory(val_mean, val_05, val_95, dep, val_init)

    def __plot_trajectory(self, val, val_low, val_high, dep, value_init):
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
            self.deposit_line.set_ydata(value_init+np.cumsum(np.array(dep)))
        else:
            self.deposit_line = self.ax_plot.semilogy([i/12 for i in range(len(dep))], value_init+np.cumsum(np.array(dep)))[0]

        # Update scale if necessary
        lims_y = self.ax_plot.get_ylim()
        if (max(val_high) > lims_y[1]*0.9) or (max(val_high) < lims_y[1] / 10) or (min(val_low) < lims_y[0]):
            print("Autoscaling")
            autoscale(self.ax_plot, axis='y', factor=2)


def main(cfg):
    main_window = MainWindow(cfg)

if __name__=="__main__":
    main(cfg_init)

import numpy as np
import matplotlib.pyplot as plt

value_init = 336000
deposit_amount = 10000

return_annual_mean = 0.04
standard_rate_mean = 0.015
n_months = 360

annual_raise = 0.02

def sample_monhtly_return(return_monhtly_mean):
    return return_monthly_mean
    #return np.random.normal()

def sample_monthly_deposit(month_idx, deposit_amount, annual_raise):
    return deposit_amount*(1+annual_raise)**(month_idx // 12)

def sample_standard_rate_annual(standard_rate_annual_mean):
    return standard_rate_annual_mean

def tax_isk(standard_rate, value_history, deposit_history, month_idx):
    rate = min(0.0125, standard_rate)
    return rate*(value_history[month_idx-12]+value_history[month_idx-9]+value_history[month_idx-6]+value_history[month_idx-3] + sum(deposit_history[month_idx-12:month_idx])) / 4

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

def simulate_trajectory(n_months, value_init, deposit_amount=0, return_annual_mean=0, standard_rate_annual_mean=0.0125, annual_raise=0):
    value_history = []
    deposit_history = []
    return_history = []

    # Pre-calculations
    return_monthly_mean = (1+return_annual_mean)**(1/12)-1

    value_history.append(value_init)
    for iMonth in range(n_months):
        return_history.append(sample_monhtly_return(return_monthly_mean))
        deposit_history.append(sample_monthly_deposit(iMonth, deposit_amount, annual_raise))
        value_new = get_new_portfolio_value(value_history, return_history, deposit_history)
        value_history.append(value_new)

    return value_history, return_history, deposit_history

def plot_trajectory(value_history):
    plt.plot(value_history)

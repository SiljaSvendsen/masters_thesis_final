from Module_aic_chi2 import *
import pandas as pd
import numpy as np


# import data
all_data = pd.read_csv('../time_delay_diff_241021.csv')

# get data
data1 = get_data(all_data, networks='GNEvGN', parameter='KaaNG', scalefactor=1.0)
data2 = get_data(all_data, networks='GNEvGN', parameter='KaaNG', scalefactor=2.0)
data3 = get_data(all_data, networks='GNEvGN', parameter='KaaNG', scalefactor=3.0)
data3 = get_data(all_data, networks='GNEvGN', parameter='KaaNG', scalefactor=4.0)

# perform analysis

dataset = data1                                    #### EDIT####
model = gaussian
title = "GNEvGN, p = KaaNG, Scale Factor = 1.0"
datacolor = "blue"

# 0. Plot data
axx = plot_data(dataset, title, color=datacolor)

# 1. Define initial parameter guesses and bounds   #### EDIT####
p0 = [1.0,1.0]
bounds = [[0, np.inf], [-np.inf,np.inf]]

# 2. Perform chi2 test to find best fit parameters
chi2_value, ndof_value, params_key, chi2_fit_params, chi2_fit_errors, chi2_p_value = chi2_fit(dataset, model, p0, bounds=bounds)

# 3. Perform AIC test to rank the models
#    use chi2_fit_params as initial guess
nll, nll_fit_params, nll_fit_errors, aicc = null_fit_aic(dataset, model, chi2_fit_params, bounds=bounds) 

# 4. Plot the best fit model on top of the data
plot_fits(min(dataset), max(dataset), model, chi2_fit_params, ax=axx, color="red")  # chi2 fit
plot_fits(min(dataset), max(dataset), model, nll_fit_params, ax=axx, color="green",
          savefig=True, filename=f"{title}") # unbinned log-likelihood fit


# 5. Save the best fit parameters and statistics to a file
filename = f"fit_results_{title}.txt"
fit_info = []

fit_info.append(f"Model: {model.__name__}")
fit_info.append(f"Data: {title}")
fit_info.append(f"=================")
# 5.1 chi2 test results
fit_info.append(f"Chi2 test results")
fit_info.append(f"=================")
fit_info.append(f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {chi2_value:.1f} / {ndof_value:.0f} = {chi2_value/ndof_value:.1f}")
fit_info.append(f"P($\\chi^2$,Ndof) = {chi2_p_value:.3f}")
for p, v, e in zip(params_key, chi2_fit_params, chi2_fit_errors):
    fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

# 5.2 negative log-likelihood results
fit_info.append(f"Negative log-likelihood results")
fit_info.append(f"=================")
fit_info.append(f"NLL = {nll:.1f}")
for p, v, e in zip(params_key, nll_fit_params, nll_fit_errors):
    fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

# 5.3 AIC test result
fit_info.append(f"AIC test results")
fit_info.append(f"=================")
fit_info.append(f"AICc = {aicc:.1f}")

# join data and add new line characters
fit_info = "\n".join(fit_info)
# save to file - overwrite if file already exists
with open(filename, "w") as f: f.write(fit_info)





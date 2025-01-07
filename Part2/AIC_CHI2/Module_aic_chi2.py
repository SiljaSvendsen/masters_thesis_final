import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit # for fitting
import os # for file handling
import pandas as pd # for reading in data
from scipy.stats import norm, chi2 # for calculating p-values


# Define the Akaike Information Criterion (AIC)

def biascorrected_aic(nll, n, k):
    """
    Calculate the Akaike Information Criterion (AIC) from the negative log-likelihood and the number of parameters.

    Parameters
    ----------
    nll : float
        The negative (unbinned) log-likelihood of the model.
    k : int
        The number of parameters in the model.
    n : int
        The number of data points.

    Returns
    -------
    float
        The bias corrected AIC value.
    """
    aic = 2 * k + 2 * nll

    return aic + 2 * k * (k + 1) / (n - k - 1)


# define function to perfrom a chi2 fit

def chi2_fit(data, model, initial_guess, bounds):
    """
    Calculate the chi-squared fit of a model to the data.

    The data is binned (# bins = sqrt(N)) and the chi-squared statistic is calculated.
    
    parameters
    ----------
    data : array
        The observed data to fit.

    model : function

    initial_guess : list

    bounds : list

    returns
    -------
    chi2 : float
        The chi-squared statistic of the fit.
    ndof : int
        The number of degrees of freedom of the fit.
    model parameters : dict
        The best-fit parameters of the model.
    model errors : dict
        The uncertainties on the best-fit parameters.
    p-value : float
        The p-value of the fit.
    """
    # Define the Chi-squared goodness of fit test

    def chi2(model, params, x, y, yerr):
        """
        Calculate the chi-squared statistic for a model given its parameters and the observed data.

        Parameters
        ----------
        model : function
            The model function that predicts the expected values.
        params : list
            The parameters of the model.
        x : array
            The x values of the observed data.
        y : array
            The y values of the observed data.
        yerr : array
            The uncertainties on the observed data.

        Returns
        -------
        float
            The chi-squared statistic.
        """
        y_pred = model(x, *params)
        chi2 = np.sum(((y - y_pred) / yerr) ** 2)

        return chi2

    # bin the data
    y, bin_edges = np.histogram(data, bins=int(np.sqrt(len(data))), density=True)
    x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    yerr = np.sqrt(y)
    
    # filter out zero bins
    non_zero_mask = y > 0
    x = x[non_zero_mask]
    y = y[non_zero_mask]
    yerr = yerr[non_zero_mask]

    # perform the fit
    chi2_value = chi2(model=model, params=initial_guess, x=x, y=y, yerr=yerr)
    m = Minuit(chi2_value, model=model, params=initial_guess, x=x, y=y, yerr=yerr, limit_sigma=bounds)
    m.errordef=1
    m.migrad()

    # calculate p-value
    p_value = chi2.sf(m.fval, m.ndof)
    
    print(m.parameters)
    return m.fval, m.ndof, m.parameters, m.values, m.errors, p_value


# define function to perform a likelihood fit

def null_fit_aic(data, model, initial_guess, bounds):
    """
    Calculate the negative (unbinned) log-likelihood fit of a model to the data.

    The data is unbinned and the negative log-likelihood is calculated.
    
    parameters
    ----------
    data : array
        The observed data to fit.

    model : function

    initial_guess : list

    bounds : list

    returns
    -------
    nll : float
        The negative log-likelihood of the fit.
    model parameters : dict
        The best-fit parameters of the model.
    model errors : dict
        The uncertainties on the best-fit parameters.
    aic : float
        The AIC statistic of the fit.
    """
    # Define negative (unbinned) log-likelihood function
    def null(model, data, params):
        """
        Calculate the negative log-likelihood of a model given the observed data.

        Parameters
        ----------
        model : function
            The model function that predicts the expected values.
        data : array
            The observed data.
        params : list
            The parameters of the model.

        Returns
        -------
        float
            The negative log-likelihood.
        """
        return -np.sum(np.log(model(data, *params)))

    null_value=null(model=model, data=data, params=initial_guess)
    # perform the fit
    m = Minuit(null_value, model=model, params=initial_guess, data=data)
    m.migrad()

    # calculate AIC
    aic = biascorrected_aic(m.fval, len(data), len(initial_guess))

    return m.fval, m.values, m.errors, aic

# Define model functions

def gaussian(x, params):
    """
    Calculate the Gaussian probability density function at x given the parameters.
    
    Parameters
    ----------
    x : float
        The x value at which to evaluate the Gaussian.
    params : list
        The parameters of the Gaussian: [sigma, mu].
    
    Returns
    -------
    float
        The value of the Gaussian PDF at x.
    """
    # For readability, unpack the parameters
    sigma = params[0]
    mu = params[1]
    return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def double_gaussian(x, params):
    """
    Calculate the sum of two Gaussian probability density functions at x given the parameters.

    Parameters
    ----------
    x : float
        The x value at which to evaluate the double Gaussian.
    params : list
        The parameters of the double Gaussian: [w, sigma1, mu1, sigma2, mu2].
    
    Returns
    -------
    float
        The value of the double Gaussian PDF at x.
    """

    # for readability, unpack the parameters
    w = params[0]
    sigma1, mu1 = params[1:3]
    sigma2, mu2 = params[3:]

    return w * gaussian(x, [sigma1, mu1]) + (1 - w) * gaussian(x, [sigma2, mu2])

def skewed_gaussian(x, params):
    """
    Calculate the skewed Gaussian probability density function at x given the parameters.

    Parameters
    ----------
    x : float
        The x value at which to evaluate the skewed Gaussian.
    params : list
        The parameters of the skewed Gaussian: [sigma, mu, alpha].
    
    Returns
    -------
    float
        The value of the skewed Gaussian PDF at x.

    """
    # for readability, unpack the parameters
    sigma = params[0]
    mu = params[1]
    alpha = params[2]

    z = (x - mu) / sigma
    return 2 / sigma * gaussian(x, [sigma, mu]) * norm.cdf(alpha * z)

def skewed_double_gaussian(x, params):
    """
    Calculate the sum of two skewed Gaussian probability density functions at x given the parameters.

    Parameters
    ----------
    x : float
        The x value at which to evaluate the double Gaussian.
    params : list
        The parameters of the double Gaussian: [w, sigma1, mu1, alpha1, sigma2, mu2, alpha2].
    
    Returns
    -------
    float
        The value of the double Gaussian PDF at x.

    """
    # for readability, unpack the parameters
    w = params[0]
    sigma1, mu1, alpha1 = params[1:4]
    sigma2, mu2, alpha2 = params[4:]

    return w * skewed_gaussian(x, [sigma1, mu1, alpha1]) + (1 - w) * skewed_gaussian(x, [sigma2, mu2, alpha2])


# define function to pick out data from pandas dataframe

def get_data(dataframe, networks, parameter, scalefactor):

    mask = ((dataframe['two_networks'] == networks) \
            & (dataframe['parameter'] == parameter) \
            & (dataframe['rel change'] == scalefactor))
    
    return dataframe[mask]['time diff'].values

# plot data

def plot_data(data, title, color, ax=None, font_size=15):
    """
    Plot the data as a histogram.

    Parameters
    ----------
    data : array
        The data to plot.
    title : str
        The title of the plot.
    color : str
        The color of the data points.
    ax : matplotlib axis, optional
        The axis on which to plot the data. If None, a new figure is created.
    font_size : int, optional
        The font size of the axis labels and legend.
    
    Returns
    -------
    matplotlib axis
        The axis on which the data is
    
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax=ax

    # bin the data
    y, bin_edges = np.histogram(data, bins=int(np.sqrt(len(data))), density=True)
    x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    yerr = np.sqrt(y)/np.sqrt(len(data)*np.diff(bin_edges)[0])

    # filter out zero bins
    #non_zero_mask = y > 0
    #x = x[non_zero_mask]
    #y = y[non_zero_mask]
    #yerr = yerr[non_zero_mask]

    # plot binned data
    ax.errorbar(x, y, yerr=yerr, marker=".", drawstyle="steps-mid", color=color, label='Data')
    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel("$\Delta t$", fontsize=font_size)
    ax.set_ylabel('Density', fontsize=font_size)
    ax.legend(fontsize=font_size)

    # figure formatting
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()

    plt.show()

    return ax

def plot_fits(xmin, xmax, model, params, label, ax=None, color='red', savefig=False, *filename):
    """
    Plot the model fit on top of the data.

    Parameters
    ----------
    xmin : float
        The minimum x value to plot.
    xmax : float
        The maximum x value to plot.
    model : function
        The model function to plot.
    params : list
        The parameters of the model.
    label : str
        The label of the model.
    ax : matplotlib axis, optional
        The axis on which to plot the data. If None, a new figure is created.
    color : str, optional
        The color of the model line.
    savefig : bool, optional
        If True, the figure is saved to a file.
    filename : str, optional
        The filename to save the figure to.
    
    Returns
    -------
    matplotlib axis
        The axis on which the data is plotted.
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax = ax
    
    x = np.linspace(xmin, xmax, 1000)
    y = model(x, params)

    ax.plot(x, y, color=color, label=label)

    plt.show()

    if savefig:
        plt.savefig(filename + '.pdf', dpi=600)



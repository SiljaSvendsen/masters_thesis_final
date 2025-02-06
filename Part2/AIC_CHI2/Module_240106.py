import numpy as np
from scipy.stats import norm
from iminuit import describe, Minuit

######################################################
# Define model functions                            #
######################################################
def gaussian(x, sigma, mu):
    return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5*(x-mu)**2/sigma**2)

def double_gaussian(x, w, sigma1, mu1, sigma2, mu2):
    return w*gaussian(x, sigma1, mu1) + (1-w)*gaussian(x, sigma2, mu2)

def skewed_gaussian(x, sigma, mu, alpha):
    return 2/sigma * gaussian(x, sigma, mu) * norm.cdf(alpha*(x-mu)/sigma)

def double_skewed_gaussian(x, w, sigma1, mu1, alpha1, sigma2, mu2, alpha2):
    return w * skewed_gaussian(x, sigma1, mu1, alpha1) + (1-w) * skewed_gaussian(x, sigma2, mu2, alpha2)


######################################################
# Define NegativeLogLikelihood, Chi2, AIC functions  #
######################################################

class NegativeLogLikelihood:
    def __init__(self, model, x_data):
        """
        Initialize the NLL class.

        Parameters:
        - model: The model function to fit.
        - x_data: Data points (independent variable).
        """
        self.model = model
        self.x_data = x_data
        self.param_names = describe(model)[1:]  # Get parameter names dynamically

    def __call__(self, *params):
        """
        Compute the Negative Log-Likelihood (NLL).

        Parameters:
        - params: Model parameters, dynamically matched to parameter names.

        Returns:
        - Negative log-likelihood value.
        """
        # Map parameter values to their names
        param_dict = dict(zip(self.param_names, params))
        
        # Calculate model predictions
        y = self.model(self.x_data, **param_dict)
        
        # Check for invalid values in the model output
        if np.any(y <= 0):
            return np.inf  # Return a large value if PDF is invalid
        
        # Compute the negative log-likelihood
        nll = -np.sum(np.log(y))
        return nll

    @property
    def func_code(self):
        """
        Required by Minuit to identify parameter names.
        """
        class FuncCode:
            def __init__(self, names):
                self.co_varnames = names
                self.co_argcount = len(names)

        return FuncCode(self.param_names)
    

class Chi2Regression:
    def __init__(self, model, x, y, sy):
        """
        Initialize the NLL class.

        Parameters:
        - model: The model function to fit.
        - x_data: Data points (independent variable).
        """
        self.model = model
        self.x = x
        self.y = y
        self.sy = sy
        self.param_names = describe(model)[1:]  # Get parameter names dynamically

    def __call__(self, *params):
        """
        Compute the Negative Log-Likelihood (NLL).

        Parameters:
        - params: Model parameters, dynamically matched to parameter names.

        Returns:
        - Negative log-likelihood value.
        """
        # Map parameter values to their names
        param_dict = dict(zip(self.param_names, params))
        
        # Calculate model predictions
        y = self.model(self.x, **param_dict)
        
        # Check for invalid values in the model output
        if np.any(y <= 0):
            return np.inf  # Return a large value if PDF is invalid
        
        # Compute the negative log-likelihood
        chi2 = np.sum(((self.y - y) / self.sy) ** 2)
        return chi2

    @property
    def func_code(self):
        """
        Required by Minuit to identify parameter names.
        """
        class FuncCode:
            def __init__(self, names):
                self.co_varnames = names
                self.co_argcount = len(names)

        return FuncCode(self.param_names)

def AIC(nll, k):
    """
    Compute the Akaike Information Criterion (AIC).

    Parameters:
    - nll: The negative log-likelihood value.
    - n: Number of data points.
    - k: Number of model parameters.

    Returns:
    - AIC value.
    """
    aic = 2 * k + 2 * nll
    return aic


def bias_corrected_AIC(nll, n, k):
    """
    Compute the bias-corrected Akaike Information Criterion (AICc).

    Parameters:
    - nll: The negative log-likelihood value.
    - n: Number of data points.
    - k: Number of model parameters.

    Returns:
    - AICc value.
    """
    aic = 2 * k + 2 * nll
    aicc = aic + 2 * k * (k + 1) / (n - k - 1)
    return aicc

#######################################
# Define weight factors function      #
#######################################

def weights_twomodels(x, xmin):
    return 1/(1 + np.exp(0.5 * (x-xmin)))


#######################################################
# Define function to select data from a specific file #
#######################################################

def select_data(dataframe, network, parameter, scalefactor):
    """
    Select data from a dataframe containing delay time differences.
    """

    data = dataframe[(dataframe['two_networks'] == network)\
                    & (dataframe['parameter'] == parameter)\
                    & (dataframe['rel change'] == scalefactor)]['time diff'].values
    
    return data


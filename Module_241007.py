import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import root, fsolve
import sympy as sb
import math
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

import numpy as np

# Equations and parametes
par0 = {
    'basal_N':1,
    'basal_G':1,# 0.5/20,
    'basal_E': 1, # E parameter to optimise
    'wf_G' : 4,
    'wf_N' : 4,
    'wf_E' : 12, # E parameter to optimise
    'tau_G' : 1,
    'tau_N' : 1,
    'tau_E' : 1,
    'K_GG' : 1.2,
    'K_NN' : 1.2,
    'K_GN' : 1.2,
    'K_NG' : 1.2,
    'K_FN' : 1, #1
    'K_FE' : 3, #3 # E parameter to optimise
    'K_EN' : 1.2, # E parameter to optimise
    'K_NE' : 1.2, # E parameter to optimise
    'K_NEG' : 1.2,
    'K_EG' : 1.2, # E parameter to optimise
    'h' : 4, # could possibly be lowered??
    'FGF' : 0.85, # we will be varying this parameter below.
    'scaleMutualRepression' : 3.5
}

par0909= {
    'basal_N':1,
    'basal_G':1,# 0.5/20,
    'basal_E': 1, # E parameter to optimise
    'basal_O': 1,
    'wf_G' : 4,
    'wf_N' : 4,
    'wf_E' : 12, # E parameter to optimise
    'wf_O' : 12, # doesnt change (G,N)ss if 12 or 4
    'tau_G' : 1,
    'tau_N' : 1,
    'tau_E' : 1,
    'tau_O': 1, # in Indra's thesis tauO = 3.6*tauN,G,E  # for now (03/09/24), keep tauO= 1, naive case .
    'K_GG' : 1.2,
    'K_NN' : 1.2,
    'K_GN' : 1.2,
    'K_NG' : 1.2,
    'K_FN' : 1, #1
    'K_FE' : 3, #3 # E parameter to optimise
    'K_FO' : 3, # doesnt change (G,N)ss if 1 or 3
    'K_EN' : 1.2, # E parameter to optimise
    'K_NE' : 1.2, # E parameter to optimise
    'K_NEG' : 1.2,
    'K_EG' : 1.2, # E parameter to optimise
    'K_OG' :1.2,
    'K_ON' :1.2,
    'K_NO' :1.2,
    'K_OE':1.2,
    'K_EO':1.2,
    'h' : 4, # could possibly be lowered??
    'FGF' : 0.85, # we will be varying this parameter below.
    'scaleMutualRepression' : 3.5
}

def equations_NG_F(t, var, p):
    """
    ODEs for the system with Nanog and Gata6.
    The equations are the same. I've modified the code to only return non-zero elements
    par0
    """
    G = var[0]
    N = var[1]
    
    scaleEar_G = 1
    
    basal_term_N =  p['scaleMutualRepression']*(p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))
    inhibition_N_by_G =  (1/(1+(G/(p['scaleMutualRepression']*\
                                   p['K_GN']))**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    activation_N_by_N_E =  ((N/p['K_NN'])**p['h'])/(1+(N/p['K_NN'])**p['h']+(p['FGF']/p['K_FN'])**p['h'])
    
    dNdt = (basal_term_N\
            + p['scaleMutualRepression']* p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E\
            - N/p['tau_N']) 
   
    inhibition_G_by_N = 1/(1+(N/(p['scaleMutualRepression']*p['K_NG']))**p['h'])
    activation_G_by_G_E = ((G/(scaleEar_G*p['K_GG']))**p['h'])/(1+(G/(scaleEar_G*p['K_GG']))**p['h']) 
    
    dGdt =  (p['basal_G']\
             + p['scaleMutualRepression']*p['wf_G'] * inhibition_G_by_N * scaleEar_G*activation_G_by_G_E\
             - G/p['tau_G'])
    return dGdt, dNdt

def E_free_func(var, p, dim=3):
    """
    Taken from Module_20220302.
    I have changed names from x, y, z to G, N, E for clarification"""
    if dim==3:
        G = var[0] 
        N = var[1]
        E = var[2]
    if dim==2:
        G = var[0] 
        N = var[1]
        E = find_ss_E(var, p)
    
    return E/(1+N/p['K_NEG'])


def equations_NGE_and_F(t, var, p): #we will specify all our equations in one function
    """
    Copied from Module_20220302
    var: gata, nanog, esrrb
    """
    G = var[0] 
    N = var[1]
    E = var[2]
    
    E_free = E_free_func(var,p)
    
    basal_term_N =  p['scaleMutualRepression']*(p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))
    inhibition_N_by_G =  (1/(1+(G/(p['scaleMutualRepression']*p['K_GN']))**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    activation_N_by_N_E =  ((N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h'])/(1+(N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(p['FGF']/p['K_FN'])**p['h'])
    
    dNdt = basal_term_N  + p['scaleMutualRepression']* p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E - N/p['tau_N'] 
   
    inhibition_G_by_N = 1/(1+(N/(p['scaleMutualRepression']*p['K_NG']))**p['h'])
    activation_G_by_G_E = ((G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h'])/(1+(G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']) 
    dGdt =  p['basal_G'] + p['scaleMutualRepression']*p['wf_G'] * inhibition_G_by_N * activation_G_by_G_E  - G/p['tau_G']
    
    basal_term_E =  (p['basal_E']/(1 + (p['FGF']/(p['K_FE']))**p['h']))
    activation_E_by_N = ((N/p['K_NE'])**p['h'])/((1+(N/p['K_NE'])**p['h']+(p['FGF']/p['K_FE'])**p['h'])) 
   
    dEdt = basal_term_E + p['wf_E'] * activation_E_by_N - E/p['tau_E']
    
    return dGdt, dNdt, dEdt

def equations_NGE_and_F_2var(t, var, p, dim=2): #we will specify all our equations in one function
    """
    Copied from Module_20220302, modulated to 2 variable function
    var: gata, nanog, esrrb
    useful in bifurcation diagram using nullcines.
    """
    G = var[0] 
    N = var[1]
    E = find_ss_E(var, p)
    
    E_free = E_free_func(var,p, dim=2)
    
    basal_term_N =  p['scaleMutualRepression']*(p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))
    inhibition_N_by_G =  (1/(1+(G/(p['scaleMutualRepression']*p['K_GN']))**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    activation_N_by_N_E =  ((N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h'])/(1+(N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(p['FGF']/p['K_FN'])**p['h'])
    
    dNdt = basal_term_N  + p['scaleMutualRepression']* p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E - N/p['tau_N'] 
   
    inhibition_G_by_N = 1/(1+(N/(p['scaleMutualRepression']*p['K_NG']))**p['h'])
    activation_G_by_G_E = ((G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h'])/(1+(G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']) 
    dGdt =  p['basal_G'] + p['scaleMutualRepression']*p['wf_G'] * inhibition_G_by_N * activation_G_by_G_E  - G/p['tau_G']
    
    return dGdt, dNdt


def equations_NGEO_and_F(t, var, p):
    """
    Built upon "equations_NGE_and_F" from Module_20220302.
    Regulatory links from and to OCT4 are copied from Indra (verified from articles/ reviews)
    except auto-regulation of Oct4 to allow an easy formulation of OCT4 steady state.
    var: gata, nanog, esrrb, oct4
    """
    G = var[0]
    N = var[1]
    E = var[2]
    O = var[3]

    E_free = E_free_func(var, p)

    basal_term_N =  p['scaleMutualRepression']*(p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))
    inhibition_N_by_G =  (1/(1+(G/(p['scaleMutualRepression']*p['K_GN']))**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    activation_N_by_N_E =  (((N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h'])\
                            /(1+(N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    
    dNdt = basal_term_N  + p['scaleMutualRepression']* p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E - N/p['tau_N'] 
   
    inhibition_G_by_N = 1/(1+(N/(p['scaleMutualRepression']*p['K_NG']))**p['h'])
    activation_G_by_G_E = (((G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h'])\
                           /(1+(G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h']))
    
    dGdt =  p['basal_G'] + p['scaleMutualRepression']*p['wf_G'] * inhibition_G_by_N * activation_G_by_G_E  - G/p['tau_G']
    
    basal_term_E =  (p['basal_E']/(1 + (p['FGF']/(p['K_FE']))**p['h']))
    activation_E_by_N = ((N/p['K_NE'])**p['h'])/((1+(N/p['K_NE'])**p['h']+(p['FGF']/p['K_FE'])**p['h'])) 
   
    dEdt = basal_term_E + p['wf_E'] * activation_E_by_N - E/p['tau_E']

    basal_term_O = p['basal_O']
    activation_O_by_N_F = (((p['FGF']/p['K_FO'])**p['h']+(N/p['K_NO'])**p['h'])\
                           /(1+(p['FGF']/p['K_FO'])**p['h']+(N/p['K_NO'])**p['h']))
    
    dOdt = basal_term_O + p['wf_O']*activation_O_by_N_F - O/p['tau_O']
    
    return dGdt, dNdt, dEdt, dOdt


def find_ss_O(var, p):
    '''
    dOdt = 0.
    
    Used when defining ICs and plotting phase portraits for the NGEO Network.

    '''
    G = var[0] # unused variable here. However, nice when implemented in a modified version of "equations_NGEO_and_F" where I use Ess and Oss
               # to plot phase plots.
    N = var[1]
    basal_term_O = p['basal_O']
    activation_O_by_N_F = (((p['FGF']/p['K_FO'])**p['h']+(N/p['K_NO'])**p['h'])\
                           /(1+(p['FGF']/p['K_FO'])**p['h']+(N/p['K_NO'])**p['h']))
    
    OSS = (basal_term_O + p['wf_O'])*p['tau_O']

    return OSS


def find_ss_E(var, p):
    '''
    Copied from Module_20220302
    Input: x = Gata6, y = Nanog, p = dict, parametes
    Returns: float, steady state value of esrrb
    '''
    G = var[0]
    N = var[1]
    
    basal_term_E =  (p['basal_E']/(1 + (p['FGF']/(p['K_FE']))**p['h']))
    activation_E_by_N = ((N/p['K_NE'])**p['h'])/((1+(N/p['K_NE'])**p['h']+(p['FGF']/p['K_FE'])**p['h'])) 
    
    EF = (basal_term_E + p['wf_E'] * activation_E_by_N) * p['tau_E'] 
    return EF

def equations_NGEOF_and_maOE(t, var, p):
    """
    Built upon "equations_NGE_and_F" from Module_20220302.
    Regulatory links from and to OCT4 are copied from Indra (verified from articles/ reviews)
    except auto-regulation of Oct4 to allow an easy formulation of OCT4 steady state.
    09/09: added mutual activation between Esrrb and Oct4 - Teresa, Ala
    var: gata, nanog, esrrb, oct4
    """
    G = var[0]
    N = var[1]
    E = var[2]
    O = var[3]

    E_free = E_free_func(var, p)

    basal_term_N =  p['scaleMutualRepression']*(p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))
    inhibition_N_by_G =  (1/(1+(G/(p['scaleMutualRepression']*p['K_GN']))**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    activation_N_by_N_E =  (((N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h'])\
                            /(1+(N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    
    dNdt = basal_term_N  + p['scaleMutualRepression']* p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E - N/p['tau_N'] 
   
    inhibition_G_by_N = 1/(1+(N/(p['scaleMutualRepression']*p['K_NG']))**p['h'])
    activation_G_by_G_E = (((G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h'])\
                           /(1+(G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h']))
    
    dGdt =  p['basal_G'] + p['scaleMutualRepression']*p['wf_G'] * inhibition_G_by_N * activation_G_by_G_E  - G/p['tau_G']
    
    basal_term_E =  (p['basal_E']/(1 + (p['FGF']/(p['K_FE']))**p['h']))

    activation_E_by_N = (((N/p['K_NE'])**p['h']+(O/p['K_OE'])**p['h'])/\
                         ((1+(N/p['K_NE'])**p['h']+(O/p['K_OE'])**p['h']+(p['FGF']/p['K_FE'])**p['h'])))
    
    dEdt = basal_term_E + p['wf_E'] * activation_E_by_N - E/p['tau_E']

    basal_term_O = p['basal_O']
    activation_O_by_N_F = (((p['FGF']/p['K_FO'])**p['h'] + (N/p['K_NO'])**p['h'] + (E/p['K_EO'])**p['h'])\
                           /(1+(p['FGF']/p['K_FO'])**p['h']+(N/p['K_NO'])**p['h'] + (E/p['K_EO'])**p['h']))
    
    dOdt = basal_term_O + p['wf_O']*activation_O_by_N_F - O/p['tau_O']
    
    return dGdt, dNdt, dEdt, dOdt

def find_ss_O_maEO(var,p):
    G = var[0]
    N = var[1]
    E = 12.974379541188263

    basal_term_O = p['basal_O']
    activation_O_by_N_F = (((p['FGF']/p['K_FO'])**p['h'] + (N/p['K_NO'])**p['h'] + (E/p['K_EO'])**p['h'])\
                           /(1+(p['FGF']/p['K_FO'])**p['h']+(N/p['K_NO'])**p['h'] + (E/p['K_EO'])**p['h']))
    
    Oss = p['tau_O']*(basal_term_O + p['wf_O']*activation_O_by_N_F)

    return Oss


def find_ss_E_maEO(var, p):
    '''
    intended to use in "equations_GNEOF_maEO_2var(t, var, p)" and when generating ICs. (need to make sure i use the correct find_ss)

    '''
    G = var[0]
    N = var[1]
    O = 12.999586968218672
    basal_term_E =  (p['basal_E']/(1 + (p['FGF']/(p['K_FE']))**p['h']))

    activation_E_by_N = (((N/p['K_NE'])**p['h']+(O/p['K_OE'])**p['h'])/\
                         ((1+(N/p['K_NE'])**p['h']+(O/p['K_OE'])**p['h']+(p['FGF']/p['K_FE'])**p['h'])))
    
    Ess = p['tau_E']*(basal_term_E + p['wf_E']*activation_E_by_N) 
    return Ess 

def equations_GNEOF_maEO_2var(t, var, p):
    '''
    USELESS !! thought I could use it when plotting the bifurcation diagram (nullcline base). The problem is, Oss and Ess are both functions of E and O,
    respectively. This results in recursion error. Adding a try/except loop does not solve the problem. The code runs forever (more than 40 min.)
    So, I will make a bifurcation diagram using solve_ivp....:,(
    '''

    G = var[0]
    N = var[1]
    E = 12.974379541188263*np.ones(np.shape(var[0]))
    O = 12.999586968218672*np.ones(np.shape(var[0]))
    

    E_free = E_free_func(np.array([G,N,E]), p)

    basal_term_N =  p['scaleMutualRepression']*(p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))
    inhibition_N_by_G =  (1/(1+(G/(p['scaleMutualRepression']*p['K_GN']))**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    activation_N_by_N_E =  (((N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h'])\
                            /(1+(N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    
    dNdt = basal_term_N  + p['scaleMutualRepression']* p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E - N/p['tau_N'] 
   
    inhibition_G_by_N = 1/(1+(N/(p['scaleMutualRepression']*p['K_NG']))**p['h'])
    activation_G_by_G_E = (((G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h'])\
                           /(1+(G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h']))
    
    dGdt =  p['basal_G'] + p['scaleMutualRepression']*p['wf_G'] * inhibition_G_by_N * activation_G_by_G_E  - G/p['tau_G']

    return dGdt, dNdt



def equations_NGEO_and_F_2var(t, var, p):
    """
    Built upon "equations_NGE_and_F" from Module_20220302.
    Regulatory links from and to OCT4 are copied from Indra (verified from articles/ reviews)
    except auto-regulation of Oct4 to allow an easy formulation of OCT4 steady state.
    var: gata, nanog, esrrb, oct4
    """
    G = var[0]
    N = var[1]
    E = find_ss_E(var,p)
    O = find_ss_O(var,p)

    E_free = E_free_func(np.array([G,N,E]), p)

    basal_term_N =  p['scaleMutualRepression']*(p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))

    inhibition_N_by_G =  (1/(1+(G/(p['scaleMutualRepression']*p['K_GN']))**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    
    #print("var values G", G,"N", N, "E",E,"O", O)

    activation_N_by_N_E =  (((N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h'])\
                            /(1+(N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    
    dNdt = basal_term_N  + p['scaleMutualRepression']* p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E - N/p['tau_N'] 
   
    inhibition_G_by_N = 1/(1+(N/(p['scaleMutualRepression']*p['K_NG']))**p['h'])
    activation_G_by_G_E = (((G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h'])\
                           /(1+(G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h']))
    
    dGdt =  p['basal_G'] + p['scaleMutualRepression']*p['wf_G'] * inhibition_G_by_N * activation_G_by_G_E  - G/p['tau_G']
    
    
    return dGdt, dNdt


# INITIAL CONDITIONS

def generate_ic_to_scan_deterministic_triangle(p, ic_range = [0.1,50], N_points_1D=10, base = 2, dim=4):
    '''
    Note: If N_points=34, it returns a (1060, 4) array with IC picked on a grid.
    x, y, z, w = gata, nanog, esrrb, oct4
    esrrb and oct4 are steady state values. they are functions of nanog and gata.
    '''
    
    start_lin = ic_range[0]
    
    end_lin = ic_range[1]

    start_log = np.log2(start_lin)
    
    end_log = np.log2(end_lin)
    
    nx, ny, nz,nw = (N_points_1D, N_points_1D, N_points_1D, N_points_1D)
    
    x = np.logspace(start_log,end_log,nx, base = base)
    y = np.logspace(start_log,end_log,ny, base = base)
    z = 0
    w = 0
    
    def calc_esrrb_and_oct4(dim, xv, yv):
        # default
        z = 0
        w = 0
        
        if dim == 2: # only gata and nanog
            z = 0
            w = 0
        elif dim == 3: # gata, nanog, esrrb
            z = find_ss_E((xv, yv), p)
            w = 0
        elif dim == 4: # gata, nanog, esrrb, oct4
            z = find_ss_E((xv, yv), p)
            w = find_ss_O((xv, yv), p)
        else:
            print("dimension has to be 2, 3, or 4")
            
        return z, w
    

    xv, yv = np.meshgrid(x, y) # this creates a grid , in the next couple of lines we massage this grdi into the same shape as we used when generated ic randomly
    ic = []
    ic = np.empty((0,4)) #preallocating space, in this case the N_points = nx*ny*nz*nw, so = N_points_1D^3
    count = 0
    for i in range(nx):
        for j in range(ny):
            if yv[i,j] <= - (end_lin/end_lin)*xv[i,j] + end_lin:
                z,w = calc_esrrb_and_oct4(dim,xv[i,j],yv[i,j])
                ic = np.concatenate((ic, np.array([xv[i,j], yv[i,j], z,w]).reshape(1,4)), axis=0)
                count+=1                
    return ic

def generate_random_point_in_triangle(par, min_val, max_val, num, dimension=2, esrrb=True, oct4=True):
    """
    Using barycentric coordinates, the random point is a weighted average of the vertices.
    With the normalisation condition on the weights, ensures that the points are randomly
    distributed within the shape; triangle (2D) or tetrahedron(3D)
    
    For 2D: u1+u2+u3=1 and point = u1*v1 + u2*v2 + u3*v3, where vi are vetix i.
    Edit: 01/09: Instead of normalising the random numbers ensuring they sum to 1, I return 1-u and 1-v if u+v>1.
    Reflecting the numbers rather than normalisation avoids a bias towards the centroid. This was problem with the prior code.

    input:
    min_val: float, minimum variable value
    max_val: float, maximum variable value
    num:     int, number of initial conditions
    dimension: int, 2 or 3
    
    return: (num, dimensions) np.array with coordinates of gata, nanog, esrrb ss
    
    """
    if dimension==2:
        # vertices in 2D (x,y):
        v1 = np.array([min_val, min_val])
        v2 = np.array([max_val, min_val])
        v3 = np.array([min_val, max_val])
        
        points = np.zeros((num, dimension))
        
        for i in range(num):
            # constants
            u1 = np.random.rand()
            u2 = np.random.rand()

            if u1 + u2 > 1:
                u1 = 1 - u1
                u2 = 1 - u2
            
            points[i] = (1 - u1 - u2) * v1 + u1 * v2 + u2 * v3
        
        return points
    
    if dimension==3 and esrrb: # z = esrrb steady state
        # vertices in 2D (x,y):
        v1 = np.array([min_val, min_val])
        v2 = np.array([max_val, min_val])
        v3 = np.array([min_val, max_val])
        
        points = np.zeros((num, dimension))
        
        for i in range(num):
            # constants
            u1 = np.random.rand()
            u2 = np.random.rand()

            if u1 + u2 > 1:
                u1 = 1 - u1
                u2 = 1 - u2
            
            points[i,0] = (1 - u1 - u2) * v1[0] + u1 *v2[0] + u2 * v3[0]
            points[i,1] = (1 - u1 - u2) * v1[1] + u1 *v2[1] + u2 * v3[1]
            points[i,2] = find_ss_E(points[i], par)
        
        return points
    
    if dimension==4 and esrrb and oct4:
        # vertices in 2D (x,y):
        v1 = np.array([min_val, min_val])
        v2 = np.array([max_val, min_val])
        v3 = np.array([min_val, max_val])
        
        points = np.zeros((num, dimension))

        for i in range(num):
            # constants
            u1 = np.random.rand()
            u2 = np.random.rand()

            if u1 + u2 > 1:
                u1 = 1 - u1
                u2 = 1 - u2

            points[i,0] = (1 - u1 - u2) * v1[0] + u1 *v2[0] + u2 * v3[0]
            points[i,1] = (1 - u1 - u2) * v1[1] + u1 *v2[1] + u2 * v3[1]
            points[i,2] = find_ss_E(points[i], par)
            points[i,3] = find_ss_O(points[i], par)
    
def plot_initial_conditions(filename, points_2D=None, points_3D=None, savefig=False):
    """
    plots the initial consitions if present.
    ensure that points_2D/3D has shape (dimension, number of points)
    
    input:
    -------
    filename: string
    points_2D: (num, 2) np.array
    points_3D: (num, 3) np.array
    savefig: boolean, default = False
    """
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    fig.suptitle(f"{np.shape(points_2D)[1]} Initial conditions")
    
    # 2D scatter plot
    if points_2D is not None:
        ax[0].scatter(points_2D[:,0], points_2D[:,1])
        ax[0].set_xlabel('GATA')
        ax[0].set_ylabel('NANOG')

    if points_3D is not None:
        # remove existing
        ax[1].remove()
        # 3D scatter plot
        ax3d = fig.add_subplot(122, projection="3d")
        ax3d.scatter(points_3D[0], points_3D[1], points_3D[2], color='blue', marker='o')
        ax3d.set_xlabel('GATA')
        ax3d.set_ylabel('NANOG')
        ax3d.set_zlabel('ESRRB')

    # adjustments, save, show, close
    plt.tight_layout()
    if savefig:
        plt.savefig(f"{filename}")
    plt.show()
    plt.close()

# BISTABILITY?
def group_points_mean_std(points, radius):
    '''
    The function groups points if they are within a radius.
    Input:
    ------
    points: (n x m) np.array with n the number of points and m the dimension
    radius: float, point within or on the boundary are grouped together.

    Output:
    num_sfp: int, number of groups
    SFP_coord_mean: arr of np.array, mean coordinate of the point within the groups.
    SFP_coord_std: arr of np.array, std of grouped points.

    '''
    def distance(point1, point2):
        return np.sqrt(sum((point1-point2)**2))

    
    n = len(points)
    groupnumber = 1
    labels = np.zeros(n)

    while np.any(labels==0):             # until all points are labelled
        p1 = np.where(labels==0)[0][0]   # first [0] unpacks tuple, 
        labels[p1] = groupnumber         # second [0] returns the index of the first un-labelled point
        for p2 in np.where(labels==0)[0]:
            if distance(points[p1], points[p2]) <= radius:
                labels[p2]= groupnumber
        groupnumber += 1
        
    # number of groups/ stable fixed points.    
    num_sfp = len(np.unique(labels))

    # get the coordinates of the stable fixed points.
    SFP_coord_mean = []
    SFP_coord_std = []
    for group in np.unique(labels):
        sfp_coord_mean = np.mean(points[np.where(labels==group)[0]], axis=0)
        sfp_coord_std = np.std(points[np.where(labels==group)[0]], axis=0)
        
        SFP_coord_mean.append(sfp_coord_mean)
        SFP_coord_std.append(sfp_coord_std)
        
    return num_sfp, np.array(SFP_coord_mean), np.array(SFP_coord_std)



def order_of_stability(model, par, dim=2, max_value=100, min_value=0, tmax=200, radius=3):
    """
    02/09/24 - not used - I figured out that "group_points_mean_std" is fast enough - even with 1000 ICs.
    I wont delete this function though..
    
    Since I am checking for bistability, I only evaluate four ICs placed at each corner in 2D grid.
    I assume that the ss values lie within a square from 0-100 and that ss is reached before tmax=200.
    Returns:
    ss_coord: (num, dim=2) array, coordinates of ss (rounded to 1. dec.) (Gata6, Nanog)
    ss_num: int, number of stable fixed points
    """
    if dim==2:
        ics = np.array([[min_value,min_value],[min_value, max_value],[max_value, min_value],[max_value, max_value]])
    if dim==3:
        ics = np.array([[min_value, min_value, min_value], [max_value, min_value, min_value],[min_value, min_value, max_value],
                        [max_value, min_value, max_value], [min_value, max_value, min_value],[max_value, max_value, min_value],
                        [max_value, max_value, max_value], [min_value, max_value, max_value]])
    steady_states = []
    for ic in ics:
        sol = solve_ivp(model, (0, tmax), ic, args=[par])
        steady_states.append(sol.y[:,-1])

    num_sfp, sfp_coord, sfp_coord_std = group_points_mean_std(points=np.array(steady_states), radius=radius)

    return num_sfp, sfp_coord, sfp_coord_std


# DELAY TIMES

def sample_delay_time(ic, parameters, model, tmax=200):
    """
    Inspired by "sample_time_to_ss_v2" function
    Integrates ODE's specified by "model" and returns the time it takes for the system to get to steady state.
    I've changed the expression for slope, so it divides by the previous time - not the newest time.
    
    Input: ic, parametes, model, tmax=200
    """
    
    sol = solve_ivp(model,(0,tmax), ic, args=[parameters])
    
    def find_time_ss(sol):
        count = 0
        dcount = 5
        slope = 10 # some high number
        ss_threshold = 0.01 # convergence

        # note: in case ss isn't reached within tmax, the time condition is added.
        while sol.t[count] < sol.t[-1] and slope > ss_threshold:
            if count-dcount > 0: # ensure that the previous element exists
                # I measure the relative change in Nanog and Gata6 levels between time point i-5 and i.
                # I use the max relative change as a condition.
                slope = np.max((sol.y[:2, count]-sol.y[:2, count-dcount])/sol.y[:2, count-dcount])
            count += 1
        return count-1, sol.t[count]
    
    count_ss, time_ss = find_time_ss(sol)
    
    return sol.y[:,-1], time_ss

def ss_coord_and_delay_time(points_ic, par, model):
    '''
    points_ic: (N, dim) array.
    Returns: ss_coord_array, time_array
    '''
    ss_coord_lst = []
    time_lst = []
    for ic in points_ic:
        ss_coord, time_ss = sample_delay_time(ic, par, model)
        ss_coord_lst.append(ss_coord)
        time_lst.append(time_ss)
    return np.array(ss_coord_lst), np.array(time_lst)

# DATA SAMPLING

def condition_2(SFP, SFP_ref, threshold=0.1):
    '''
    SFP: (2,dim) array, GNE-F or GNEO-F using updated parameter set.
    SFP_ref: (2,2) array, GN-F ss using reference parameter set.
    Returns True or False depending on condition is fulfilled.
    True if each coordinates haven't changed more 10% relative to reference = (GN-F and using ref par set).
    
    note: return rel_change_coord is not possible when sampling to dataframe
    
    '''
    # Ensure to sort the SFP as EPI, PRE
    SFP =  SFP[SFP[:,0].argsort()][:,:2] # [:,:2] means I keep gata, nanog - cut out esrrb and oct4.
                                         # argsort ensures that I compare epi with epi and gata with gata.
    SFP_ref= SFP_ref[SFP_ref[:,0].argsort()] #SFP_ref only contrains gata, nanog
    
    # Calculate the relative change in coordinates of SFP
    rel_change_coord = (SFP-SFP_ref)/SFP_ref
    
    if np.all(rel_change_coord < threshold):
        return True
    else:
        return False

def euclidean_distance(start_arr, final_arr):
    """
    returns one (N, M) array with the euclidean distances
    between points from two (N, M) arrays, where N is the number of points and M is the coordinate.
    Use to calculate the velocity.
    """
    return np.sqrt(np.sum((final_arr-start_arr)**2, axis=-1))


# PLOT DATA
def show_nine_subplots(data, mean, var, title, title_subplots, filename=None, savefig=False):
    '''
    To show the mean and variance of delay times as a function of one variable parameter.
    
    Input:
    title: string, name of parameter of interest
    title_subplots: list/ tuple/ np.array of int, ensure that it has 9 elements.
    '''
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))  # Adjust figsize as needed
    
    fig.suptitle(f"{title}")
    
    x = np.arange(1,1001,1)
    sigma = np.sqrt(var)
    
    # Flatten the 2D array of axes to iterate easily
    axs = axs.flatten()
    
    # Plot each dataset in a separate subplot
    for i, subplot_title in zip(range(9), title_subplots):
        axs[i].plot(data[i], '.', markersize=6)
        axs[i].axhline(y=mean[i], color='red', label=r"$\mu$")
        axs[i].fill_between(x, mean[i]-sigma[i],mean[i]+sigma[i], color='red', alpha=0.3,
                            label=fr"$\sigma = {sigma[i]}$")
        axs[i].set_title(f'Relative change of parameter = {subplot_title}')
        axs[i].set_xlabel('# measurement')
        axs[i].set_ylabel('Delay time [AU]')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    if savefig:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.close(fig)
    # Show the plot
    plt.show()


# plotting boxplots of velocity and delay time diff.
def melted_velocity_data(dataframe, oct4=True):
    '''
    0509
    Prepare dataframe to plot velocity data.
    The "melted" dataframe splits the coloums named "vGNF", "vGNEF", "vGNEOF"
    into two categories: "Network" and "velocity" enabling me to plot velocity data
    to the corresponding network in a boxplot.
    
    Return: a melted dataframe
    Input: a dataframe
    '''
    if oct4==True:    
        df_melt = dataframe.melt(id_vars=["parameter varied", "rel change"],
                       value_vars = ["vGNF", "vGNEF", "vGNEOF"],
                       var_name="Network",
                       value_name="velocity")
    elif oct4==False:
        df_melt = dataframe.melt(id_vars=["parameter varied", "rel change"],
                       value_vars = ["vGNF", "vGNEF"],
                       var_name="Network",
                       value_name="velocity")
        
    return df_melt


def melted_velocity_data_oct4(dataframe):
    '''
    0509
    Prepare dataframe to plot velocity data.
    The "melted" dataframe splits the coloums named "vGNF", "vGNEF", "vGNEOF"
    into two categories: "Network" and "velocity" enabling me to plot velocity data
    to the corresponding network in a boxplot.
    
    Return: a melted dataframe
    Input: a dataframe
    '''
        
    df_melt = dataframe.melt(id_vars=["parameter varied", "rel change"],
                       value_vars = ["vGNEO", "vGNEO2"],
                       var_name="Network",
                       value_name="velocity")
        
    return df_melt    

def melted_timediff_data(dataframe, oct4=True):
    
    """
    0509
    Prepare dataframe to plot timediff data.
    The "melted" dataframe splits the coloums named "deltat GNE-GN", "deltat GNEO-GN"
    into two categories: "Network" and "timediff" enabling me to plot delay time difference data
    to the corresponding network in a boxplot.
    
    Return: a melted dataframe
    Input: a dataframe
    """
    if oct4:
        df_melt = dataframe.melt(id_vars=["parameter varied", "rel change"],
                       value_vars = ["deltat GNE-GN", "deltat GNEO-GN"],
                       var_name="Network",
                       value_name="timediff")
        
    else:
        df_melt = dataframe.melt(id_vars=["parameter varied", "rel change"],
                       value_vars = ["deltat GNE-GN"],
                       var_name="Network",
                       value_name="timediff")

    return df_melt


def melted_timediff_data_oct4(dataframe):
    
    """
    0509
    Prepare dataframe to plot timediff data.
    The "melted" dataframe splits the coloums named "deltat GNE-GN", "deltat GNEO-GN"
    into two categories: "Network" and "timediff" enabling me to plot delay time difference data
    to the corresponding network in a boxplot.
    
    Return: a melted dataframe
    Input: a dataframe
    """
    df_melt = dataframe.melt(id_vars=["parameter varied", "rel change"],
                       value_vars = ["deltat GNEO-GN", "deltat GNEO2-GN"],
                       var_name="Network",
                       value_name="timediff")
    return df_melt
    


def filter_params(df_melted, param):
        '''
        0509
        returns filtered (melted) dataframe regaring caterory "parameter varied".
        Meaning, the dataframe has to contain a coloum with "parameter varied".
        '''
        return df_melted[df_melted["parameter varied"]==param]


    
def plot_indicate_missing_conditions(df_all_melted,
                                     df_filtered,
                                     condition,
                                     ax,
                                     category,
                                     shift,
                                     boxwidth=0.3):
        '''
        Date:05/09
        Input:
        -------
        df_all_melted: pd dataframe. It contains all dataframes (all networks and parameter varied.)
                       melted means that the coloum containing velocity or time delay diff data is splitted
                       into two columns: "Network" and "velocity" OR "timediff"
        df_filtered:   pd dataframe, A filtered melted dataframe. It is filtered regarding if
                       the two conditions are met: 1. bi-stability 2. similar SFP to reference SFP.
        
        condition:     string, "not bistable" OR "disagree SFPref".
                       Non-bistable systems are indicated with shaded grey area.
                       Disagreement with SFP coord is indicated with added stribes.
        ax:            ax, name of the axis defined in the plt.subplots()
        category:      string, "velocity" OR "timediff"
        boxwidth:      float, width of shaded area. For three boxplots 0.3 is a good value
        shift:         float, position of shaded area. For three boxplots, +- 0.3 is a good value.
        
        Output:
        -------
        A plot, indicates boxplots where the system is bi-stable and/ or disagrees with ref. SFPs
        '''
        # Define the correct order of networks
        if category == "velocity":
            correct_order_of_networks = ["vGNF", "vGNEF", "vGNEOF", "vGNEO", "vGNEO2"]
            
        if category == "timediff":
            correct_order_of_networks = ["deltat GNE-GN", "deltat GNEO-GN", 'deltat GNEO2-GN']
            
        df_filtered.loc[:, "Network"] = pd.Categorical(df_filtered["Network"],
                                                           categories= correct_order_of_networks,
                                                           ordered=True)
        
        sorted_unique_networks = df_filtered.loc[:, "Network"].unique()
        
        
        if not df_filtered.empty:
            
            # Get current y-limits
            ymax = np.max(df_all_melted[category]) + 0.5
            ymin = np.min(df_all_melted[category]) - 0.5
            
            # Loop through each catergory in the filtered data
            for network in sorted_unique_networks:
                
                df_network = df_filtered[df_filtered["Network"] == network]
                
                print("network", network) # for debugging purposes
                
                for factor in df_network.sort_values("rel change")["rel change"].unique():
                    
                    # Get the position of the boxplot for the specific factor
                    positions = df_all_melted.sort_values("rel change")["rel change"].unique().tolist() # THIS LIINE!!
                    box_position = positions.index(factor) + shift
                    
                    # Calculate the box width (assuming evenly spaced factors)
                    box_width = boxwidth
                    
                    if condition == "not bistable":
                    
                        # Shade the area behind the boxplot
                        ax.fill_betweenx([ymin ,ymax],
                                         box_position - box_width/2,
                                         box_position + box_width/2,
                                         color="gray", alpha=0.2)
                    
                    if condition == "disagree SFPref":
                        
                        # Striped Shade the area behind the boxplot
                        ax.bar(box_position,
                               ymax - ymin,
                               width=box_width/2, 
                               bottom=ymin,
                               color='none',
                               edgecolor='gray',
                               hatch='//', linewidth=0, alpha=0.3)


# Bifurcation diagrams
                        

def func_nullclines(model_2D, par, t=0, x_start=0, x_stop=50, y_start=0, y_stop=50, curve_step=100):
    """
    29/07/24
    Last edited 31/07/24: added the if condition in case model_2D gives None. Added in context of implementing deMott 2var model.
    created with the purpose of creating func_bifurcation_diagram_one_par
    """
    # variable grid
    curve_x, curve_y = np.meshgrid(np.linspace(x_start, x_stop, curve_step),
                                   np.linspace(y_start, y_stop, curve_step))
    curve_var = curve_x, curve_y
    
    # nullclines
    fignull = plt.figure()
    
    model_value = model_2D(t, curve_var, par)
    if np.all(model_value) == None: # checks if all values are True. "None" is considered a False value. 
        return None                 # So, if the statement is True, not all elements are True aka there exists one or more elements equal to None.

    x_nullcline = plt.contour(curve_x, curve_y, model_value[0],levels=[0])
    y_nullcline = plt.contour(curve_x, curve_y, model_value[1],levels=[0])
    plt.close(fignull) # do not show plot
    
    return x_nullcline, y_nullcline

def func_group_ss_to_sfp(points, radius):
            '''
            The function groups points if they are within a radius.
            Input:
            ------
            points: (n x m) np.array with n the number of points and m the dimension
            radius: float, point within or on the boundary are grouped together.
        
            Output:
            num_sfp: int, number of groups
            SFP_coord: np.array, mean coordinate of the point within the groups.
        
            '''
            
            
            def distance(point1, point2):
                return np.sqrt(sum((point1-point2)**2))

            
            n = len(points)
            groupnumber = 1
            labels = np.zeros(n)

            while np.any(labels==0):
                p1 = np.where(labels==0)[0][0]
                labels[p1] = groupnumber
                for p2 in np.where(labels==0)[0]:
                    if distance(points[p1], points[p2]) <= radius:
                        labels[p2]= groupnumber
                groupnumber += 1
                
            # number of groups/ stable fixed points.    
            num_sfp = len(np.unique(labels))

            # get the coordinates of the stable fixed points.
            SFP_coord = []
            for group in np.unique(labels):
                sfp_coord = np.mean(points[np.where(labels==group)[0]], axis=0)
                SFP_coord.append(sfp_coord)
                
            return num_sfp, SFP_coord

def func_line_intersection(p1, p2, p3, p4):
    """
    29/07/24
    Find intercection point between two parametric line segments in 2D:
    
    r1(t) = p1 + t(p2-p1)
    r2(u) = p3 + u(p4-p3)
    
    Finding the intersection point by solving a set of linear equations using Cramer's rule :
    
    t(x2-x1) - u(x4-x3) = x3-x1
    t(y2-y1) - u(y4-y3) = y3-y1
    
    Input:
    p1, p2, p3, p4: 1D np.arrays with each two entries (x,y)
    
    Output:
    np.array with intersection points
    """
    # define direction vectors
    d1 = p2-p1
    d2 = p4-p3 
    
    # ensure that the lines are not parallel
    detA = d1[1]*d2[0]-d1[0]*d2[1]
    if detA == 0:
        return None # Parallel lines
    
    # use Cramer's rule to find t and u
    t = ((p3[1]-p1[1])*d2[0]-(p3[0]-p1[0])*d2[1])/detA
    u = (d1[0]*(p3[1]-p1[1])-d1[1]*(p3[0]-p1[0]))/detA
    
    if 0<=t<=1 and 0<=u<=1: # ensure the intersection is within the line segments
        ix = p1[0]+t*d1[0]
        iy = p1[1]+t*d1[1]
        return np.array([ix, iy])
    else:
        return None
        p2 = xnullcline_concat[i+1]

def func_find_fixed_points_from_nullclines2907(x_nullcline, y_nullcline, radius):
    """
    Finding fixed points using the nullclines (plt.contour). 
    PRO:
        Do not need to depend on solve_ivp
        wondering if I have run the system for a sufficient amount of time
        Finding all the fixed points instead of just the stable fixed points.
        Hence, I can now make a more comprehensive bifurcation diagram.
          
    Input: 
        x_nullcline : plt.contour
        y_nullcline : plt.contour
        epsil       : float, finding crossing points
        radius      : float, grouping fixed points
    Output:
        num_fixed_pints: int, number of fixed points
        fixed_points_coord: list of arrays, shape = (num_fixed_points, 2)
    """
    # CONCATENATE THE SEGMENTS FOR EACH NULLCLINE
    level = 0 # I only have one level (didt = 0) # contour.allsegs[level][segment][x=0 and y=1]

    xnullcline_concat = np.empty((0,2), int)
    ynullcline_concat = np.empty((0,2), int)

    for (xsegment, ysegment) in zip(x_nullcline.allsegs[level],y_nullcline.allsegs[level]):
        xnullcline_concat = np.concatenate((xnullcline_concat, xsegment), axis=0)
        ynullcline_concat = np.concatenate((ynullcline_concat, ysegment), axis=0)

    crossings=[]
    
    #FIND CROSSING POINTS (fixed points) by checking each segment.
    for i in range(len(xnullcline_concat)-1):
        p1 = xnullcline_concat[i]
        p2 = xnullcline_concat[i+1]
        for j in range(len(ynullcline_concat)-1):
            p3 = ynullcline_concat[j]
            p4 = ynullcline_concat[j+1]
            intersection = func_line_intersection(p1,p2,p3,p4)
            if intersection is not None:
                crossings.append(intersection)

    num_fixed_points, fixed_points_coord = func_group_ss_to_sfp(np.array(crossings), radius)

    return num_fixed_points, np.array(fixed_points_coord)

def func_stability_analysis(model_2D, fixed_points_coord, par):
    """
    Returns the stable fixed points
    
    Note:
        needed packages: sympy as sb, numpy
    Input:
        model_2D: function that takes t, var, par (dict) and 
                  returns np.array([dxdt, dydt])
        fixed_points_coord: list of np.arrays
    Output:
        stable_points: list of np.arrays
    
    """
    # Define variables as symbols
    x, y = sb.symbols("x y")
    sys_eq = sb.Matrix(model_2D(0, (x,y), par))

    # Compute Jacobian
    dxdt = sys_eq[0]
    dydt = sys_eq[1]
    J = sb.Matrix([[sb.diff(dxdt, x), sb.diff(dxdt, y)],
                   [sb.diff(dydt, x), sb.diff(dydt, y)]])


    # stability analysis
    stable_points = []
    for fp in fixed_points_coord:
        eigenvalues = J.subs([(x,fp[0]), (y, fp[1])]).eigenvals()
        if all(ev.as_real_imag()[0] < 0 for ev in eigenvalues):
            stable_points.append(fp)
        
    return np.array(stable_points)

def func_bifurcation_diagram_one_par(model_2D, par, key,filename,
                                par_bifur_start = 0.0,
                                par_bifur_stop = 5.0,
                                par_bifur_stepsize=0.1,
                                par_bifur_log2=True,
                                par_bifur_step = 50,
                                commontitle="Bifurcation diagram",
                                xlabel="Fgf/ ERK",
                                x_ticks=None,
                                left_ylabel="NANOG",
                                right_ylabel="GATA6",
                                font_size=16, 
                                save_fig=False,
                                t=0,x_start=0,x_stop=50,y_start=0,y_stop=50,
                                curve_step=100,
                                radius=4, 
                                markersize=20,
                                stablefixedpoints=True,
                                return_bistable=False,
                                return_tristable=False,
                                return_quadstable=False,
                                fig_global = None,
                                axs_global = None, 
                                num_ax=2):
    """
    29/07/24
    Last edited 31/07/24: Added condition if one of the nullclines==None, the function should continue to the next bifurcation parameter.
    Last edited 09/09/24: Added log2 space of bifurcation parameter, if par_bifur_log2=True. Remember, par_bifur_start > 0.
                          Added colors on y-axis labels

    Input:
    model_2D: function, evolution equations with output np.array([dxdt, dydt])
    par: dict, parameter values
    key: string, name of key in par. the bifurcation parameter.
    filename: string, name of the saved file
    radius: float, set after at smallest distance between fixed points
    + more default settings.

    Output:
    Bifurcation diagram. one parameter, two variables.
    If the model returns dxdt, dydt, the bifurcation diagram will plot x-values on the left plot and y-values on the right plot.
    """
    # create figure
    if fig_global is not None and axs_global is not None:
        fig, axs = fig_global, axs_global
    else:
        fig, axs = plt.subplots(1,num_ax, figsize=(12,6))
    
    fig.suptitle(commontitle, fontsize=font_size)
    
    for ax in axs: ax.set_xlabel(xlabel, fontsize=font_size)
    axs[0].set_ylabel(left_ylabel, fontsize=font_size, color="red")
    axs[1].set_ylabel(right_ylabel, fontsize=font_size, color="green")

    par_bifur_arr = np.arange(par_bifur_start, par_bifur_stop + par_bifur_stepsize, par_bifur_stepsize)

    if par_bifur_log2:
        par_bifur_arr = np.logspace(np.log2(par_bifur_start), np.log2(par_bifur_stop), base=2, num=par_bifur_step)


    bi_stable=[]
    tri_stable=[]
    quad_stable=[]
    for par_bifuri in par_bifur_arr:
        
        par[key]=par_bifuri
        # fixed points
        xnull, ynull=func_nullclines(model_2D, par, t=t, x_start=x_start, x_stop=x_stop,
                                     y_start=y_start, y_stop=y_stop, curve_step=curve_step)
        
        if xnull==None or ynull==None:
            continue # go to the next value in par_bifur_arr
        
        num_fp, coord_fp = func_find_fixed_points_from_nullclines2907(xnull, ynull, radius=radius)
        
        #print("fixed points coordinates", coord_fp)
        
        if len(coord_fp) !=0:
            axs[0].scatter(par[key]*np.ones(len(coord_fp[:,0])), coord_fp[:,0], facecolors="none", edgecolor="red", s=markersize)
            axs[1].scatter(par[key]*np.ones(len(coord_fp[:,1])), coord_fp[:,1], facecolors="none", edgecolor="green", s=markersize)   
        
        # stable fixed points
        if stablefixedpoints:
            coord_sfp=func_stability_analysis(model_2D, coord_fp, par)
            if len(coord_sfp) !=0:
                axs[0].scatter(par[key]*np.ones(len(coord_sfp[:,0])), coord_sfp[:,0], facecolors="red", edgecolor="red", s=markersize)
                axs[1].scatter(par[key]*np.ones(len(coord_sfp[:,1])), coord_sfp[:,1], facecolors="green", edgecolor="green", s=markersize)

            # order of stability
            if len(coord_sfp)==2:
                bi_stable.append(par_bifuri)
            if len(coord_sfp)==3:
                tri_stable.append(par_bifuri)
            if len(coord_sfp)==4:
                quad_stable.append(par_bifuri)
    
    # bifurcation points. I assume the order of stability is the same for both variables.
    if len(bi_stable) != 0:
        bistable=np.array([bi_stable[0],bi_stable[-1]])
    if len(tri_stable) !=0:
        tristable=np.array([tri_stable[0], tri_stable[-1]])
    if len(quad_stable) !=0:
        quadstable=np.array([quad_stable[0]], quad_stable[-1])
    
    # plot order of stability region of interest.
    if return_bistable:
        axs[0].axvspan(bi_stable[0],bi_stable[-1], alpha=0.2)
        axs[1].axvspan(bi_stable[0],bi_stable[-1], alpha=0.2)
    if return_tristable:
        axs[0].axvspan(tri_stable[0],tri_stable[-1], alpha=0.2)
        axs[1].axvspan(tri_stable[0],tri_stable[-1], alpha=0.2)
    if quad_stable:
        axs[0].axvspan(quad_stable[0],quad_stable[-1], alpha=0.2)
        axs[1].axvspan(quad_stable[0],quad_stable[-1], alpha=0.2)
    
    
    # figure adjustments + saving
    ymax = max([x_stop, y_stop])
    ymin = min([x_start, y_start])
    axs[0].set_ylim(ymin, ymax)
    axs[1].set_ylim(ymin, ymax)


    if par_bifur_log2:
        axs[0].set_xscale("log")
        axs[1].set_xscale("log")
    if x_ticks:
        # Specify the custom tick positions
        axs[0].set_xticks(x_ticks)
        axs[1].set_xticks(x_ticks)
        # Set custom tick labels to match the tick positions
        axs[0].set_xticklabels(x_ticks)
        axs[1].set_xticklabels(x_ticks)




    plt.tight_layout()
    if save_fig:
        fig.savefig(f"{filename}.png", dpi=600)
    
    #returns - always put returns at the end of the function.
    if return_bistable:
        return bistable
    if return_tristable:
        return tri_stable
    if return_quadstable:
        return quadstable

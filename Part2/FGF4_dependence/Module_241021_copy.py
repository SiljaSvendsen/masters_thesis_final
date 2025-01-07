import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import root, fsolve
# Statistical test
import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
# analytic solutions
import sympy as sb
import math
# reading files
import pandas as pd
import glob
import os
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


import numpy as np

# Parameters edited 21/10/2024 because I forgot to redefine some constants correctly!!
# Equations and parametes

# par0 only for GN and GNE
par0 = {
    'basal_N':3.5,
    'basal_G':1,# 0.5/20,
    'basal_E': 1, # E parameter to optimise
    'wf_G' : 14,
    'wf_N' : 14,
    'wf_E' : 12, # E parameter to optimise
    'tau_G' : 1,
    'tau_N' : 1,
    'tau_E' : 1,
    'K_GG' : 1.2,
    'K_NN' : 1.2,
    'K_GN' : 4.2,
    'K_NG' : 4.2,
    'K_FN' : 1, #1
    'K_FE' : 3, #3 # E parameter to optimise
    'K_EN' : 1.2, # E parameter to optimise
    'K_NE' : 1.2, # E parameter to optimise
    'K_NEG' : 1.2,
    'K_EG' : 1.2, # E parameter to optimise
    'h' : 4, # could possibly be lowered??
    'FGF' : 0.85, # we will be varying this parameter below.
}
# par0909 for GN, GNE, GNEO1 and GNEO2
par0909= {
    'basal_N':3.5,
    'basal_G':1,# 0.5/20,
    'basal_E': 1, # E parameter to optimise
    'basal_O': 1,
    'wf_G' : 14,
    'wf_N' : 14,
    'wf_E' : 12, # E parameter to optimise
    'wf_O' : 12, # doesnt change (G,N)ss if 12 or 4
    'tau_G' : 1,
    'tau_N' : 1,
    'tau_E' : 1,
    'tau_O': 1, # in Indra's thesis tauO = 3.6*tauN,G,E  # for now (03/09/24), keep tauO= 1, naive case .
    'K_GG' : 1.2,
    'K_NN' : 1.2,
    'K_GN' : 4.2,
    'K_NG' : 4.2,
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
}

def equations_NG_F(t, var, p):
    """
    ODEs for the system with Nanog and Gata6.
    The equations are the same. I've modified the code to only return non-zero elements
    par0
    """
    G = var[0]
    N = var[1]
    
    basal_term_N =  (p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))
    inhibition_N_by_G =  (1/(1+(G/p['K_GN'])**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    activation_N_by_N_E =  ((N/p['K_NN'])**p['h'])/(1+(N/p['K_NN'])**p['h']+(p['FGF']/p['K_FN'])**p['h'])
    
    dNdt = (basal_term_N\
            + p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E\
            - N/p['tau_N']) 
   
    inhibition_G_by_N = 1/(1+(N/p['K_NG'])**p['h'])
    activation_G_by_G_E = ((G/p['K_GG'])**p['h'])/(1+(G/p['K_GG'])**p['h']) 
    
    dGdt =  (p['basal_G']\
             + p['wf_G'] * inhibition_G_by_N * activation_G_by_G_E\
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
    
    basal_term_N =  (p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))
    inhibition_N_by_G =  (1/(1+(G/p['K_GN'])**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    activation_N_by_N_E =  ((N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h'])/(1+(N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(p['FGF']/p['K_FN'])**p['h'])
    
    dNdt = basal_term_N  + p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E - N/p['tau_N'] 
   
    inhibition_G_by_N = 1/(1+(N/p['K_NG'])**p['h'])
    activation_G_by_G_E = ((G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h'])/(1+(G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']) 
    dGdt =  p['basal_G'] + p['wf_G'] * inhibition_G_by_N * activation_G_by_G_E  - G/p['tau_G']
    
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
    
    basal_term_N =  (p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))
    inhibition_N_by_G =  (1/(1+(G/p['K_GN'])**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    activation_N_by_N_E =  ((N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h'])/(1+(N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(p['FGF']/p['K_FN'])**p['h'])
    
    dNdt = basal_term_N  + p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E - N/p['tau_N'] 
   
    inhibition_G_by_N = 1/(1+(N/p['K_NG'])**p['h'])
    activation_G_by_G_E = ((G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h'])/(1+(G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']) 
    dGdt =  p['basal_G'] + p['wf_G'] * inhibition_G_by_N * activation_G_by_G_E  - G/p['tau_G']
    
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

    basal_term_N =  (p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))
    inhibition_N_by_G =  (1/(1+(G/p['K_GN'])**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    activation_N_by_N_E =  (((N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h'])\
                            /(1+(N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    
    dNdt = basal_term_N  + p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E - N/p['tau_N'] 
   
    inhibition_G_by_N = 1/(1+(N/p['K_NG'])**p['h'])
    activation_G_by_G_E = (((G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h'])\
                           /(1+(G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h']))
    
    dGdt =  p['basal_G'] + p['wf_G'] * inhibition_G_by_N * activation_G_by_G_E  - G/p['tau_G']
    
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

    basal_term_N =  (p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))
    inhibition_N_by_G =  (1/(1+(G/p['K_GN'])**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    activation_N_by_N_E =  (((N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h'])\
                            /(1+(N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    
    dNdt = basal_term_N  + p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E - N/p['tau_N'] 
   
    inhibition_G_by_N = 1/(1+(N/p['K_NG'])**p['h'])
    activation_G_by_G_E = (((G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h'])\
                           /(1+(G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h']))
    
    dGdt =  p['basal_G'] + p['wf_G'] * inhibition_G_by_N * activation_G_by_G_E  - G/p['tau_G']
    
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

    basal_term_N =  (p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))
    inhibition_N_by_G =  (1/(1+(G/p['K_GN'])**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    activation_N_by_N_E =  (((N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h'])\
                            /(1+(N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    
    dNdt = basal_term_N  + p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E - N/p['tau_N'] 
   
    inhibition_G_by_N = 1/(1+(N/p['K_NG'])**p['h'])
    activation_G_by_G_E = (((G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h'])\
                           /(1+(G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h']))
    
    dGdt =  p['basal_G'] + p['wf_G'] * inhibition_G_by_N * activation_G_by_G_E  - G/p['tau_G']

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

    basal_term_N =  (p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))

    inhibition_N_by_G =  (1/(1+(G/p['K_GN'])**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    
    #print("var values G", G,"N", N, "E",E,"O", O)

    activation_N_by_N_E =  (((N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h'])\
                            /(1+(N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(O/p['K_ON'])**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    
    dNdt = basal_term_N  + p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E - N/p['tau_N'] 
   
    inhibition_G_by_N = 1/(1+(N/p['K_NG'])**p['h'])
    activation_G_by_G_E = (((G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h'])\
                           /(1+(G/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']+(O/p['K_OG'])**p['h']))
    
    dGdt =  p['basal_G'] + p['wf_G'] * inhibition_G_by_N * activation_G_by_G_E  - G/p['tau_G']
    
    
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


# boxplots
def load_data_w_glob(directory_path, file_pattern):
    """
    directory_path: string, directory path
    file_pattern: string, common part of the filenames
    """

    # Combine the directory path with the file pattern
    files = glob.glob(os.path.join(directory_path, file_pattern))

    # Read and concatenate the CSV files
    dataframes = []

    for file in files:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    return pd.concat(dataframes, ignore_index=True)

def velocity_boxplots(dataframe,
                      date,
                      networks=["GN", "GNE"],
                      font_size=30, #titles, axis labels
                      font_size_legend=14,
                      shift=0.2,
                      box_width=0.3,
                      savefig=False
                     ):
    """
    dataframe has to contain the following categories:
    network, rel change, velocity, parameter, condition 1, condition2
    
    the categories have to contain the following values
    network = GN, GNE, GNEO1, GNEO2 (string)
    parameter = ['alphaGN','alphaEO','KaaNG','KmiNG', 'KmiEN',
                 'KNEG','KEG','KOG','KmaON','KmaOE']
    
    input:
    -----
    networks = ["GN", "GNE"] or ["GN", "GNE", "GNEO1", "GNEO2"]
    the restriction of options is due to the legend_elements (legend)
    
    output:
    ------
    grouped boxplots of velocity data.
    
    """
    # 0. settings for the plots
    
    # custom color palette
    custom_palette = {'GN': 'red',
                      'GNE': 'green',
                      'GNEO1': 'blue',
                      'GNEO2':'orange'
                     }
    if networks==["GN", "GNE"]:
        parameters_of_interest=["alphaGN", "alphaEO", "KaaNG","KmiNG", 
                                "KmiEN", "KNEG", "KEG"]
        
        # legends
        legend_elements = [Patch(facecolor="red", edgecolor="red", label="GN-F"),
                       Patch(facecolor="green", edgecolor="green", label="GNE-F"),
                       Patch(facecolor="gray", edgecolor="gray", label=r"Non bi-stable"),
                       Patch(facecolor="white", edgecolor="gray", hatch="//",label=r"Rel. change (SFP) > 20 %")
                      ]
        # figure titles
        figure_titles = [r"$\alpha_G$, $\alpha_N$",
                         r"$\alpha_E$",
                         "$K_{GG}$, $K_{NN}$",
                         "$K_{GN}$, $K_{NG}$",
                         "$K_{EN}$, $K_{NE}$",
                         "$K_{NEG}$",
                         "$K_{EG}$"
                         ]
    
    if networks==["GN", "GNE", "GNEO1", "GNEO2"]:
        parameters_of_interest=['alphaGN','alphaEO','KaaNG',
                                'KmiNG', 'KmiEN','KNEG','KEG',
                                'KOG','KmaON','KmaOE']
        # legends
        legend_elements = [Patch(facecolor="red", edgecolor="red", label="GN-F"),
                           Patch(facecolor="green", edgecolor="green", label="GNE-F"),
                           Patch(facecolor="blue", edgecolor="blue", label="GNEO 1"),
                           Patch(facecolor="orange", edgecolor="orange", label="GNEO 2"),
                           Patch(facecolor="gray", edgecolor="gray", label=r"Non bi-stable"),
                           Patch(facecolor="white", edgecolor="gray", hatch="//",label=r"Rel. Change (SFP) > 20 %")
                          ]
        # figure titles
        figure_titles = [r"$\alpha_G$, $\alpha_N$",
                         r"$\alpha_E$",
                         "$K_{GG}$, $K_{NN}$",
                         "$K_{GN}$, $K_{NG}$",
                         "$K_{EN}$, $K_{NE}$",
                         "$K_{NEG}$",
                         "$K_{EG}$",
                         "$K_{OG}$",
                         "$K_{ON}$, $K_{NO}$",
                         "$K_{OE}$,$K_{EO}$"
                        ]
    
    # 1. prepare dataframe
    velocity_df = dataframe[["network", "rel change", "velocity", "parameter", "condition 1", "condition 2"]]

    # filter for parameters only relevant to networks involved
    mask0 = velocity_df["parameter"].isin(parameters_of_interest)
    mask1 = velocity_df["network"].isin(networks)
    df_masked= velocity_df[mask0&mask1]
    
    # 2. plot figures
    for parameter, title in zip(parameters_of_interest,figure_titles):
        
        # 0. create new figure
        fig, ax = plt.subplots(figsize=(10,6))
    
        # 1. create grouped boxplot
    
        # 1.1 filter for parameter
        mask = df_masked["parameter"]==parameter
    
        # 1.2 ensure the categorical order for 'network' (hue) is explicitly set + sort_values("network")
        df_masked.loc[:,'network'] = pd.Categorical(df_masked['network'], categories=networks, ordered=True)
        df_masked_param=df_masked[mask].sort_values("network")
    
        sns.boxplot(x="rel change",
                    y="velocity",
                    hue="network",
                    hue_order=networks,
                    data=df_masked_param,
                    palette=custom_palette,
                    ax=ax
                   )
    
        # 2. indicate non bi-stability and disagreement with reference sfp
        ymax = np.max(df_masked["velocity"]) + 0.5
        ymin = np.min(df_masked["velocity"]) - 0.5

        dx_arr = np.linspace(-shift, shift, len(networks))

        for network, dx in zip(networks, dx_arr):
            
            df_network = df_masked_param[df_masked_param["network"]==network]
            
            
            # Indicate non-bistability
            # x_pos = rel change values at which condition 1 is not True.
            x_pos=df_network[df_network["condition 1"] != True]["rel change"].unique()
            for x in x_pos:
                # all_pos = all rel change at which the PSA is performed. due to .index, it has to be a list.
                all_pos = np.sort(df_masked["rel change"].unique()).tolist()
                box_position = all_pos.index(x) + dx
                
                # Shade the area behind the boxplot
                ax.fill_betweenx([ymin ,ymax],
                             box_position - box_width/2,
                             box_position + box_width/2,
                             color="gray", alpha=0.2)
                
            # indicate disagreement with reference sfp
            x_pos = df_network[df_network["condition 2"] != True]["rel change"].unique().tolist()
            
            for x in x_pos:
                all_pos = np.sort(df_masked["rel change"].unique()).tolist()
                box_position = all_pos.index(x) + dx
                
                # Striped Shade the area behind the boxplot
                ax.bar(box_position,
                       ymax - ymin,
                       width=box_width, 
                       bottom=ymin,
                       color='none',
                       edgecolor='gray',
                       hatch='//', linewidth=0, alpha=0.3)
    
        # 4. title, axis labels, legends
        ax.set_title(f"{title}", fontsize=font_size)
        ax.set_xlabel("Scale Factor", fontsize=font_size)
        ax.set_ylabel("Velocity", fontsize=font_size)
    
        ax.legend(handles=legend_elements, loc="best", fontsize=font_size_legend)
    
        # 5. adjust y-axis (ymin, ymax, logscale)
        ymax = np.max(velocity_df["velocity"]) + 0.5
        ymin = np.min(velocity_df["velocity"])
    
        if ymin < 0: # avoid trouble with logscale
            ymin=0
            
        ax.set_ylim(ymin, ymax)
        ax.set_yscale("log")
    
        # 6. adjust figure
        plt.tight_layout()
    
        # 7. save figures
        if savefig:
            plt.savefig(f"velocity_boxplots_{networks}_{parameter}_{date}.pdf", dpi=600)

def time_delay_boxplots(dataframe,
                        date="241010",
                        networks=["GNEvGN"],
                        font_size=30, #titles and axis labels,
                        font_size_legend=20,
                        shift=0,
                        box_width=0.3,dy=0,
                        savefig=True):
    """
    dataframe needs to contain the following categories:
    two_networks, parameter, rel change, condition 1, condition 2, time diff.
    
    the categories have to contain following values
    two_networks = GNEvGN, GNEO1vGN, GNEO2vGN
    parameter = ['alphaGN','alphaEO','KaaNG','KmiNG', 'KmiEN','KNEG','KEG','KOG','KmaON','KmaOE']
    
    input:
    ------
    networks=["GNEvGN"] or ['GNEvGN','GNEO1vGN','GNEO2vGN']
    
    output:
    -------
    grouped boxplots of time delay data.
    """
    
    # 0. Settings for the plots
    # Custom color palette
    custom_palette = {'GNEvGN': "green",
                      'GNEO1vGN':"blue",
                      'GNEO2vGN':"orange"}

    if networks==["GNEvGN"]: # if only GN and GNE
        parameters_of_interest=["alphaGN", "alphaEO", "KaaNG",
                                "KmiNG", "KmiEN", "KNEG", "KEG"]

        #legends
        legend_elements = [Patch(facecolor="green", edgecolor="green", label=r"$\Delta\tau_{GNE-GN}$"),
                           Patch(facecolor="gray", edgecolor="gray", label=r"Non bi-stable"),
                           Patch(facecolor="white", edgecolor="gray", hatch="//",label=r"Rel. Change (SFP) > 20 %")
                          ]


        # Costum figure titles
        figure_titles = [r"$\alpha_G$, $\alpha_N$",
                         r"$\alpha_E$",
                         "$K_{GG}$, $K_{NN}$",
                         "$K_{GN}$, $K_{NG}$",
                         "$K_{EN}$, $K_{NE}$",
                         "$K_{NEG}$",
                         "$K_{EG}$"
                        ]

        filename_key = parameters_of_interest

    if networks==['GNEvGN','GNEO1vGN','GNEO2vGN']: # compare all networks to GN
        
        parameters_of_interest=['alphaGN','alphaEO','KaaNG',
                                'KmiNG', 'KmiEN','KNEG','KEG',
                                'KOG','KmaON','KmaOE']

        #legend
        legend_elements = [Patch(facecolor="green", edgecolor="green", label=r"$\Delta\tau_{GNE-GN}$"),
                           Patch(facecolor="blue", edgecolor="blue", label=r"$\Delta\tau_{GNEO1-GN}$"),
                           Patch(facecolor="orange", edgecolor="orange", label=r"$\Delta\tau_{GNEO2-GN}$"),
                           Patch(facecolor="gray", edgecolor="gray", label=r"Non bi-stable"),
                           Patch(facecolor="white", edgecolor="gray", hatch="//",label=r"Rel. Change (SFP) > 20 %")
                          ]


        # Costum figure titles
        figure_titles = [r"$\alpha_G$, $\alpha_N$",
                         r"$\alpha_E$",
                         "$K_{GG}$, $K_{NN}$",
                         "$K_{GN}$, $K_{NG}$",
                         "$K_{EN}$, $K_{NE}$",
                         "$K_{NEG}$",
                         "$K_{EG}$",
                         "$K_{OG}$",
                         "$K_{ON}$, $K_{NO}$",
                         "$K_{OE}$,$K_{EO}$"
                        ]

        filename_key = parameters_of_interest

        
    # 0. prepare dataframe
    #filter for parameters only relevant to the GN and GNE network
    mask0 = dataframe["parameter"].isin(parameters_of_interest)
    mask1 = dataframe["two_networks"].isin(networks)

    df_masked = dataframe[mask0&mask1]
    
    # Plot the figure
    
    for parameter, title in zip(filename_key,figure_titles):
    
    
        # 0. create new figure
        fig, ax = plt.subplots(figsize=(10,6))
    
        # 1.0 clarify if time diff is positive or negative
        ax.axhline(y=0.0, color='gray', linestyle='-', zorder=0)
    
        
        # 1.1 create grouped boxplot
        # 1.2 filter for parameter
        mask = df_masked["parameter"]==parameter
    
        # 1.3 ensure the categorical order for 'network' (hue) is explicitly set + sort_values("network")
        df_masked.loc[:,'two_networks'] = pd.Categorical(df_masked['two_networks'], categories=networks, ordered=True)
        df_masked_param=df_masked[mask].sort_values("two_networks")

        sns.boxplot(x="rel change",
                y="time diff",
                hue="two_networks",
                hue_order=networks,  # enforce order here
                data=df_masked_param,
                palette=custom_palette,
                ax=ax
               )
    
        # 2. indicate non bi-stability and disagreement with reference sfp
        ymax = np.max(df_masked["time diff"]) + 0.5
        ymin = np.min(df_masked["time diff"]) - dy

        dx_arr = np.linspace(-shift, shift, len(networks))

        for network, dx in zip(networks, dx_arr):
        
            df_network = df_masked_param[df_masked_param["two_networks"]==network]
        
        
            # Indicate non-bistability
            # x_pos = rel change values at which condition 1 is not True.
            x_pos=df_network[df_network["condition 1"] == False]["rel change"].unique()
            for x in x_pos:
                # all_pos = all rel change at which the PSA is performed. due to .index, it has to be a list.
                all_pos = np.sort(df_masked["rel change"].unique()).tolist()
                box_position = all_pos.index(x) + dx
            
                # Shade the area behind the boxplot
                ax.fill_betweenx([ymin ,ymax],
                         box_position - box_width/2,
                         box_position + box_width/2,
                         color="gray", alpha=0.2)
            
            # indicate disagreement with reference sfp
            x_pos = df_network[df_network["condition 2"] == False]["rel change"].unique().tolist()
        
            for x in x_pos:
                all_pos = np.sort(df_masked["rel change"].unique()).tolist()
                box_position = all_pos.index(x) + dx
                
                # Striped Shade the area behind the boxplot
                ax.bar(box_position,
                       ymax - ymin,
                       width=box_width, 
                       bottom=ymin,
                       color='none',
                       edgecolor='gray',
                       hatch='//', linewidth=0, alpha=0.3)
    
        # 4. title, axis labels, legends
        ax.set_title(f"{title}", fontsize=font_size)
        ax.set_xlabel("Scale Factor", fontsize=font_size)
        ax.set_ylabel(r"$\Delta\tau$", fontsize=font_size)
    
        ax.legend(handles=legend_elements, loc="lower left", fontsize=font_size_legend)
    
        # 5. adjust y-axis (ymin, ymax)
        ymax = np.max(dataframe["time diff"])
        ymin = np.min(dataframe["time diff"])-dy
        ax.set_ylim(ymin, ymax)
    
        # 6. adjust figure
        plt.tight_layout()
    
        # 7. save figures
        if savefig:
            plt.savefig(f"timedelay_boxplots_{networks}_{parameter}_{date}.pdf", dpi=600)


# Bifurcation diagrams, on sampled data
def bifurcation_diagram_of_sampled_data_relchange(df_network, # non-default
                                                  model,
                                                  dim,
                                                  date, # string, non-default
                                                  parameter, # non-default
                                                  parameter1, # non-default
                                                  par=None, # this will default to None and we use par0.copy()
                                                  n=40, # default, length of bifur_range
                                                  savefile=False
                                                 ):
    """
    par : dict, has to contain "wf_G", "wf_N", "K_NG", and "K_GN"
    parameter = "alphaGN" or "KmiNG"
    parameter1 = "wf_G" or "K_GN"
    """

    # If par is not provided, use a copy of par0 to avoid modifying the original dictionary
    if par is None:
        par = par0.copy()  # Shallow copy of par0 to avoid changing the original object

    ref = par.copy()  # Shallow copy of par to restore during iteration
    refpar = ref[parameter1]
    
    print("Initial refpar:", refpar)  # Debugging print to check initial value
    
    network = df_network["network"].unique()[0]  # For readability

    # Filter for a particular parameter of interest
    df_network_param = df_network[df_network["parameter"] == parameter]

    # Bifurcation range = rel change range but more fine-grained (25 points).
    bifur_range = np.logspace(np.log2(min(df_network_param["rel change"])), 
                              np.log2(max(df_network_param["rel change"])), 
                              num=n, base=2)

    # Remove duplicate IC points, reset index, and go from pd.df to np array to enable slicing
    ics = df_network_param[["ic gata6", "ic nanog", "ic esrrb"]].drop_duplicates().reset_index(drop=True).to_numpy()

    all_data = []
    for bifur in bifur_range:
        # Update bifurcation parameter
        if parameter1 == "wf_G":
            par["wf_N"] = par[parameter1] = bifur * refpar
        if parameter1 == "K_GN":
            par["K_NG"] = par[parameter1] = bifur * refpar

        for ic in ics:
            # Deal with shape of ics...
            if dim == 2:
                ic = ic[:dim]
                esrrb = 0
            if dim == 3:
                ic = ic[:dim]
                esrrb = ic[dim-1]
            
            # Calculate steady state
            ss, time_ss = sample_delay_time(ic=ic, parameters=par, model=model)
            
            # Deal with shape of ss...
            if dim == 2:
                ss_esrrb = 0
            if dim == 3:
                ss_esrrb = ss[2]

            # Sample data
            data = {"ic gata6": ic[0],
                    "ic nanog": ic[1],
                    "ic essrb": esrrb,
                    "ss gata6": ss[0],
                    "ss nanog": ss[1],
                    "ss esrrb": ss_esrrb,
                    "rel change": bifur,
                    "parameter": parameter,
                    "network": network,
                   }
            
            # Append data dictionary to all_data list
            all_data.append(data)
        
        # Reset par to the shallow copy of the original ref dictionary
        par = ref.copy()
        if parameter1=="wf_G":
            par["wf_N"] = par["wf_G"] = par[parameter1]  # Ensure wf_N gets reset as well
        if parameter1=="K_GN":
            par["K_NG"] = par["K_GN"] = par[parameter1]
    
    # From dict to pandas dataframe
    df_all_data = pd.DataFrame(all_data)

    if savefile: # Save the DataFrame to a CSV file
        df_all_data.to_csv(f'{network}_{parameter}_{date}.csv', index=False)  
        print("DONE")


def plot_bifurcation_diagram_gata_nanog_alpha_kmiNG(dataframe,
                                                    filename_figure,
                                                    date,
                                                    font_size_axis_title,
                                                    font_size=14, #default
                                                    ymax=70, # default, but adjust when plotting
                                                    savefig=False,
                                                    pdf=True):
    """
    14.okt.2024
    plots the bifurcation diagram to show how sensitive GN and GNE is to alphaGN and kmiNG regarding
    bistability and robustness in stable fixed points.
    
    Input:
    ------
    dataframe: pandas dataframe, contain categories "rel change", "ss gata6", and "ss nanog"
    filename_figure: string, filename of figure (if savefig=True)
    date: string, part of the filename
    font_size: int, fontsize of the title, axis, and legends
    ymax: int, ymax
    savefig: boolean, if True, the figure is saved as a pdf file, dpi=600
    
    Output:
    ------
    a bifurcation diagram with alphaNG or kmiNG as the bifurcation parameter.
    note: Specify the bifurcation parameter in the filename
    """

    # Annotate the lines
    def annotate_axhline(text, y, ax, x=np.log2(0.12)):
        """
        text: str, annotation text
        x, y: float, position of the annotation
        ax: specify ax (the subplot)
        """
    
        ax.annotate(f"{text}", xy=(0.5, y), xycoords='data', 
                    fontsize=12, color="gray", 
                    xytext=(x, y),  # Position on top of the line
                    textcoords='data',  # Keep the x-coordinate in data coordinates
                    verticalalignment='center',
                    horizontalalignment='center',
                    bbox=dict(facecolor='white', edgecolor='none', pad=2.0))  # White background box
    
    # 0. data
    x = dataframe["rel change"]
    gata6 = dataframe["ss gata6"]
    nanog = dataframe["ss nanog"]

    # NB: I take log2 of both all data and tickslabels to "zoom" in on the region
    # rel change in [0.2, 1.0] to highlight the bifurcation (from mono- to bi-stable)
    # Most importantly; the values on axis and labelling are correct.

    # 1. create figure and plot data
    fig, ax = plt.subplots(figsize=(4.5,6)) 
    ax.plot(np.log2(x), np.log2(gata6), ".",color="red", label="GATA6")
    ax.plot(np.log2(x), np.log2(nanog),".",color="green", label="NANOG")

    # 2. Add reference case
    # The reference Epi and PrE states are at
    # (Gata6, Nanog)=[ 1.10273171 11.46535802] and (Gata6, Nanog)=[13.62042978 2.41356951]

    # 2.1 plot Epi reference
    ax.axhline(y=np.log2(1.10273171),color="gray", linestyle="--", label="Epi")  # gata6
    ax.axhline(y=np.log2(11.46535802), color="gray", linestyle="--") # nanog

    # 2.2 plot PrE refernce
    ax.axhline(y=np.log2(13.62042978), color="gray", linestyle=":", label="PrE") # gata6
    ax.axhline(y=np.log2(2.41356951),color="gray", linestyle=":") # nanog

    # 2.3 add annotations
    for text_i, y_i in zip(["GATA6 ", "NANOG", "GATA6 ", "NANOG"],[1.10273171,11.46535802,13.62042978,2.41356951]):
        annotate_axhline(text=text_i,y=np.log2(y_i), ax=ax)

    #2.4 indicate bi-stability
    stability_lst = []
    bifur_range=np.sort(np.unique(dataframe["rel change"].to_numpy()))
    for s in bifur_range:
        mask = dataframe["rel change"]==s
        stability=len(np.unique(np.round(dataframe[mask]["ss nanog"].to_numpy())))
        stability_lst.append(stability)
    bistable_start = bifur_range[np.where(np.array(stability_lst)==2)[0][0]] # first 0 = unpack tuple, second 0 = get first element
    bistable_stop = bifur_range[np.where(np.array(stability_lst)==2)[0][-1]]
    
    print("bi-stable in scale factor range:",bistable_start, bistable_stop)
    ax.fill_betweenx([0, np.log2(ymax)], np.log2(bistable_start), np.log2(bistable_stop), color="lightblue", alpha=0.2)
    
    # 3. title, axis label, legend
    ax.set_title(f"{dataframe['network'].unique()[0]}", fontsize=font_size_axis_title)
    ax.set_xlabel("Scale Factor", fontsize=font_size_axis_title)
    ax.set_ylabel("Protein Level", fontsize=font_size_axis_title)
    plt.legend(ncol=2, loc="upper left",fontsize=font_size)
    
    # 4. costumise ticks
    ytick_labels = [0.025*ymax, 0.05*ymax, 0.1*ymax,0.25*ymax, 0.5*ymax, ymax ]
    ytick = ax.set_yticks(ticks=np.log2(ytick_labels), labels=ytick_labels)
    xtick_labels = [0.2, 0.25, 0.5, 0.75, 1.0, 1.25,2, 3, 4, 5]
    xtick=ax.set_xticks(ticks=np.log2(xtick_labels), labels=xtick_labels, rotation=45)
    
    # 5. Adjust figure
    ax.set_xlim(np.log2(0.07), np.log2(5))
    plt.tight_layout()

    # 6. save figure
    if savefig and pdf:
        plt.savefig(f"bifurcation_diagram_{filename_figure}_{date}.pdf", dpi=600)
    if savefig and pdf==False:
        plt.savefig(f"bifurcation_diagram_{filename_figure}_{date}.png", dpi=600)


# Bifurcation diagram, general
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
        fig.savefig(f"bifurcation_diagram_fgf4_{filename}.pdf", dpi=600)
    
    #returns - always put returns at the end of the function.
    if return_bistable:
        return bistable
    if return_tristable:
        return tri_stable
    if return_quadstable:
        return quadstable

def func_bifurcation_diagram_steady_state(model,
                                          par,
                                          network,
                                          date,
                                          dim,
                                          ic_range=[0.1,5.0],
                                          ic_num=36,
                                          bstart=0.03,
                                          bstop=2.5,
                                          b_num=50,
                                         ):
    
    '''
    samples data to bifurcation diagrams for "model" as function of fgf4.
    the data is saved in a csv file named "fgf4_dependence_data_{network}_{date}.csv"
    steady state values using solve_ivp.
    '''
    
    def func_steady_state(model, par, ic):
        sol = solve_ivp(model, (0,200), ic, args=[par])
        return sol.y[:,-1]
    
    
    # 0.0 define initial conditions
    ics = generate_ic_to_scan_deterministic_triangle(p=par,
                                                     ic_range=ic_range,
                                                     N_points_1D=ic_num,
                                                     base=2, dim=dim)
    
    ics = ics[:,:dim] # fx. if GN model; evaluate only nanog, gata6 coordinates in solve_ivp 
    
    
    
    # 0.1 define fgf4 range
    fgf4 = np.logspace(np.log2(bstart), np.log2(bstop), num=b_num, base=2)
    
    # 1.0 sample bifurcation data to external csv file.
    all_data = []
    
    for fgf in fgf4:
        
        # 0. update fgf value
        par["FGF"]=fgf
        
        for ic in ics:
            
            # to acommodate that I sample all ics regardless.
            if dim==2:
                esrrb=0
                oct4=0
            if dim==3:
                esrrb=ic[2]
                oct4=0
            if dim==4:
                esrrb=ic[2]
                oct4=ic[3]
            
            
            steady_state = func_steady_state(model=model, par=par, ic=ic)
            
            # to acommodate that I sample all ics regardless.
            if dim==2:
                ss_esrrb=0
                ss_oct4=0
            if dim==3:
                ss_esrrb=steady_state[2]
                ss_oct4=0
            if dim==4:
                ss_esrrb=steady_state[2]
                ss_oct4=steady_state[3]
            
            data = {
                "network": network,
                "ic gata6": ic[0],
                "ic nanog": ic[1],
                "ic esrrb": esrrb,
                "ic oct4": oct4,
                "FGF4": fgf,
                "ss gata6": steady_state[0],
                "ss nanog": steady_state[1],
                "ss esrrb": ss_esrrb,
                "ss oct4": ss_oct4
            }
            
            all_data.append(data)
    bifurcation_data_df = pd.DataFrame(all_data)
    bifurcation_data_df.to_csv(f"fgf4_dependence_data_{network}_{date}.csv", index=False)
    print(f"fgf4 dependece data is saved to fgf4_dependence_data_{network}_{date}.csv")
    
    
def func_plot_fgf4_bifurcation_diagram(dataframe,
                                       model_2D,
                                       par,
                                       dim,
                                       network,
                                       save_fig=False,
                                       return_bistable=True,
                                       font_size=30,
                                       bistable_start=None,
                                       bistable_stop=None):
    '''
    21.10.24
    The function plots the bifurcation diagrams for a network as function of Fgf4.
    The bifurcation diagrams are plotted in two methods.
    
    Method 1: steady state based (dataframe, dim)
    Method 2: nullcline + linear stability analysis based (model_2D, par)
    
    dataframe is evaluated in Method 1. 

    model_2D is evaluated in Method 2. For 3-node and 4-node networks, I've used the steady state solutions
    of the Esrrb and Oct4 evolution equations.
    
    dim:      2, 3 or 4. It determines the number of subplots.
    save_fig: boolean, if True, I save the figure as a pdf file named
              "bifurcation_diagram_fgf4_{network}.pdf"
    network: string, if GNEO2, the bifurcation diagram is only plotted using method 1,
             since the steady state solutions of the Esrrb,f, and Oct4,g, evolution equations depended on
             eachother. f(O) and g(E).
    
    
    '''
    figure_width=3*dim
    fig, axs = plt.subplots(1, dim, sharey=True, sharex=True, figsize=(figure_width, 4))

    x_ticks=[0.5, 1, 2]

    # METHOD 1

    # GATA6
    axs[0].semilogx(dataframe["FGF4"],dataframe["ss gata6"],'*', color="red", markersize = 3, alpha = 0.2)
    axs[0].set_xlabel('[FGF4]', fontsize=font_size)
    axs[0].set_ylabel('[GATA6]', fontsize=font_size, color="red")
    axs[0].set_ylim(-1,18)
    axs[0].set_xlim(0.1,2.6)

    # NANOG
    axs[1].semilogx(dataframe["FGF4"],dataframe["ss nanog"],'*', color="green", markersize = 3, alpha = 0.2)
    axs[1].set_xlabel('[FGF4]', fontsize=font_size)
    axs[1].set_ylabel('[NANOG]', fontsize=font_size, color="green")


    if dim==3:
        
        # ESRRB
        axs[2].semilogx(dataframe["FGF4"], dataframe["ss esrrb"], "*", color="blue", markersize=3, alpha=0.2)
        axs[2].set_xlabel('[FGF4]', fontsize=font_size)
        axs[2].set_ylabel('[ESRRB]', fontsize=font_size, color="blue")
        axs[0].set_xticks(x_ticks)
        axs[0].set_xticklabels(x_ticks)

    if dim==4:
        
        # ESRRB
        axs[2].semilogx(dataframe["FGF4"], dataframe["ss esrrb"], "*", color="blue", markersize=3, alpha=0.2)
        axs[2].set_xlabel('[FGF4]', fontsize=font_size)
        axs[2].set_ylabel('[ESRRB]', fontsize=font_size, color="blue")
        
        # OCT4
        axs[3].semilogx(dataframe["FGF4"], dataframe["ss oct4"], "*", color="orange", markersize=3, alpha=0.2)
        axs[3].set_xlabel('[FGF4]', fontsize=font_size)
        axs[3].set_ylabel('[OCT4]', fontsize=font_size, color="orange")
        axs[0].set_xticks(x_ticks)
        axs[0].set_xticklabels(x_ticks)


    # METHOD 2
    if network!="GNEO2": # Oss and Ess are functions of each other.
        bistable_range=func_bifurcation_diagram_one_par(model_2D=model_2D, par=par,
                                         key="FGF", filename=network,
                                         par_bifur_start=0.3, par_bifur_stop=2.5, par_bifur_step=30,
                                         x_stop=18, y_stop=18,x_ticks = x_ticks,
                                         commontitle=" ",font_size=font_size,
                                         left_ylabel="[GATA6]", right_ylabel="[NANOG]", xlabel="[FGF4]",
                                         save_fig=save_fig, radius=1, return_bistable=return_bistable,
                                         fig_global=fig, axs_global=axs
                                        )
    if network=="GNEO2":
        fig.suptitle(" ", fontsize=font_size)
        axs[0].axvspan(bistable_start,bistable_stop, alpha=0.2)
        axs[1].axvspan(bistable_start,bistable_stop, alpha=0.2)
        plt.tight_layout()
        if save_fig:
            plt.savefig(f"bifurcation_diagram_fgf4_{network}.pdf", dpi=600)
    if return_bistable:
        return bistable_range
# Phase plots
def func_phase_portrait_new(model_2D,
                            par,
                            filename,
                            t = 0,
                            x_start = -5,
                            y_start = -5,
                            x_stop = 50,
                            y_stop = 50,
                            curve_step = 100,
                            vec_step = 30,
                            tmin = 0,
                            tmax = 100,
                            epsil=0.5,
                            radius=20,
                            xmin = None,
                            xmax = None,
                            ymin = None,
                            ymax = None,
                            x_color = "red",
                            y_color = "green",
                            title = "Phase portrait",
                            x_label = "[NANOG]",
                            y_label = "[GATA6]",
                            fontsize_label_title = 16,
                            x_legend = "NANOG nullcline",
                            y_legend = "GATA6 nullcline",
                            fontsize_legend=10,
                            loc_legend="best",
                            ax = None,
                            vectorfield=True,
                            logscale = False,
                            return_nullclines=False,
                            add_legends = False,
                            add_legends_fp=True,
                            find_fixed_points=True,
                            stable_fixed_points_return = True,
                            basin_of_attraction = False,
                            tight_layout=True,
                            save_figure=False):
    '''
    Last edited: 13/05/24
    Last edited: 17/06/24
    Last edited: 18/06/24: enabled logscale
    Last edited: 01/07/24: enabled to go orders lower than 0 if logscale.
    Last edited: 29/07/24: smarter way to find the fixed points.
    The function is tested and built in Jupyter Notebook file "240506_nanog_esrrb_gata_building_up_compariing_tristability"
    The function is updated in Jypyter Notebook file "0617 update plot phase portrait function" (1706)
    The function is updated in Jypyter Notebook file "0617 update plot phase portrait function" (1806)

    Output: A function that plots phase portraits.
    
    NOTE:
    By default, the limits on the axis are the same as the limits on the region that is simulated.
    By default, the steady state is assumed to have been reached at tax=100. I use solve_ivp.
    "bassin_of_attraction_vector_field_plot" and "correct_stable_fixed_points" have to be true before a colorcoded vector
    field is plotted. Else, a basic one is plotted.
    if logscale, the minimum x_start and y_start values is 1.

    Input:
    ------
    model_2D  :    function, evolution equation that outputs np.array([dxdt, dydt])
    par       :    dictionary, parameters
    filename  :    string, name of the figure file
    t = 0     :    int, required for solve_ivp. just a dummy variable
    x_start   :    float, simulation limit
    y_start   :    float, simualtion limit
    x_stop    :    float, 
    y_stop    :    float
    curve_step:    int, resolution of nullclines
    vec_step  :    int, resolution of vector field
    stable_fixed_point_step: int, resolution of grid when searching for stable fixed points
    tmin = 0  :    int, solve_ivp init time
    tmax = 100:    int, solve_ivp final time
    epsil     :    float, set epsil - look at the resolution of the nullclines
    radius    :    float, set radius - look at the distance between crossings.
    xmin      :    float, min limit on x-axis
    xmax      :    float, max limit on x-axis
    ymin      :    float, min limit on y-axis
    ymax      :    float, max limit on y-axis
    x_color   :    string, color of x-nullcline (see matplotlib list of colours)
    y_color   :    string, color of y-nullcline (see matplotlib list of colours)
    title     :    string, figure tilte
    x_label   :    string, x-axis label
    y_label   :    string, y-axis label
    fontsize_label_title: int, font size on axis labels and title
    x_legend  :    string, legend on x-nullcline
    y_legend  :    string, legend on y-nullcline
    ax        :    None/ name of the ax, enables subplots.
    add_legends:   True/False, enables legends
    stable_fixed_points_return: boolean, returns sfp coordinates
    bassin_of_attraction: boolean, enables color coded vectorfield according to tbe bassin of attraction
    tight_layout:  boolean, ensures the plot fits within the frame
    save_figure:   boolean, saves the figure.
    '''
    

    # CREATE GRID POINTS, VARIABLE VALUES.
    if vectorfield:
        vec_x, vec_y = np.meshgrid(np.linspace(x_start, x_stop, vec_step),
                           np.linspace(y_start, y_stop, vec_step))
        vec_var = vec_x, vec_y

    if vectorfield and logscale:
        vec_x, vec_y = np.meshgrid(np.logspace(x_start, np.log10(np.abs(x_stop)), vec_step),
                                   np.logspace(y_start, np.log10(np.abs(y_stop)), vec_step))
        vec_var = vec_x, vec_y


    curve_x, curve_y = np.meshgrid(np.linspace(x_start, x_stop, curve_step),
                               np.linspace(y_start, y_stop, curve_step))
    curve_var = curve_x, curve_y

    if logscale:
        curve_x, curve_y = np.meshgrid(np.logspace(x_start, np.log10(np.abs(x_stop)), curve_step),
                                       np.logspace(y_start, np.log10(np.abs(y_stop)), curve_step))
        curve_var = curve_x, curve_y



    # CREATE FIGURE
    if ax is None: # should I edit it to ax is ax?
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.set_title(title, fontsize=fontsize_label_title)
        ax.set_xlabel(x_label, fontsize=fontsize_label_title)
        ax.set_ylabel(y_label, fontsize=fontsize_label_title)
        
        # PLOT NULLCLINES (curve step)
        x_nullcline = ax.contour(curve_x, curve_y, model_2D(t, curve_var, par)[0],
                                 levels=[0], colors=x_color)
        y_nullcline = ax.contour(curve_x, curve_y, model_2D(t, curve_var, par)[1],
                                 levels=[0], colors=y_color)
        
        if add_legends:
            ax.text(x_stop+0.1, y_stop-0.1, r'$-$', color=x_color, fontsize=12,
                    bbox=dict(facecolor='white', edgecolor='none'))
            ax.text(x_stop+0.3, y_stop-0.1, x_legend , color='black',
                    bbox=dict(facecolor='white', edgecolor='none'))
            
            ax.text(x_stop+0.1, y_stop-0.5, r'$-$', color=y_color, fontsize=12,
                    bbox=dict(facecolor='white', edgecolor='none'))
            ax.text(x_stop+0.3, y_stop-0.5, y_legend, color='black',
                    bbox=dict(facecolor='white', edgecolor='none'))
           
        
        # PLOT VECTOR FIELDS (basic, vec step)
        if vectorfield and not(basin_of_attraction):
            dxdt, dydt = model_2D(t, vec_var, par)
            magnitude = np.sqrt(dxdt**2 + dydt**2)
            ax.quiver(vec_x, vec_y, dxdt/magnitude, dydt/magnitude, 
                      color='tan', linewidths=1)
        
            if add_legends:
                ax.text(x_stop+0.1, y_stop-0.9,r'$\rightarrow$',
                        color='tan', fontsize=12,
                        bbox=dict(facecolor='white', edgecolor='none'))
                ax.text(x_stop+0.3, y_stop-0.9,'Flow           ', color='black',
                        bbox=dict(facecolor='white', edgecolor='none'))
        
        # PLOT NULLCLINES
        x_nullcline = ax.contour(curve_x, curve_y, model_2D(t, curve_var, par)[0],
                                 levels=[0], colors=x_color)
        y_nullcline = ax.contour(curve_x, curve_y, model_2D(t, curve_var, par)[1],
                                 levels=[0], colors=y_color)

        if find_fixed_points:
            # FIND FIXED POINTS
            num_fixed_points, fixed_points_coord = func_find_fixed_points_from_nullclines2907(x_nullcline, y_nullcline, radius)

            # PLOT STABLE FIXED POINTS
            stable_points = func_stability_analysis(model_2D, fixed_points_coord, par)
            for stable_fp in stable_points:
                ax.scatter(stable_fp[0], stable_fp[1], facecolors="black", edgecolor="black",
                           label=f"({stable_fp[0]:.2e},{stable_fp[1]:.2e})")
                           #label=f"({np.round(stable_fp[0])},{np.round(stable_fp[1])})")

            # PLOT UNSTABLE FIXED POINTS
            for fixed_point in fixed_points_coord:
                if np.all(fixed_point != stable_points):
                    ax.scatter(fixed_point[0], fixed_point[1],facecolors="none", edgecolor="black",
                               label=f"({fixed_point[0]:.2e},{fixed_point[1]:.2e})")
                               #label=f"({np.round(fixed_point[0])},{np.round(fixed_point[1])})")

            if add_legends_fp:
                ax.legend(loc=loc_legend, fontsize=fontsize_legend)
        
            # COLOUR CODED VECTOR FIELD ACCORDINGLY TO THE BASIN OF ATTRACTION
            if vectorfield and basin_of_attraction and not(logscale):
                color_tuple = ("indigo", "teal", "firebrick", "gold")
                for (color, sfp) in zip(color_tuple, stable_points):
                    for x in np.linspace(x_start, x_stop, vec_step):
                        for y in np.linspace(x_start, y_stop, vec_step):
                            results=solve_ivp(model_2D, [tmin, tmax], (x, y), args=[par])
                            steady_state=results.y[:,-1]
                            if np.all(np.round(steady_state)==np.round(sfp)):
                                dx1dt, dx2dt = model_2D(t, (x, y) , par)
                                magnitude = np.sqrt(dx1dt**2 + dx2dt**2)
                                ax.quiver(x, y, dx1dt/magnitude, dx2dt/magnitude, color=color, scale=25, width=0.005, alpha=0.2)

        
            if vectorfield and basin_of_attraction and logscale:
                color_tuple = ("indigo", "teal", "firebrick", "gold")
                for (color, sfp) in zip(color_tuple, stable_points):
                    for x in np.logspace(x_start, np.log10(np.abs(x_stop)), vec_step):
                        for y in np.logspace(x_start, np.log10(np.abs(y_stop)), vec_step):
                            results=solve_ivp(model_2D, [tmin, tmax], (x, y), args=[par])
                            steady_state=results.y[:,-1]
                            if np.all(np.round(steady_state)==np.round(sfp)):
                                dx1dt, dx2dt = model_2D(t, (x, y) , par)
                                magnitude = np.sqrt(dx1dt**2 + dx2dt**2)
                                ax.quiver(x, y, dx1dt/magnitude, dx2dt/magnitude, color=color, scale=25, width=0.005, alpha=0.2)

        
        # FIGURE ADJUSTMENTS
        
        # by default, the figure shows all data points unless mentioned.
        if not(logscale):
            if xmin == None:
                xmin = np.power(10.0,x_start)
            else:
                xmin = xmin
        
            if xmax == None:
                xmax = x_stop
            else:
                xmax = xmax
            
            if ymin == None:
                ymin = np.power(10.0, y_start)
            else:
                ymin = ymin
        
            if ymax == None:
                ymax = y_stop
            else:
                ymax = ymax

        if logscale:
            if xmin == None:
                xmin = np.power(10.0,x_start)
            else:
                xmin = xmin
        
            if xmax == None:
                xmax = x_stop
            else:
                xmax = xmax
            
            if ymin == None:
                ymin = np.power(10.0, y_start)
            else:
                ymin = ymin
        
            if ymax == None:
                ymax = y_stop
            else:
                ymax = ymax

        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        if logscale:
            ax.set_xscale("log", nonpositive='clip')
            ax.set_yscale("log", nonpositive='clip')
        
        # fit within the frame.
        if tight_layout:
            plt.tight_layout()
            
            
        # SAVE FIGURE
        if save_figure:
            plt.savefig(f"{filename}.pdf", dpi=600)
            
            
        # RETURN
        if find_fixed_points and stable_fixed_points_return:
            return fixed_points_coord, stable_points
        if return_nullclines:
            return x_nullcline, y_nullcline


# statistical tests
def shapiro_wilk_test_and_plots(dataframe,
                                network, # or "network" (velocity data)
                                datatype, # or "velocity"
                                date,
                                alpha_shapiro = 0.05,
                                verbose=True, # plot the distribution
                                savefig=True): # save the distribution
    
    """
    17.oct.2024
    dataframe: pandas dataframe, diff_tau_all or velocity_all.
               (read notebook "statistical analysis on PSA data")
    network:  string, if delay time diff, use "two_networks". If velocity data, use "network"
    datatype: string, "time_diff" or "velocity"
    date:     string, date of today
    alpha_shapiro: float, significance level = 0.05 by default
    verbose:  boolean, if True, it plots the distrbutions
    savefig:  boolean, if True, it saves the plotted distributions with filename
              "f"{datatype}_dist_{nets}_{param}_{change}_{date}.pdf""
             
    """
    # sample data
    shapiro_test_results = []

    for nets in dataframe[network].unique():
        for param in dataframe["parameter"].unique():
            for change in dataframe["rel change"].unique():
                
                # handle one condition at a time
                mask_networks = dataframe[network]==nets
                mask_param = dataframe["parameter"]==param
                mask_relchange = dataframe["rel change"]==change
                
                # which datatype
                if datatype=="time_diff":
                    analyse_data = dataframe[mask_networks & mask_param & mask_relchange]["time diff"]
                    xlabel=r"$\Delta\tau$"
                if datatype=="velocity":
                    analyse_data = dataframe[mask_networks & mask_param & mask_relchange]["velocity"]
                    xlabel="Velocity"
                
                
                #condition 1 and 2
                condition1 = dataframe[mask_networks & mask_param & mask_relchange]["condition 1"].unique()[0]
                condition2 = dataframe[mask_networks & mask_param & mask_relchange]["condition 2"].unique()[0]
                
                # perform shapiro wilk test
                stat, p_value = shapiro(analyse_data)
                
                
                # check if we reject the null hypothesis (normality)
                if p_value > alpha_shapiro:
                    reject = False #Fail to reject H₀: Data looks normally distributed
                else:
                    reject = True  #Reject H₀: Data does not look normally distributed
                
                #print(nets, param, change, p_value, alpha_shapiro, reject)
                
                # plot the distribution
                if verbose:
                    Nbins = int(np.sqrt(len(analyse_data)))
                    plt.figure()
                    freq, bins, patches = plt.hist(analyse_data, bins=Nbins)
                    plt.xlabel(xlabel)
                    plt.ylabel("Frequency")
                    plt.title(f"Nbins={Nbins}, p-value = {'{:.2e}'.format(p_value)}, reject = {reject} \n network = {nets} parameter = {param}, rel. change = {change}, \n bistable={condition1}, ref_sfp: {condition2}",
                              fontsize=10)
                    plt.tight_layout()
                    if savefig:
                        plt.savefig(f"{datatype}_dist_{nets}_{param}_{change}_{date}.pdf", dpi=600)
                    plt.close()
                
                
                data = {"networks": nets,
                        "parameter":param,
                        "rel change":change,
                        "p_value": p_value,
                        "significance level":alpha_shapiro,
                        "reject": reject,
                        "condition 1": condition1,
                        "condition 2": condition2,
                       }
                
                shapiro_test_results.append(data)

    # Convert the list of dictionaries into a Pandas DataFrame
    df_shapiro_results = pd.DataFrame(shapiro_test_results)

    # Export the DataFrame to a CSV file
    df_shapiro_results.to_csv(f"shapiro_test_results_{datatype}_{date}.csv", index=False)

    print(f"Shapiro-Wilk test results saved to shapiro_test_results{datatype}_{date}.csv")
    
def MW_test(df_compare,
            df_ref,
            datatype,
            date,
            alternative,
           ):
    """
    17.oct.24
    One-sided Mann-Whitney U-test using scipy.stats package.
    
    H0: the time_diff in df_compare is equal to or smaller than the time_diff in df_ref (alternative="greater").
    H0: the "velocity" in df_compare is equal to or larger than the "velocity" in df_ref (alternative="less")
    
    Input:
    -------
    df_compare: pandas dataframe, contains only data for one network
    df_ref:     pandas dataframe, contains only data for one network
    datatype:   string, "time diff" or "velocity"
    date:       string, todays date
    alternative: string, greater or less. depends on the null hypothesis.
    
    Output:
    -------
    csv file with MW-results
    
    """
    
    # significance levels
    alpha_mannwhitney_005 = 0.05
    alpha_mannwhitney_001 = 0.01
    alpha_mannwhitney_0001=0.001
    
    # to the data csv file
    if datatype=="time diff":
        networktype = "two_networks"
    if datatype=="velocity":
        networktype = "network"

    MWresults = []
    # the function
    for param in df_compare["parameter"].unique():
        for change in df_compare["rel change"].unique():
            
            # prepare/ organise data
            mask_compare = (df_compare["parameter"]==param) & (df_compare["rel change"]==change)
            mask_reference = (df_ref["parameter"]==param) & (df_ref["rel change"]==change)
            
            df_compare_param_change = df_compare[mask_compare][datatype].reset_index(drop=True)
            df_ref_param_change = df_ref[mask_reference][datatype].reset_index(drop=True)
            
            if not df_compare_param_change.empty and not df_ref_param_change.empty:
                
                # perfrom one sided mann whitney u test
                u_stat, p_value = mannwhitneyu(df_compare_param_change, df_ref_param_change, alternative=alternative)
            else:
                continue #skip if either dataframe is empty.
            
            # check if we can reject the null hypothesis
            
            if p_value > alpha_mannwhitney_005:
                reject_005 = False #Fail to reject H₀: no evidence that df_compare > df_ref
            elif p_value < alpha_mannwhitney_005:
                reject_005 = True
            
            if p_value > alpha_mannwhitney_001:
                reject_001 = False
            elif p_value < alpha_mannwhitney_001:
                reject_001 = True
            
            if p_value > alpha_mannwhitney_0001:
                reject_0001 = False
            elif p_value < alpha_mannwhitney_0001:
                reject_0001 = True
                
            data = {"df compare": df_compare[networktype].unique()[0],
                    "df ref": df_ref[networktype].unique()[0],
                    "parameter":param,
                    "rel change": change,
                    "p-value":p_value,
                    "reject 0.05":reject_005,
                    "reject 0.01":reject_001,
                    "reject 0.001": reject_0001
                   }
            MWresults.append(data)
            
    df_mannwhitney_results = pd.DataFrame(MWresults)

    # Export the DataFrame to a CSV file
    df_mannwhitney_results.to_csv(f"MW_results_{datatype}_{df_compare[networktype].unique()[0]}_{df_ref[networktype].unique()[0]}_{date}.csv", index=False)
    print(f"Mann-Whitney U test results are saved to MW_results_{datatype}_{df_compare[networktype].unique()[0]}_{df_ref[networktype].unique()[0]}_{date}.csv ")

def wilcoxon_test_GNE_timedata(dataframe,
                               date,
                               datatype="time diff",
                               ):
    """
    Wilcoxon test. Answering if the delay time diff between GNE and GN is larger than zero.
    H0: The delay time diffence between the GNE and GN is equal or smaller than zero.
    
    Input:
    ------
    dataframe: pandas dataframe, contains categories: parameter, rel change, two_networks, time diff
    date: string, todays date. used in filename of csv file
    
    Output
    ------
    csv file named: wilcoxon_test_{dataframe['two_networks'].unique()[0]}_{date}.csv
    
    """
    wilcoxon_test_results=[]
    
    alpha_005 = 0.05
    alpha_001 = 0.01
    alpha_0001 = 0.001
    
    for param in dataframe["parameter"].unique():
        for change in dataframe["rel change"].unique():
            
            # filter for param and change
            mask = (dataframe["parameter"]==param) & (dataframe["rel change"]==change)
            dataframe_param_change = dataframe[mask][datatype].reset_index(drop=True)
            
            if not dataframe_param_change.empty:
                
                # Perform the one-sample Wilcoxon signed-rank test
                stat, p_value = wilcoxon(dataframe_param_change, alternative='greater', zero_method='wilcox')
            else:
                continue
                
            if p_value > alpha_005:
                reject_005 = False     #Fail to reject H₀: no evidence that delay time diff > 0
            elif p_value < alpha_005:
                reject_005 = True
                
            if p_value > alpha_001:
                reject_001 = False
            elif p_value < alpha_001:
                reject_001 = True
                
            if p_value > alpha_0001:
                reject_0001 = False
            elif p_value < alpha_0001:
                reject_0001 = True
                
            # sample data
            data = {"network":dataframe["two_networks"].unique()[0],
                    "parameter": param,
                    "rel change": change,
                    "p value":p_value,
                    "reject 0.05": reject_005,
                    "reject 0.01": reject_001,
                    "reject 0.001": reject_0001
            }
            wilcoxon_test_results.append(data)
    df_wilcoxon_test_results = pd.DataFrame(wilcoxon_test_results)
    df_wilcoxon_test_results.to_csv(f"wilcoxon_test_{dataframe['two_networks'].unique()[0]}_{date}.csv", index=False)
    print(f"Wilcoxon test results are saved in csv file wilcoxon_test_{dataframe['two_networks'].unique()[0]}_{date}.csv")



def plot_distributions_at_diff_scale_factors(dataframe, network, parameter, scalefactor_array,
                                             title, x_label, date, font_size=30, font_size_legend=14,
                                             datatype="time diff",legend_loc="best",xmin=-25, xmax=15,
                                             savefig=False):

    """
    Created: 05.11.2024
    plot multiple histograms in one figure.
    If savefig=True, the figure is saved as "{datatype}_distributions_{parameter}_{date}.pdf" with dpi=600.
    
    input:
    dataframe: pandas dataframe containing categories "two_networks" if datatype="time diff" or "network" if datatype="velocity",
               "parameter", and "rel change"
    network:   string, possible values "GNEvGN", "GNEO1vGN", "GNEO2vGN"
    paraemter: string, possible values (see par0)
    scalefactor_array: list or numpy array, possible values are 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 3.0, 4.0, and 5.0
    title:     string
    x_label:   string
    date:      string
    
    """

    # colors of the histograms
    color_array = ["cornflowerblue", "darkorange", "darkorchid", "lightseagreen"]
    
    # filters/ masks
    if datatype=="time diff":
        mask0 = dataframe["two_networks"]==network
    if datatype=="velocity":
        mask0 = dataframe["network"]==network

    mask1 = dataframe["parameter"]==parameter
    
    # reference data
    maskref = dataframe["rel change"]==1.0
    ref_data = dataframe[mask0 & mask1 & maskref][datatype]
    Nbins = int(np.sqrt(len(ref_data)))
    
    # create figure + plot reference data
    plt.figure(figsize=(6,5))
    freqref, binsref, patchesref = plt.hist(ref_data, bins=Nbins, density=True, histtype="step",
                                           color="black", label="s = 1.0")

    # plot comparison data
    for color, scalefactor in zip(color_array[:len(scalefactor_array)], scalefactor_array):
        
        mask2 = dataframe["rel change"]==scalefactor    
        analyse_data = dataframe[mask0 & mask1 & mask2][datatype]
        freq, bins, patches = plt.hist(analyse_data, bins=Nbins, density=True, histtype="step",
                                       color=color, label=f"s = {scalefactor}")
        
    # figure title, axis labels
    plt.title(f"{title}", fontsize=font_size)
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel("Normalised to unity", fontsize=font_size_legend)
    plt.legend(loc=legend_loc,fontsize=font_size_legend)
    
    # adjust + save
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    
    if savefig:
        plt.savefig(f"{datatype}_distributions_{parameter}_{date}.pdf", dpi=600)

    


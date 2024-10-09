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
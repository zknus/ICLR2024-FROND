import torch
import math

def fractional_pow(base, exponent):
    eps = 1e-4
    return torch.pow(base, exponent)

def Predictor(func,y0,beta,tspan,**options):
    """Use one-step Adams-Bashforth (Euler) method to integrate Caputo equation
        D^beta y(t) = f(t,y)
        Args:
          beta: fractional exponent in the range (0,1)
          f: callable(y,t) returning a numpy array of shape (d,)
             Vector-valued function to define the right hand side of the system
          y0: array of shape (d,) giving the initial state vector y(t==0)
          tspan (array): The sequence of time points for which to solve for y.
            These must be equally spaced, e.g. np.arange(0,10,0.005)
            tspan[0] is the intial time corresponding to the initial state y0.
        Returns:
          y: array, with shape (len(tspan), len(y0))
             With the initial value y0 in the first row
        Raises:
          FODEValueError
        See also:
          K. Diethelm et al. (2004) Detailed error analysis for a fractional Adams
             method
          C. Li and F. Zeng (2012) Finite Difference Methods for Fractional
             Differential Equations
        """
    N = len(tspan)
    # print("N: ", N)
    h = (tspan[-1] - tspan[0]) / (N - 1)
    # print("h: ", h)
    gamma_beta =  1 / math.gamma(beta)
    fhistory = []
    device = y0.device
    yn = y0.clone()


    for k in range(N):
        tn = tspan[k]
        f_k = func(tn,yn)
        fhistory.append(f_k)

        # can apply short memory here
        if 'memory' not in options:
            memory = k
        else:
            memory = options['memory']
        memory_k = max(0, k - memory)


        j_vals = torch.arange(0, k + 1, dtype=torch.float32,device=device).unsqueeze(1)
        b_j_k_1 = (fractional_pow(h, beta) / beta) * (fractional_pow(k + 1 - j_vals, beta) - fractional_pow(k - j_vals, beta))
        temp_product = torch.stack([b_j_k_1[i] * fhistory[i] for i in range(memory_k,k + 1)])
        b_all_k = torch.sum(temp_product, dim=0)
        yn = y0 + gamma_beta * b_all_k
    # release memory
    del fhistory
    del b_j_k_1
    del temp_product
    return yn


def Predictor_Corrector(func,y0,beta,tspan,**options):
    """Use one-step Adams-Bashforth (Euler) method to integrate Caputo equation
        D^beta y(t) = f(t,y)

        Raises:
          FODEValueError
        See also:
          K. Diethelm et al. (2004) Detailed error analysis for a fractional Adams
             method
          C. Li and F. Zeng (2012) Finite Difference Methods for Fractional
             Differential Equations
        """
    N = len(tspan)
    # print("N: ", N)
    h = (tspan[-1] - tspan[0]) / (N - 1)
    # print("h: ", h)
    gamma_beta =  1 / math.gamma(beta)
    fhistory = []
    a_item = torch.pow(h, beta) / (beta * (beta + 1))
    device = y0.device
    # predictor
    yn = y0.clone()
    for k in range(N):
        tn = tspan[k]
        f_k = func(tn,yn)
        fhistory.append(f_k)
        j_vals = torch.arange(0, k + 1, dtype=torch.float32,device=device).unsqueeze(1)
        b_j_k_1 = (fractional_pow(h, beta) / beta) * (fractional_pow(k + 1 - j_vals, beta) - fractional_pow(k - j_vals, beta))
        temp_product = torch.stack([b_j_k_1[i] * fhistory[i] for i in range(k + 1)])
        b_all_k = torch.sum(temp_product, dim=0)
        yn = y0 + gamma_beta * b_all_k


    fhistory_new = torch.zeros((N, *y0.shape), dtype=torch.float32,device=device)
    # corrector
    if 'corrector_step' not in options:
        corrector_step = 1
    else:
        corrector_step = options['corrector_step']
    for _ in range(corrector_step):
        yn_corrector = y0.clone()
        for k in range(N):
            tn = tspan[k]
            fhistory_new[k] = func(tn, yn_corrector)
            a_j_k_1 = torch.zeros((k + 1, 1), dtype=torch.float32,device=device)
            a_j_k_1[0] = a_item * (torch.pow(k, beta + 1) - (k - beta) * torch.pow(k + 1, beta))
            for j in range(1, k + 1):
                a_j_k_1[j] = a_item * (torch.pow(k + 2 - j, beta + 1) + torch.pow(k - j, beta + 1) - 2 * torch.pow(k + 1 - j,beta + 1))

            if a_j_k_1.shape != fhistory_new[:k + 1].shape:
                a_j_k_1 = a_j_k_1.unsqueeze(-1)
            a_j_k_all = a_j_k_1 * fhistory_new[:k+1]
            a_all = torch.sum(a_j_k_all, dim=0)
            a_k_k = (a_item * fhistory[k])
            yn_corrector = y0 + gamma_beta * (a_all + a_k_k)
        fhistory = fhistory_new

    # release memory
    del fhistory
    del b_j_k_1
    del fhistory_new

    return yn_corrector







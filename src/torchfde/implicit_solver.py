import torch
import math

def Implicit_l1(func,y0,beta,tspan,**options):
    """Use one-step Implicit_l1 method to integrate Caputo equation
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
    h = (tspan[N - 1] - tspan[0]) / (N - 1)
    yn = y0.clone()
    device = y0.device

    yn_all = []
    u_h = (torch.pow(h, beta) * math.gamma(2 - beta))
    yn_all.append(yn)

    for k in range(1, N):
        tn = tspan[k]
        fhistory_k = func(tn, yn)
        y_sum = 0
        for j in range(0, k - 2):
            R_k_j = torch.pow(k - j, 1 - beta) - torch.pow(k - j - 1, 1 - beta)
            y_sum = y_sum + R_k_j * (yn_all[j + 1] - yn_all[j])
        yn = yn + u_h * fhistory_k - y_sum
        yn_all.append(yn)

    return yn

    # if 'corrector_step' not in options:
    #     corrector_step = 0
    # else:
    #     corrector_step = options['corrector_step']
    #
    # if corrector_step > 0:
    #     for _ in range(corrector_step):
    #         yn_corrector = y0.clone()
    #         yn_all_corrector = []
    #         yn_all_corrector.append(yn_corrector)
    #         for k in range(1, N):
    #             f_predictor = func(tspan[k], yn_all[k])
    #             y_sum = 0
    #             for j in range(0, k - 2):
    #                 R_k_j = torch.pow(k - j, 1 - beta) - torch.pow(k - j - 1, 1 - beta)
    #                 y_sum = y_sum + R_k_j * (yn_all_corrector[j + 1] - yn_all_corrector[j])
    #             yn_corrector = yn_corrector + u_h * f_predictor - y_sum
    #             yn_all_corrector.append(yn_corrector)
    #         yn_all = yn_all_corrector
    #     return yn_corrector
    # else:
    #     return yn




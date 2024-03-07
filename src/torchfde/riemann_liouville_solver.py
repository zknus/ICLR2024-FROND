import torch
import math

def GLmethod(func,y0,beta,tspan,**options):
    """Use GL  method to integrate Riemann-Liouville equation
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
    device = y0.device
    c = torch.zeros(N + 1, dtype=torch.float64,device=device)
    c[0] = 1
    for j in range(1, N + 1):
        c[j] = (1 - (1 + beta) / j) * c[j - 1]
    yn = y0.clone()
    y_history = []
    y_history.append(yn)
    for k in range(1, N):
        tn = tspan[k]
        right = 0
        for j in range(1, k + 1):
            right = (right + c[j] * y_history[k - j])
        yn = func(tn, yn) * torch.pow(h, beta) - right
        y_history.append(yn)
    return yn

def RLcoeffs(index_k, index_j, alpha):
    """Calculates coefficients for the RL differintegral operator.

    see Baleanu, D., Diethelm, K., Scalas, E., and Trujillo, J.J. (2012). Fractional
        Calculus: Models and Numerical Methods. World Scientific.
    """

    if index_j == 0:
        return ((index_k - 1) ** (1 - alpha) - (index_k + alpha - 1) * index_k ** -alpha)
    elif index_j == index_k:
        return 1
    else:
        return ((index_k - index_j + 1) ** (1 - alpha) + (index_k - index_j - 1) ** (1 - alpha) - 2 * (
                    index_k - index_j) ** (1 - alpha))


def Product_Trap(func,y0,beta,tspan,**options):
    N = len(tspan)
    h = (tspan[N - 1] - tspan[0]) / (N - 1)
    device = y0.device
    c = torch.zeros(N + 1, dtype=torch.float64,device=device)
    c[0] = 1
    for j in range(1, N+1):
        c[j] = (1 - (1+beta)/j) * c[j-1]
    yn = y0.clone()
    y_history = []
    y_history.append(yn)
    for k in range(1, N):
        tn = tspan[k]
        right = 0
        for j in range(0, k):
            right = (right + RLcoeffs(k, j, beta) * y_history[j])
        yn = math.gamma(2 - beta) * func(tn, yn) * torch.pow(h, beta) - right
        y_history.append(yn)
    return yn



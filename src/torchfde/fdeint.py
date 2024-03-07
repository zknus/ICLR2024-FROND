'''
This file is part of torchfde, a library for solving fractional differential equations in PyTorch.
date: 2023-11-29
author: Kai Zhao
last modified: 2023-11-29
last update: create this file
'''





from .utils_fde import _check_inputs
from .explicit_solver import Predictor,Predictor_Corrector
from .implicit_solver import Implicit_l1
from .riemann_liouville_solver import GLmethod,Product_Trap
SOLVERS = {"predictor":Predictor,
          "corrector":Predictor_Corrector,
           "implicitl1":Implicit_l1,
           "gl":GLmethod,
           "trap":Product_Trap

}

def fdeint(func,y0,beta,t,step_size,method,options=None):
    """Integrate a system of ordinary differential equations.

      Solves the initial value problem for a non-stiff system of first order ODEs:
          ```
          D^(\beta)_t = func(t, y), y(t[0]) = y0
          ```
      where y is a Tensor or tuple of Tensors of any shape.

      Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

      Args:
          func: Function that maps a scalar Tensor `t` and a Tensor holding the state `y`
              into a Tensor of state derivatives with respect to time. Optionally, `y`
              can also be a tuple of Tensors.
          y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
              can also be a tuple of Tensors.
          t: float torch tensor,the integrate terminate time. the default initial time point is set as 0.
          step_size: float torch tensor, the step size of the integrate method.
          method: optional string indicating the integration method to use.
          options: optional dict of configuring options for the indicated integration
              method. Can only be provided if a `method` is explicitly set.
      Returns:
          y: Tensor, where the first dimension corresponds to different
              time points. Contains the solved value of y for each desired time point in
              `t`, with the initial value `y0` being the first element along the first
              dimension.

      Raises:
          ValueError: if an invalid `method` is provided.
      """
    func, y0, tspan, method, beta= _check_inputs(func, y0, t,step_size,method,beta, SOLVERS)
    if options is None:
        options = {}
    solution = SOLVERS[method](func=func, y0=y0, beta=beta, tspan=tspan,**options)

    return solution


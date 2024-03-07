import torch

import warnings

def _check_inputs(func, y0, t, step_size,method,beta, SOLVERS):








    if method not in SOLVERS:
        raise ValueError('Invalid method "{}". Must be one of {}'.format(method,
                                                                         '{"' + '", "'.join(SOLVERS.keys()) + '"}.'))



    # check t is a float tensor, if not  convert it to one
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32, device=y0.device)
        # print("t converted to tensor")
    else:
        t = t.to(y0.device)
    # check t is > 0 else raise error
    if not (t > 0).all():
        raise ValueError("t must be > 0")
    # ~Backward compatibility

    # # Add perturb argument to func.
    # func = _PerturbFunc(func)

    # check beta is a float tensor, if not  convert it to one
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=torch.float32, device=y0.device)
        # print("beta converted to tensor")
    else:
        beta = beta.to(y0.device)
    # check beta is > 0 else raise error
    if not (beta > 0).all():
        raise ValueError("beta must be > 0")
    # check beta is <= 1 else raise warning
    if not (beta <= 1).all():
        warnings.warn("beta should be <= 1 for the initial value problem")

    # check stepsize is a float tensor, if not  convert it to one
    if not isinstance(step_size, torch.Tensor):
        step_size = torch.tensor(step_size, dtype=torch.float32, device=y0.device)
        # print("step_size converted to tensor")
    else:
        step_size = step_size.to(y0.device)
    # check step_size is > 0 else raise error
    if not (step_size > 0).all():
        raise ValueError("step_size must be > 0")

    # check step_size is <= t else raise error
    if not (step_size < t).all():
        raise ValueError("step_size must be < t")
    tspan = torch.arange(0,t,step_size)




    return func, y0, tspan, method,beta


def _check_timelike(name, timelike, can_grad):
    assert isinstance(timelike, torch.Tensor), '{} must be a torch.Tensor'.format(name)
    _assert_floating(name, timelike)
    assert timelike.ndimension() == 1, "{} must be one dimensional".format(name)
    if not can_grad:
        assert not timelike.requires_grad, "{} cannot require gradient".format(name)
    # diff = timelike[1:] > timelike[:-1]
    # assert diff.all() or (~diff).all(), '{} must be strictly increasing or decreasing'.format(name)


def _assert_floating(name, t):
    if not torch.is_floating_point(t):
        raise TypeError('`{}` must be a floating point Tensor but is a {}'.format(name, t.type()))
class _ReverseFunc(torch.nn.Module):
    def __init__(self, base_func, mul=1.0):
        super(_ReverseFunc, self).__init__()
        self.base_func = base_func
        self.mul = mul

    def forward(self, t, y):
        return self.mul * self.base_func(-t, y)

def _assert_increasing(name, t):
    assert (t[1:] > t[:-1]).all(), '{} must be strictly increasing or decreasing'.format(name)

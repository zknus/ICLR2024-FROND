from function_transformer_attention import ODEFuncTransformerAtt
from function_GAT_attention import ODEFuncAtt
from function_laplacian_diffusion import LaplacianODEFunc

from function_laplacian_convection import ODEFuncLapCONV

from function_GAT_convection import ODEFuncAttConv


from function_transformer_convection import ODEFuncTransConv


from block_constant_fractional import ConstantODEblock_FRAC
from block_transformer_fractional import AttODEblock_FRAC

from function_laplacian_graphcon import LaplacianODEFunc_graphcon
from function_transformer_graphcon import ODEFuncTransformerAtt_graphcon
from function_GAT_graphcon import ODEFuncAtt_graphcon



class BlockNotDefined(Exception):
  pass

class FunctionNotDefined(Exception):
  pass


def set_block(opt):
  ode_str = opt['block']
  if ode_str == 'constant_frac':
    block = ConstantODEblock_FRAC
  elif ode_str == 'att_frac':
    block = AttODEblock_FRAC
  else:
    raise BlockNotDefined
  return block


def set_function(opt):
  ode_str = opt['function']
  if ode_str == 'laplacian':
    f = LaplacianODEFunc
  elif ode_str == 'GAT':
    f = ODEFuncAtt
  elif ode_str == 'transformer':
    f = ODEFuncTransformerAtt
  elif ode_str == 'lapconv':
    f = ODEFuncLapCONV
  elif ode_str == 'gatconv':
    f = ODEFuncAttConv
  elif ode_str == 'transconv':
    f = ODEFuncTransConv
  elif ode_str == 'lapgraphcon':
    f = LaplacianODEFunc_graphcon
  elif ode_str == 'transgraphcon':
    f = ODEFuncTransformerAtt_graphcon
  elif ode_str == 'gatgraphcon':
    f = ODEFuncAtt_graphcon
  else:
    raise FunctionNotDefined
  return f

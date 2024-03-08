import torch
from torch import nn
import torch_sparse

from base_classes import ODEFunc
from utils import MaxNFEException
from torch_geometric.utils.loop import add_remaining_self_loops,remove_self_loops
from torch_geometric.utils import get_laplacian

# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(LaplacianODEFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))

    # if opt['self_loop_weight'] > 0:
    #   self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
    #                                                                fill_value=opt['self_loop_weight'])
    # else:
    #   self.edge_index, self.edge_weight = data.edge_index, data.edge_attr
    #
    # self.edge_index, self.edge_weight = remove_self_loops(self.edge_index, self.edge_weight)
    # self.lin2 = nn.Linear(in_features * 2, out_features)
    # nn.init.xavier_normal_(self.lin2.weight, gain=1.414)

  def sparse_multiply(self, x):
    if self.opt['block'] in ['attention','att_frac']:  # adj is a multihead attention
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    self.nfe += 1
    # print("x shape: ", x.shape)
    # print("edge_index shape: ", self.edge_index.shape)
    ax = self.sparse_multiply(x)

    # ax = torch.cat([x, ax], axis=1)
    # ax = self.lin2(ax)

    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    # f = ax -x
    f = alpha * (ax - x)
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0
    return f

import torch
from torch import nn
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops,remove_self_loops
import numpy as np
# from data import get_dataset
from utils import MaxNFEException, squareplus
from base_classes import ODEFunc
from torch_scatter import scatter
import math
from torch_geometric.utils import get_laplacian
import torch.nn.functional as F


class ODEFuncLapCONV(ODEFunc):

  def __init__(self, in_features, out_features, opt, data, device):
    super(ODEFuncLapCONV, self).__init__(opt, data, device)

    if opt['self_loop_weight'] > 0:
      self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                                   fill_value=opt['self_loop_weight'])
    else:
      self.edge_index, self.edge_weight = data.edge_index, data.edge_attr
    # print("self.edge_index: ", self.edge_index.shape)
    self.multihead_att_layer = SpGraphTransAttentionLayer(in_features, out_features,  opt,device,edge_weights=self.edge_weight).to(
      device)
    self.device = device

    self.edge_index,self.edge_weight = remove_self_loops (self.edge_index, self.edge_weight)

    self.edge_index_lap, self.edge_weight_lap = get_laplacian(self.edge_index, self.edge_weight,normalization='sym')
    self.edge_index_lap = self.edge_index_lap.to(device)
    self.edge_weight_lap = self.edge_weight_lap.to(device)



    self.gate = nn.Linear(2 * in_features, 1)
    nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    self.lin1 = nn.Linear( in_features, out_features)
    nn.init.xavier_normal_(self.lin1.weight, gain=1.414)

    self.lin2 = nn.Linear(in_features * 2, out_features)
    nn.init.xavier_normal_(self.lin2.weight, gain=1.414)

    self.weight_low, self.weight_high, self.weight_mlp = (
      nn.Parameter(torch.FloatTensor(in_features, out_features).to(device)),
      nn.Parameter(torch.FloatTensor(in_features, out_features).to(device)),
      nn.Parameter(torch.FloatTensor(in_features, out_features).to(device)),
    )

    self.output_low, self.output_high, self.output_mlp = (
      nn.Parameter(torch.FloatTensor(out_features, out_features).to(device)),
      nn.Parameter(torch.FloatTensor(out_features, out_features).to(device)),
      nn.Parameter(torch.FloatTensor(out_features, out_features).to(device)),
    )





    stdv = 1.0 / math.sqrt(self.weight_mlp.size(1))


    self.weight_low.data.uniform_(-stdv, stdv)
    self.weight_high.data.uniform_(-stdv, stdv)
    self.weight_mlp.data.uniform_(-stdv, stdv)

    self.output_low.data.uniform_(-stdv, stdv)
    self.output_high.data.uniform_(-stdv, stdv)
    self.output_mlp.data.uniform_(-stdv, stdv)

    self.bn_in_1 = torch.nn.BatchNorm1d(opt['hidden_dim'])
    self.bn_in_2 = torch.nn.BatchNorm1d(opt['hidden_dim'])

    self.lamda = nn.Parameter(torch.tensor(0.0))

  def multiply_attention(self, x, attention, v=None):
    num_heads = 4
    mix_features = 0
    if mix_features:
      vx = torch.mean(torch.stack(
        [torch_sparse.spmm(self.edge_index, attention[:, idx], v.shape[0], v.shape[0], v[:, :, idx]) for idx in
         range(num_heads)], dim=0),
        dim=0)
      ax = self.multihead_att_layer.Wout(vx)
    else:
      # mean_attention = attention.mean(dim=1)
      mean_attention = attention
      grad_x = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x) - x
      grad_x_abs = torch.abs(grad_x)
      grad_x_norm = torch.sqrt(torch.sum(torch.clamp(grad_x_abs * grad_x_abs, min=1e-1), 1))
      grad_x_norm_inv = 1 / grad_x_norm
      gu = grad_x_norm_inv[self.edge_index[0, :]]
      gv = grad_x_norm_inv[self.edge_index[1, :]]
      attention2 = gu * gu + gu * gv
      # new_attn = mean_attention * softmax(attention2, self.edge_index[0])
      # attn = scatter(attention2,self.edge_index[0] )[self.edge_index[0]]
      # new_attn = mean_attention * torch.tanh(scatter(attention2,self.edge_index[0] )[self.edge_index[0]])

      new_attn = mean_attention * torch.tanh(attention2)
      # Da = torch.diag(grad_x_norm_inv)
      W = torch.sparse.FloatTensor(self.edge_index, new_attn, (x.shape[0], x.shape[0])).coalesce()
      rowsum = torch.sparse.mm(W, torch.ones((W.shape[0], 1), device=self.device)).flatten()
      diag_index = torch.stack((torch.arange(x.shape[0]), torch.arange(x.shape[0]))).to(self.device)
      dx = torch_sparse.spmm(diag_index, rowsum, x.shape[0], x.shape[0], x)
      ax = torch_sparse.spmm(self.edge_index, new_attn, x.shape[0], x.shape[0], x)
    return ax - dx

  def forward(self, t, x):  # t is needed when called by the integrator

    # attention, values = self.multihead_att_layer(x, self.edge_index)
    # print("attention", attention.shape)

    # x1 = torch.mm(x, self.weight_high)
    # x2 = torch.mm(x, self.weight_low)

    x2 =x

    src = x[self.edge_index[0, :], :]
    dst_k = x[self.edge_index[1, :], :]
    h2 = torch.cat([src, dst_k], dim=1)
    attention1 = torch.tanh(self.gate(h2)).squeeze()

    x_new = F.relu(torch.mm(src-dst_k, self.weight_mlp)) * dst_k
    # print("x_new: ", x_new.shape)

    ax3 = torch_sparse.spmm(self.edge_index, attention1, x_new.shape[0], x_new.shape[0], x_new)
    ax3 = scatter(ax3, self.edge_index[1, :].T, dim=0, reduce="sum")
    # ax3 = scatter(x_new,self.edge_index[1, :].T,dim=0,reduce="sum")

    # ax3 = spmm(self.edge_index,x_new, reduce = "sum")
    # print("ax3: ", ax3.shape)

    # print("attention1", attention1)
    # attention, values = self.multihead_att_layer(x2, self.edge_index)
    # ax1 = torch_sparse.spmm(self.edge_index, attention1, x.shape[0], x.shape[0], x1)

    # ax1 = torch_sparse.spmm(self.edge_index_lap, self.edge_weight_lap, x.shape[0], x.shape[1], x1)
    # ax2 = torch_sparse.spmm(self.edge_index, attention.mean(dim=1), x.shape[0], x.shape[0], x2)
    ax2 = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x2)

    # print("ax2: ", ax2.shape)

    ax3 = self.bn_in_1(ax3)
    ax2 = self.bn_in_2(ax2)

    ax = torch.mm(ax3, self.output_high) + torch.mm(ax2, self.output_low)

    # ax = torch.sigmoid(self.lamda) * ax2 + (1 - torch.sigmoid(self.lamda)) * ax3

    ax = torch.cat([x, ax], axis=1)
    ax = self.lin2(ax)






    # ax = self.multiply_attention(x, attention1, v=None)

    # ax = self.lin1(ax1) + self.lin2(ax2)
    # f = ax
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    f = alpha * (ax - x)
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0

    # f = ax - x
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphTransAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, opt, device, concat=True, edge_weights=None):
    super(SpGraphTransAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = opt['leaky_relu_slope']
    self.concat = concat
    self.device = device
    self.opt = opt
    self.h = int(opt['heads'])
    self.edge_weights = edge_weights

    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % self.h == 0, "Number of heads ({}) must be a factor of the dimension size ({})".format(
      self.h, self.attention_dim)
    self.d_k = self.attention_dim // self.h

    if self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
      self.output_var_x = nn.Parameter(torch.ones(1))
      self.lengthscale_x = nn.Parameter(torch.ones(1))
      self.output_var_p = nn.Parameter(torch.ones(1))
      self.lengthscale_p = nn.Parameter(torch.ones(1))
      self.Qx = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Qx)
      self.Vx = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Vx)
      self.Kx = nn.Linear(opt['hidden_dim']-opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Kx)

      self.Qp = nn.Linear(opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Qp)
      self.Vp = nn.Linear(opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Vp)
      self.Kp = nn.Linear(opt['pos_enc_hidden_dim'], self.attention_dim)
      self.init_weights(self.Kp)

    else:
      if self.opt['attention_type'] == "exp_kernel":
        self.output_var = nn.Parameter(torch.ones(1))
        self.lengthscale = nn.Parameter(torch.ones(1))

      self.Q = nn.Linear(in_features, self.attention_dim)
      self.init_weights(self.Q)

      self.V = nn.Linear(in_features, self.attention_dim)
      self.init_weights(self.V)

      self.K = nn.Linear(in_features, self.attention_dim)
      self.init_weights(self.K)

    self.activation = nn.Sigmoid()  # nn.LeakyReLU(self.alpha)

    self.Wout = nn.Linear(self.d_k, in_features)
    self.init_weights(self.Wout)

  def init_weights(self, m):
    if type(m) == nn.Linear:
      # nn.init.xavier_uniform_(m.weight, gain=1.414)
      # m.bias.data.fill_(0.01)
      nn.init.constant_(m.weight, 1e-5)

  def forward(self, x, edge):
    """
    x might be [features, augmentation, positional encoding, labels]
    """
    # if self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
    if self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
      label_index = self.opt['feat_hidden_dim'] + self.opt['pos_enc_hidden_dim']
      p = x[:, self.opt['feat_hidden_dim']: label_index]
      x = torch.cat((x[:, :self.opt['feat_hidden_dim']], x[:, label_index:]), dim=1)

      qx = self.Qx(x)
      kx = self.Kx(x)
      vx = self.Vx(x)
      # perform linear operation and split into h heads
      kx = kx.view(-1, self.h, self.d_k)
      qx = qx.view(-1, self.h, self.d_k)
      vx = vx.view(-1, self.h, self.d_k)
      # transpose to get dimensions [n_nodes, attention_dim, n_heads]
      kx = kx.transpose(1, 2)
      qx = qx.transpose(1, 2)
      vx = vx.transpose(1, 2)
      src_x = qx[edge[0, :], :, :]
      dst_x = kx[edge[1, :], :, :]

      qp = self.Qp(p)
      kp = self.Kp(p)
      vp = self.Vp(p)
      # perform linear operation and split into h heads
      kp = kp.view(-1, self.h, self.d_k)
      qp = qp.view(-1, self.h, self.d_k)
      vp = vp.view(-1, self.h, self.d_k)
      # transpose to get dimensions [n_nodes, attention_dim, n_heads]
      kp = kp.transpose(1, 2)
      qp = qp.transpose(1, 2)
      vp = vp.transpose(1, 2)
      src_p = qp[edge[0, :], :, :]
      dst_p = kp[edge[1, :], :, :]

      prods = self.output_var_x ** 2 * torch.exp(
        -torch.sum((src_x - dst_x) ** 2, dim=1) / (2 * self.lengthscale_x ** 2)) \
              * self.output_var_p ** 2 * torch.exp(
        -torch.sum((src_p - dst_p) ** 2, dim=1) / (2 * self.lengthscale_p ** 2))

      v = None

    else:
      q = self.Q(x)
      k = self.K(x)
      v = self.V(x)

      # perform linear operation and split into h heads

      k = k.view(-1, self.h, self.d_k)
      q = q.view(-1, self.h, self.d_k)
      v = v.view(-1, self.h, self.d_k)

      # transpose to get dimensions [n_nodes, attention_dim, n_heads]

      k = k.transpose(1, 2)
      q = q.transpose(1, 2)
      v = v.transpose(1, 2)

      src = q[edge[0, :], :, :]
      dst_k = k[edge[1, :], :, :]

    if not self.opt['beltrami'] and self.opt['attention_type'] == "exp_kernel":
      prods = self.output_var ** 2 * torch.exp(-(torch.sum((src - dst_k) ** 2, dim=1) / (2 * self.lengthscale ** 2)))
    elif self.opt['attention_type'] == "scaled_dot":
      prods = torch.sum(src * dst_k, dim=1) / np.sqrt(self.d_k)
    elif self.opt['attention_type'] == "cosine_sim":
      cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
      prods = cos(src, dst_k)
    elif self.opt['attention_type'] == "pearson":
      src_mu = torch.mean(src, dim=1, keepdim=True)
      dst_mu = torch.mean(dst_k, dim=1, keepdim=True)
      src = src - src_mu
      dst_k = dst_k - dst_mu
      cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
      prods = cos(src, dst_k)

    if self.opt['reweight_attention'] and self.edge_weights is not None:
      prods = prods * self.edge_weights.unsqueeze(dim=1)
    if self.opt['square_plus']:
      attention = squareplus(prods, edge[self.opt['attention_norm_idx']])
    else:
      attention = softmax(prods, edge[self.opt['attention_norm_idx']])
    return attention, (v, prods)

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'heads': 2, 'K': 10,
         'attention_norm_idx': 0, 'add_source': False,
         'alpha_dim': 'sc', 'beta_dim': 'sc', 'max_nfe': 1000, 'mix_features': False
         }
  dataset = get_dataset(opt, '../data', False)
  t = 1
  func = ODEFuncBelFA(dataset.data.num_features, 6, opt, dataset.data, device)
  out = func(t, dataset.data.x)

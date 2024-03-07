import argparse
import time
import os
import numpy as np
import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
from GNN import GNN
from data import get_dataset, set_train_val_test_split
from best_params import best_params_dict
from utils import ROOT_DIR
import sys
import json
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import is_undirected, to_undirected
import random
from run_config import parser
torch.autograd.set_detect_anomaly(True)

#run_GNN_frac_all.py --dataset Citeseer --cuda 0 --block constant_frac --function laplacian --time 5 --step_size 1 --hidden_dim 64 --lr 0.01 --input_dropout 0.4 --dropout 0.4 --runtime 1 --seed 123 --epoch 100 --decay 0.01 --method ceuler
def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))


def add_labels(feat, labels, idx, num_classes, device):
  onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
  if idx.dtype == torch.bool:
    idx = torch.where(idx)[0]  # convert mask to linear index
  onehot[idx, labels.squeeze()[idx]] = 1

  return torch.cat([feat, onehot], dim=-1)


def get_label_masks(data, mask_rate=0.5):
  """
  when using labels as features need to split training nodes into training and prediction
  """
  if data.train_mask.dtype == torch.bool:
    idx = torch.where(data.train_mask)[0]
  else:
    idx = data.train_mask
  mask = torch.rand(idx.shape) < mask_rate
  train_label_idx = idx[mask]
  train_pred_idx = idx[~mask]
  return train_label_idx, train_pred_idx


def train(model, optimizer, data):
  model.train()
  optimizer.zero_grad()
  feat = data.x
  if model.opt['use_labels']:
    train_label_idx, train_pred_idx = get_label_masks(data, model.opt['label_rate'])

    feat = add_labels(feat, data.y, train_label_idx, model.num_classes, model.device)
  else:
    train_pred_idx = data.train_mask

  out = model(feat)

  if model.opt['dataset'] == 'ogbn-arxiv':
    lf = torch.nn.functional.nll_loss
    loss = lf(out.log_softmax(dim=-1)[data.train_mask], data.y.squeeze(1)[data.train_mask])
  else:
    lf = torch.nn.CrossEntropyLoss()
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])

  model.fm.update(model.getNFE())
  model.resetNFE()
  loss.backward()
  optimizer.step()
  model.bm.update(model.getNFE())
  model.resetNFE()
  return loss.item()



@torch.no_grad()
def test_OGB(model, data, opt):


  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)


  model.eval()

  if opt['dataset'] == 'ogbn-arxiv':
    name = 'ogbn-arxiv'
    evaluator = Evaluator(name=name)
    out = model(feat).log_softmax(dim=-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
      'y_true': data.y[data.train_mask],
      'y_pred': y_pred[data.train_mask],
    })['acc']
    valid_acc = evaluator.eval({
      'y_true': data.y[data.val_mask],
      'y_pred': y_pred[data.val_mask],
    })['acc']
    test_acc = evaluator.eval({
      'y_true': data.y[data.test_mask],
      'y_pred': y_pred[data.test_mask],
    })['acc']


  return train_acc, valid_acc, test_acc

@torch.no_grad()
def test(model, data,  opt=None):  # opt required for runtime polymorphism
  model.eval()
  feat = data.x
  if model.opt['use_labels']:
    feat = add_labels(feat, data.y, data.train_mask, model.num_classes, model.device)
  logits, accs = model(feat), []
  logits = F.log_softmax(logits, dim=1)
  if opt['dataset'] in [ 'minesweeper', 'workers', 'questions']:
    # print("using ROC-AUC metric")
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
      # pred = logits.max(1)[1]
      # acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
      mask_idx = torch.where(mask)[0]
      y_true = data.y[mask_idx].cpu().numpy()
      y_score = logits[mask_idx].cpu().numpy()
      acc = roc_auc_score(y_true=data.y[mask_idx].cpu().numpy(),
                                         y_score=logits[:, 1][mask_idx].cpu().numpy()).item()
      accs.append(acc)

  else:

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
      pred = logits[mask].max(1)[1]
      acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
      accs.append(acc)
  return accs


def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)




def merge_cmd_args(cmd_opt, opt):
  if cmd_opt['function'] is not None:
    opt['function'] = cmd_opt['function']
  if cmd_opt['block'] is not None:
    opt['block'] = cmd_opt['block']
  if cmd_opt['attention_type'] != 'scaled_dot':
    opt['attention_type'] = cmd_opt['attention_type']
  if cmd_opt['self_loop_weight'] is not None:
    opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  if cmd_opt['method'] is not None:
    opt['method'] = cmd_opt['method']
  if cmd_opt['step_size'] != 1:
    opt['step_size'] = cmd_opt['step_size']
  if cmd_opt['time'] is not None:
    opt['time'] = cmd_opt['time']
  if cmd_opt['epoch'] is not None:
    opt['epoch'] = cmd_opt['epoch']
  if not cmd_opt['not_lcc']:
    opt['not_lcc'] = False
  if cmd_opt['num_splits'] != 1:
    opt['num_splits'] = cmd_opt['num_splits']
  if cmd_opt['dropout'] is not None:
    opt['dropout'] = cmd_opt['dropout']
  if cmd_opt['hidden_dim'] is not None:
    opt['hidden_dim'] = cmd_opt['hidden_dim']
  if cmd_opt['decay'] is not None:
    opt['decay'] = cmd_opt['decay']
  if cmd_opt['self_loop_weight'] is not None:
    opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  if cmd_opt['edge_homo']  != 0:
    opt['edge_homo'] = cmd_opt['edge_homo']
  if cmd_opt['use_mlp'] is not None:
    opt['use_mlp'] = cmd_opt['use_mlp']
  if cmd_opt['data_norm'] is not None:
    opt['data_norm'] = cmd_opt['data_norm']

  if cmd_opt['lr'] is not None:
    opt['lr'] = cmd_opt['lr']
  if cmd_opt['input_dropout'] is not None:
    opt['input_dropout'] = cmd_opt['input_dropout']

  if cmd_opt['patience'] is not None:
    opt['patience'] = cmd_opt['patience']
  if cmd_opt['max_nfe'] is not None:
    opt['max_nfe'] = cmd_opt['max_nfe']


def set_seed(seed=123):
  random.seed(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

def get_optimizer_group(optimizer_name, grouped_parameters, **kwargs):
  if optimizer_name == 'adam':
    optimizer = torch.optim.Adam(grouped_parameters, **kwargs)
  elif optimizer_name == 'sgd':
    optimizer = torch.optim.SGD(grouped_parameters, **kwargs)
  elif optimizer_name == 'rmsprop':
    optimizer = torch.optim.RMSprop(grouped_parameters, **kwargs)
  elif optimizer_name == 'adagrad':
    optimizer = torch.optim.Adagrad(grouped_parameters, **kwargs)
  elif optimizer_name == 'adamax':
    optimizer = torch.optim.Adamax(grouped_parameters, **kwargs)
  # Add more optimizers here as needed
  else:
    raise ValueError("Invalid optimizer name")

  return optimizer
def combined_optimizer(model, opt):
  parameters_alphaode = [p for name, p in model.named_parameters() if p.requires_grad and 'alpha_ode' in name]
  parameters_other = [p for name, p in model.named_parameters() if p.requires_grad and 'alpha_ode' not in name]

  grouped_parameters = [
    {'params': parameters_other, 'lr': opt['lr'], 'weight_decay': opt['decay']},
    {'params': parameters_alphaode, 'lr': opt['lr'], 'weight_decay': opt['decay']}
  ]

  optimizer = get_optimizer_group(opt['optimizer'], grouped_parameters)
  return optimizer

def main(opt,split):


  set_seed(opt['seed'])
  dataset = get_dataset(opt, f'{ROOT_DIR}/data', opt['not_lcc'],split)
  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if opt['cuda'] >-1 :
    device = torch.device('cuda:' + str(opt['cuda']) if torch.cuda.is_available() else 'cpu')
  else:
    device = 'cpu'

  num_features = dataset.num_features
  num_classes = dataset.num_classes
  opt['num_classes'] = num_classes
  num_nodes = dataset.data.x.shape[0]
  opt['num_nodes'] = num_nodes

  print("num of nodes: ", num_nodes)
  print("num of features: ", num_features)
  print("num of classes: ", num_classes)

  model = GNN(opt, dataset, device).to(device)
  #
  if not opt['planetoid_split'] and opt['dataset'] in ['Cora','Citeseer','Pubmed']:
    dataset.data = set_train_val_test_split(opt['seed'], dataset.data, num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)

  data = dataset.data.to(device)



  data.edge_index = to_undirected(data.edge_index)
  print("num of train samples: ", len(torch.nonzero(data.train_mask,as_tuple=True)[0]))
  print("num of val samples: ", len(torch.nonzero(data.val_mask,as_tuple=True)[0]))
  print("num of test samples: ", len(torch.nonzero(data.test_mask,as_tuple=True)[0]))

  parameters = [p for p in model.parameters() if p.requires_grad]
  print_model_params(model)
  optimizer = combined_optimizer(model, opt)


  best_time = best_epoch = train_acc = val_acc = test_acc = 0

  this_test = test_OGB if opt['dataset'] == 'ogbn-arxiv' else test
  counter = 0
  for epoch in range(1, opt['epoch']):
    start_time = time.time()



    loss = train(model, optimizer, data)
    tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, opt)

    best_time = opt['time']
    if tmp_val_acc > val_acc:
      best_epoch = epoch
      train_acc = tmp_train_acc
      val_acc = tmp_val_acc
      test_acc = tmp_test_acc
      best_time = opt['time']
      counter = 0
    else:
      counter = counter + 1
      if counter == opt['patience']:
        break

    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best time: {:.4f}'

    print(log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, tmp_train_acc, tmp_val_acc, tmp_test_acc, best_time))
  print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d} and best time {:03f}'.format(val_acc, test_acc,
                                                                                                     best_epoch,
                                                                                                     best_time))
  return train_acc, val_acc, test_acc,opt


if __name__ == '__main__':



  args = parser.parse_args()

  cmd_opt = vars(args)

  try:
    best_opt = best_params_dict[cmd_opt['dataset']]
    opt = {**cmd_opt, **best_opt}
    merge_cmd_args(cmd_opt, opt)
  except KeyError:
    opt = cmd_opt

  best = []
  timestr = time.strftime("%H%M%S")

  # mkdir for log
  if not os.path.exists("log_frac"):
    os.makedirs("log_frac")
  filename = "log_frac/" + str(args.dataset) + str(args.method) + str(args.function) + str(args.block) + str(
    args.time) + timestr + ".txt"
  command_args = " ".join(sys.argv)
  with open(filename, 'a') as f:
    json.dump(command_args, f)
    f.write("\n")

  for i in range(opt['runtime']):
    opt['seed'] = opt['seed'] + i
    train_acc, val_acc, test_acc, opt_final = main(opt,i)

    best.append(test_acc)
    with open(filename, 'a') as f:
      json.dump(test_acc, f)
      f.write("\n")
    print("test acc: ", best)
    # opt['seed'] += 1
  print('Mean test accuracy: ', np.mean(np.array(best) * 100), 'std: ', np.std(np.array(best) * 100))
  print("test acc: ", best)

  with open(filename, 'a') as f:
    f.write(str(np.mean(np.array(best) * 100)))
    f.write(",")
    f.write(str(np.std(np.array(best) * 100)))
    f.write("\n")
    json.dump(opt_final, f, indent=2)
  # change file name to include best test acc
  os.rename(filename, filename[:-4] + str(np.mean(np.array(best) * 100)) + ".txt")






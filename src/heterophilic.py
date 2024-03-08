"""
Code taken from https://github.com/jianhao2016/GPRGNN/blob/master/src/dataset_utils.py
"""

import torch
import numpy as np
import os.path as osp

from typing import Optional, Callable, List, Union
from torch_sparse import SparseTensor, coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import remove_self_loops
from itertools import repeat
from utils import ROOT_DIR
from torch_geometric.io import read_txt_array
import sys
import pickle
import os
class Actor(InMemoryDataset):
  r"""The actor-only induced subgraph of the film-director-actor-writer
  network used in the
  `"Geom-GCN: Geometric Graph Convolutional Networks"
  <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
  Each node corresponds to an actor, and the edge between two nodes denotes
  co-occurrence on the same Wikipedia page.
  Node features correspond to some keywords in the Wikipedia pages.
  The task is to classify the nodes into five categories in term of words of
  actor's Wikipedia.

  Args:
      root (string): Root directory where the dataset should be saved.
      transform (callable, optional): A function/transform that takes in an
          :obj:`torch_geometric.data.Data` object and returns a transformed
          version. The data object will be transformed before every access.
          (default: :obj:`None`)
      pre_transform (callable, optional): A function/transform that takes in
          an :obj:`torch_geometric.data.Data` object and returns a
          transformed version. The data object will be transformed before
          being saved to disk. (default: :obj:`None`)
  """

  url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

  def __init__(self, root: str, transform: Optional[Callable] = None,
               pre_transform: Optional[Callable] = None):
    super().__init__(root, transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_file_names(self) -> List[str]:
    return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'
            ] + [f'film_split_0.6_0.2_{i}.npz' for i in range(10)]

  @property
  def processed_file_names(self) -> str:
    return 'data.pt'

  def download(self):
    for f in self.raw_file_names[:2]:
      download_url(f'{self.url}/new_data/film/{f}', self.raw_dir)
    for f in self.raw_file_names[2:]:
      download_url(f'{self.url}/splits/{f}', self.raw_dir)

  def process(self):

    with open(self.raw_paths[0], 'r') as f:
      data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

      rows, cols = [], []
      for n_id, col, _ in data:
        col = [int(x) for x in col.split(',')]
        rows += [int(n_id)] * len(col)
        cols += col
      x = SparseTensor(row=torch.tensor(rows), col=torch.tensor(cols))
      x = x.to_dense()

      y = torch.empty(len(data), dtype=torch.long)
      for n_id, _, label in data:
        y[int(n_id)] = int(label)

    with open(self.raw_paths[1], 'r') as f:
      data = f.read().split('\n')[1:-1]
      data = [[int(v) for v in r.split('\t')] for r in data]
      edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
      # Remove self-loops
      edge_index, _ = remove_self_loops(edge_index)
      # Make the graph undirected
      edge_index = to_undirected(edge_index)
      edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

    train_masks, val_masks, test_masks = [], [], []
    for f in self.raw_paths[2:]:
      tmp = np.load(f)
      train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
      val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
      test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
    train_mask = torch.stack(train_masks, dim=1)
    val_mask = torch.stack(val_masks, dim=1)
    test_mask = torch.stack(test_masks, dim=1)

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                val_mask=val_mask, test_mask=test_mask)
    data = data if self.pre_transform is None else self.pre_transform(data)
    torch.save(self.collate([data]), self.processed_paths[0])


class WikipediaNetwork(InMemoryDataset):
  r"""The Wikipedia networks introduced in the
  `"Multi-scale Attributed Node Embedding"
  <https://arxiv.org/abs/1909.13021>`_ paper.
  Nodes represent web pages and edges represent hyperlinks between them.
  Node features represent several informative nouns in the Wikipedia pages.
  The task is to predict the average daily traffic of the web page.

  Args:
      root (string): Root directory where the dataset should be saved.
      name (string): The name of the dataset (:obj:`"chameleon"`,
          :obj:`"crocodile"`, :obj:`"squirrel"`).
      geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
          pre-processed data as introduced in the `"Geom-GCN: Geometric
          Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
          in which the average monthly traffic of the web page is converted
          into five categories to predict.
          If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
          available.
      transform (callable, optional): A function/transform that takes in an
          :obj:`torch_geometric.data.Data` object and returns a transformed
          version. The data object will be transformed before every access.
          (default: :obj:`None`)
      pre_transform (callable, optional): A function/transform that takes in
          an :obj:`torch_geometric.data.Data` object and returns a
          transformed version. The data object will be transformed before
          being saved to disk. (default: :obj:`None`)

  """

  def __init__(self, root: str, name: str,
               transform: Optional[Callable] = None,
               pre_transform: Optional[Callable] = None):
    self.name = name.lower()
    assert self.name in ['chameleon', 'squirrel']
    super().__init__(root, transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_dir(self) -> str:
    return osp.join(self.root, self.name, 'raw')

  @property
  def processed_dir(self) -> str:
    return osp.join(self.root, self.name, 'processed')

  @property
  def raw_file_names(self) -> Union[str, List[str]]:
    return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

  @property
  def processed_file_names(self) -> str:
    return 'data.pt'

  def download(self):
    pass

  def process(self):
    with open(self.raw_paths[0], 'r') as f:
      data = f.read().split('\n')[1:-1]
    x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
    x = torch.tensor(x, dtype=torch.float)
    y = [int(r.split('\t')[2]) for r in data]
    y = torch.tensor(y, dtype=torch.long)

    with open(self.raw_paths[1], 'r') as f:
      data = f.read().split('\n')[1:-1]
      data = [[int(v) for v in r.split('\t')] for r in data]
    edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
    # Remove self-loops
    edge_index, _ = remove_self_loops(edge_index)
    # Make the graph undirected
    edge_index = to_undirected(edge_index)
    edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)

    if self.pre_transform is not None:
      data = self.pre_transform(data)

    torch.save(self.collate([data]), self.processed_paths[0])


class WebKB(InMemoryDataset):
  r"""The WebKB datasets used in the
  `"Geom-GCN: Geometric Graph Convolutional Networks"
  <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
  Nodes represent web pages and edges represent hyperlinks between them.
  Node features are the bag-of-words representation of web pages.
  The task is to classify the nodes into one of the five categories, student,
  project, course, staff, and faculty.
  Args:
      root (string): Root directory where the dataset should be saved.
      name (string): The name of the dataset (:obj:`"Cornell"`,
          :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
      transform (callable, optional): A function/transform that takes in an
          :obj:`torch_geometric.data.Data` object and returns a transformed
          version. The data object will be transformed before every access.
          (default: :obj:`None`)
      pre_transform (callable, optional): A function/transform that takes in
          an :obj:`torch_geometric.data.Data` object and returns a
          transformed version. The data object will be transformed before
          being saved to disk. (default: :obj:`None`)
  """

  url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
         'master/new_data')

  def __init__(self, root, name, transform=None, pre_transform=None):
    self.name = name.lower()
    assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

    super(WebKB, self).__init__(root, transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_dir(self):
    return osp.join(self.root, self.name, 'raw')

  @property
  def processed_dir(self):
    return osp.join(self.root, self.name, 'processed')

  @property
  def raw_file_names(self):
    return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

  @property
  def processed_file_names(self):
    return 'data.pt'

  def download(self):
    for name in self.raw_file_names:
      download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

  def process(self):
    with open(self.raw_paths[0], 'r') as f:
      data = f.read().split('\n')[1:-1]
      x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
      x = torch.tensor(x, dtype=torch.float32)

      y = [int(r.split('\t')[2]) for r in data]
      y = torch.tensor(y, dtype=torch.long)

    with open(self.raw_paths[1], 'r') as f:
      data = f.read().split('\n')[1:-1]
      data = [[int(v) for v in r.split('\t')] for r in data]
      edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
      edge_index = to_undirected(edge_index)
      # We also remove self-loops in these datasets in order not to mess up with the Laplacian.
      edge_index, _ = remove_self_loops(edge_index)
      edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)
    data = data if self.pre_transform is None else self.pre_transform(data)
    torch.save(self.collate([data]), self.processed_paths[0])

  def __repr__(self):
    return '{}()'.format(self.name)


def index_to_mask(index, size):
  mask = torch.zeros(size, dtype=torch.bool, device=index.device)
  mask[index] = 1
  return mask


def generate_random_splits(data,  train_rate=0.6, val_rate=0.2,Flag = 0):
  """Generates training, validation and testing masks for node classification tasks."""
  num_classes = len(torch.unique(data.y))
  percls_trn = int(round(train_rate * len(data.y) / num_classes))
  val_lb = int(round(val_rate * len(data.y)))

  indices = []
  for i in range(num_classes):
    index = (data.y == i).nonzero().view(-1)
    index = index[torch.randperm(index.size(0))]
    indices.append(index)

  train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

  if Flag == 0:

      rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
      rest_index = rest_index[torch.randperm(rest_index.size(0))]

      data.train_mask = index_to_mask(train_index, size=data.num_nodes)
      data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
      data.test_mask = index_to_mask(rest_index[val_lb:], size=data.num_nodes)
  else:
      val_index = torch.cat([i[percls_trn:percls_trn + val_lb]
                             for i in indices], dim=0)
      rest_index = torch.cat([i[percls_trn + val_lb:] for i in indices], dim=0)
      rest_index = rest_index[torch.randperm(rest_index.size(0))]

      data.train_mask = index_to_mask(train_index, size=data.num_nodes)
      data.val_mask = index_to_mask(val_index, size=data.num_nodes)
      data.test_mask = index_to_mask(rest_index, size=data.num_nodes)






  y_train = data.y[data.train_mask]
  y_test = data.y[data.test_mask]
  y_val = data.y[data.val_mask]

  indices_train = []
  for i in range(num_classes):
    index = (y_train == i).nonzero().view(-1)
    index = index[torch.randperm(index.size(0))]
    indices_train.append(len(index))
  print("label distribution of train: ", indices_train)

  indices_test = []

  for i in range(num_classes):
    index = (y_test == i).nonzero().view(-1)
    index = index[torch.randperm(index.size(0))]
    indices_test.append(len(index))
  print("label distribution of test: ", indices_test)

  indices_val = []

  for i in range(num_classes):
    index = (y_val == i).nonzero().view(-1)
    index = index[torch.randperm(index.size(0))]
    indices_val.append(len(index))
  print("label distribution of val: ", indices_val)

  return data


def get_fixed_splits(data, dataset_name,path, seed):
  #todo just added this to test sheaf experiments. Remove when done
  dataset_str = dataset_name
  if dataset_name == 'gg_cora':
    dataset_str = 'cora'
  if dataset_name == 'cora_acm':
      dataset_str = 'cora'
  if dataset_name == 'citeseer_acm':
      dataset_str = 'citeseer'
  if dataset_name == 'pubmed_acm':
      dataset_str = 'pubmed'
  # if dataset_name in ['chameleon', 'squirrel']:
  #   split_path = f'{path}/{dataset_name}/geom_gcn/raw/{dataset_name}_split_0.6_0.2_{seed}.npz'
  # else:
  #
  #   split_path = f'{path}/{dataset_name}/raw/{dataset_name}_split_0.6_0.2_{seed}.npz'
  split_path = f'{path}/{dataset_name}/raw/{dataset_str}_split_0.6_0.2_{seed}.npz'

  with np.load(split_path) as splits_file:
    train_mask = splits_file['train_mask']
    val_mask = splits_file['val_mask']
    test_mask = splits_file['test_mask']

  data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
  data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
  data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

  y_train = data.y[train_mask]
  y_test = data.y[test_mask]
  y_val = data.y[val_mask]

  indices_train = []
  num_classes = len(torch.unique(data.y))
  for i in range(num_classes):
    index = (y_train == i).nonzero().view(-1)
    index = index[torch.randperm(index.size(0))]
    indices_train.append(len(index))
  print("label distribution of train: ", indices_train)

  indices_test = []

  for i in range(num_classes):
    index = (y_test == i).nonzero().view(-1)
    index = index[torch.randperm(index.size(0))]
    indices_test.append(len(index))
  print("label distribution of test: ", indices_test)

  indices_val = []

  for i in range(num_classes):
    index = (y_val == i).nonzero().view(-1)
    index = index[torch.randperm(index.size(0))]
    indices_val.append(len(index))
  print("label distribution of val: ", indices_val)

  # label_train = torch.nonzero(y_train,as_tuple=True)

  # Remove the nodes that the label vectors are all zeros, they aren't assigned to any class
  if dataset_name in {'cora_acm', 'citeseer_acm', 'pubmed_acm'}:
    # data.train_mask[data.non_valid_samples] = False
    # data.test_mask[data.non_valid_samples] = False
    # data.val_mask[data.non_valid_samples] = False
    # print("Non zero masks", torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask))
    # print("Nodes", data.x.size(0))
    # print("Non valid", len(data.non_valid_samples))
    print("data.x.size(0): ",data.x.size(0))
    print("torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask): ",torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask))
  else:
    assert torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask) == data.x.size(0)


  return data



class Planetoid2(InMemoryDataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        split (string): The type of dataset split
            (:obj:`"public"`, :obj:`"full"`, :obj:`"random"`).
            If set to :obj:`"public"`, the split will be the public fixed split
            from the
            `"Revisiting Semi-Supervised Learning with Graph Embeddings"
            <https://arxiv.org/abs/1603.08861>`_ paper.
            If set to :obj:`"full"`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"random"` split. (default: :obj:`20`)
        num_val (int, optional): The number of validation samples in case of
            :obj:`"random"` split. (default: :obj:`500`)
        num_test (int, optional): The number of test samples in case of
            :obj:`"random"` split. (default: :obj:`1000`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    # url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root: str, name: str, split: str = "public",
                 num_train_per_class: int = 20, num_val: int = 500,
                 num_test: int = 1000, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.split = split
        assert self.split in ['public', 'full', 'random']

        if split == 'full':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.{self.name}.{name}' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        # for name in self.raw_file_names:
        #     download_url('{}/{}'.format(self.url, name), self.raw_dir)
        pass

    def process(self):
        data = read_planetoid_data2(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'


def read_planetoid_data2(folder, prefix):
  names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
  items = [read_file(folder, prefix, name) for name in names]
  x, tx, allx, y, ty, ally, graph, test_index = items
  train_index = torch.arange(y.size(0), dtype=torch.long)
  val_index = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
  sorted_test_index = test_index.sort()[0]

  if prefix.lower() == 'citeseer':
    # There are some isolated nodes in the Citeseer graph, resulting in
    # none consecutive test indices. We need to identify them and add them
    # as zero vectors to `tx` and `ty`.
    len_test_indices = (test_index.max() - test_index.min()).item() + 1

    tx_ext = torch.zeros(len_test_indices, tx.size(1))
    tx_ext[sorted_test_index - test_index.min(), :] = tx
    ty_ext = torch.zeros(len_test_indices, ty.size(1))
    ty_ext[sorted_test_index - test_index.min(), :] = ty

    tx, ty = tx_ext, ty_ext

  if prefix.lower() == 'nell.0.001':
    tx_ext = torch.zeros(len(graph) - allx.size(0), x.size(1))
    tx_ext[sorted_test_index - allx.size(0)] = tx

    ty_ext = torch.zeros(len(graph) - ally.size(0), y.size(1))
    ty_ext[sorted_test_index - ally.size(0)] = ty

    tx, ty = tx_ext, ty_ext

    x = torch.cat([allx, tx], dim=0)
    x[test_index] = x[sorted_test_index]

    # Creating feature vectors for relations.
    row, col, value = SparseTensor.from_dense(x).coo()
    rows, cols, values = [row], [col], [value]

    mask1 = index_to_mask(test_index, size=len(graph))
    mask2 = index_to_mask(torch.arange(allx.size(0), len(graph)),
                          size=len(graph))
    mask = ~mask1 | ~mask2
    isolated_index = mask.nonzero(as_tuple=False).view(-1)[allx.size(0):]

    rows += [isolated_index]
    cols += [torch.arange(isolated_index.size(0)) + x.size(1)]
    values += [torch.ones(isolated_index.size(0))]

    x = SparseTensor(row=torch.cat(rows), col=torch.cat(cols),
                     value=torch.cat(values))
  else:
    x = torch.cat([allx, tx], dim=0)
    x[test_index] = x[sorted_test_index]

  y = torch.cat([ally, ty], dim=0).max(dim=1)[1]
  y[test_index] = y[sorted_test_index]

  train_mask = index_to_mask(train_index, size=y.size(0))
  val_mask = index_to_mask(val_index, size=y.size(0))
  test_mask = index_to_mask(test_index, size=y.size(0))

  edge_index = edge_index_from_dict(graph, num_nodes=y.size(0))

  data = Data(x=x, edge_index=edge_index, y=y)
  data.train_mask = train_mask
  data.val_mask = val_mask
  data.test_mask = test_mask

  return data



def edge_index_from_dict(graph_dict, num_nodes=None, do_coalesce=True):
    """
    copied from the pytorch_geometric repository
    """
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    print("# graph_dict size", len(graph_dict.items()))
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    if do_coalesce:
        # NOTE: There are some duplicated edges and self loops in the datasets.
        #       Other implementations do not remove them!
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    print(edge_index.shape)
    return edge_index



def read_file(folder, prefix, name):
    """
    copied from the following repository
    https://github.com/pyg-team/pytorch_geometric/blob/614e3a6402a542bfdd0b9ecf4ec37607b4b61756/torch_geometric/io/planetoid.py#L17
    """
    path = osp.join(folder, 'ind.{}.{}'.format(prefix, name))

    if name == 'test.index':
        return read_txt_array(path, dtype=torch.long)

    with open(path, 'rb') as f:
        if sys.version_info > (3, 0):
            out = pickle.load(f, encoding='latin1')
        else:
            out = pickle.load(f)

    if name == 'graph':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    out = torch.Tensor(out)
    return out


def random_disassortative_splits(labels, num_classes):
    # * 0.6 labels for training
    # * 0.2 labels for validation
    # * 0.2 labels for testing
    labels, num_classes = labels.cpu(), num_classes.cpu().numpy()
    indices = []
    for i in range(num_classes):
      index = torch.nonzero((labels == i)).view(-1)
      index = index[torch.randperm(index.size(0))]
      indices.append(index)
    percls_trn = int(round(0.6 * (labels.size()[0] / num_classes)))
    val_lb = int(round(0.2 * labels.size()[0]))
    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=labels.size()[0])
    val_mask = index_to_mask(rest_index[:val_lb], size=labels.size()[0])
    test_mask = index_to_mask(rest_index[val_lb:], size=labels.size()[0])

    return train_mask, val_mask, test_mask


from deeprobust.graph.data import Dataset
import os.path as osp
import numpy as np


class CustomDataset_cora(Dataset):
    def __init__(self, root, name, setting='gcn', seed=None, require_mask=False):
        '''
        Adopted from https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/data/dataset.py
        '''
        self.name = name.lower()
        self.setting = setting.lower()

        self.seed = seed
        self.url = None
        self.root = osp.expanduser(osp.normpath(root))
        self.data_folder = osp.join(root, self.name)
        self.data_filename = self.data_folder + '.npz'
        # Make sure dataset file exists
        assert osp.exists(self.data_filename), f"{self.data_filename} does not exist!"
        self.require_mask = require_mask

        self.require_lcc = True if setting == 'nettack' else False
        self.adj, self.features, self.labels = self.load_data()
        self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test()
        if self.require_mask:
            self.get_mask()

    def get_adj(self):
        adj, features, labels = self.load_npz(self.data_filename)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1

        if self.require_lcc:
            lcc = self.largest_connected_components(adj)

            # adj = adj[lcc][:, lcc]
            adj_row = adj[lcc]
            adj_csc = adj_row.tocsc()
            adj_col = adj_csc[:, lcc]
            adj = adj_col.tolil()

            features = features[lcc]
            labels = labels[lcc]
            assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

        return adj, features, labels

    def get_train_val_test(self):
        if self.setting == "exist":
            with np.load(self.data_filename) as loader:
                idx_train = loader["idx_train"]
                idx_val = loader["idx_val"]
                idx_test = loader["idx_test"]
            return idx_train, idx_val, idx_test
        else:
            return super().get_train_val_test()
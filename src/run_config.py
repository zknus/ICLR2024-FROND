import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--use_cora_defaults', action='store_true',
                  help='Whether to run with best params for cora. Overrides the choice of dataset')
parser.add_argument('--cuda', default=1, type=int)
# data args
parser.add_argument('--dataset', type=str, default='twitch-gamer',
                  help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv,chameleon, squirrel,'
                       'wiki-cooc, roman-empire, amazon-ratings, minesweeper, workers, questions',)
parser.add_argument('--data_norm', type=str, default='gcn',
                  help='rw for random walk, gcn for symmetric gcn norm')
parser.add_argument('--self_loop_weight', default=1,type=float, help='Weight of self-loops.')
parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
parser.add_argument('--geom_gcn_splits', default=True, dest='geom_gcn_splits', action='store_true',
                  help='use the 10 fixed splits from '
                       'https://arxiv.org/abs/2002.05287')
parser.add_argument('--num_splits', type=int, dest='num_splits', default=1,
                  help='the number of splits to repeat the results on')
parser.add_argument('--label_rate', type=float, default=0.5,
                  help='% of training labels to use when --use_labels is set.')
parser.add_argument('--planetoid_split', action='store_true',
                  help='use planetoid splits for Cora/Citeseer/Pubmed')

parser.add_argument('--random_splits',action='store_true',help='fixed_splits')

parser.add_argument('--edge_homo', type=float, default=0.0, help="edge_homo")


# GNN args
parser.add_argument('--hidden_dim',default=64, type=int,  help='Hidden dimension.')
parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                  help='Add a fully connected layer to the decoder.')
parser.add_argument('--input_dropout', type=float,default=0.2,  help='Input dropout rate.')
parser.add_argument('--dropout', type=float,default=0.4, help='Dropout rate.')
parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--decay', type=float,default=0.0001,  help='Weight decay for optimization')
parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                  help='apply sigmoid before multiplying by alpha')
parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
parser.add_argument('--block', default='constant_graph',type=str,  help='constant, mixed, attention, hard_attention')
parser.add_argument('--function',default='laplacian', type=str, help='laplacian, transformer, dorsey, GAT')
parser.add_argument('--use_mlp', type=bool,
                  help='Add a fully connected layer to the encoder.')
parser.add_argument('--add_source', dest='add_source', action='store_true',
                  help='If try get rid of alpha param and the beta*x0 source term')
parser.add_argument('--cgnn', dest='cgnn', action='store_true', help='Run the baseline CGNN model from ICML20')

parser.add_argument('--patience', type=int, default=100, help='Number of training patience per iteration.')

# ODE args
parser.add_argument('--time',default=3,  type=float, help='End time of ODE integrator.')
parser.add_argument('--augment', action='store_true',
                  help='double the length of the feature vector by appending zeros to stabilist ODE learning')
parser.add_argument('--method',default='ceuler',  type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
parser.add_argument('--step_size', type=float, default=1,
                  help='fixed step size when using fixed step solvers e.g. rk4')
parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                  help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                  help='use the adjoint ODE method to reduce memory footprint')
parser.add_argument('--adjoint_step_size', type=float, default=1,
                  help='fixed step size when using fixed step adjoint solvers e.g. rk4')
parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                  help="multiplier for adjoint_atol and adjoint_rtol")
parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
parser.add_argument("--max_nfe", type=int, default=100000000000,
                  help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
parser.add_argument("--no_early", action="store_true",
                  help="Whether or not to use early stopping of the ODE integrator when testing.")
parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
parser.add_argument("--max_test_steps", type=int, default=100,
                  help="Maximum number steps for the dopri5Early test integrator. "
                       "used if getting OOM errors at test time")

# Attention args
parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                  help='slope of the negative part of the leaky relu used in attention')
parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
parser.add_argument('--attention_dim', type=int, default=64,
                  help='the size to project x to before calculating att scores')
parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                  help='apply a feature transformation xW to the ODE')
parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                  help="multiply attention scores by edge weights before softmax")
parser.add_argument('--attention_type', type=str, default="scaled_dot",
                  help="scaled_dot,cosine_sim,pearson, exp_kernel")
parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')



# rewiring args
parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")


parser.add_argument('--alpha_ode', type=float, default=0.5, help='alpha_ode')
parser.add_argument('--runtime', type=int, default=1, help="runtime")
parser.add_argument('--seed', type=int, default=123, help="seed")







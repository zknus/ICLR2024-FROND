

This repository contains the code for our ICLR 2024 accepted **Spotlight** paper, *[Unleashing the Potential of Fractional Calculus in Graph Neural Networks with FROND](https://openreview.net/forum?id=wcka3bd7P4)*.

## Table of Contents

- [Requirements](#requirements)
- [Reproducing Results](#reproducing-results)
- [Reference](#reference)
- [Citation](#citation)

## Requirements

To install the required dependencies, refer to the environment.yaml file

## Reproducing Results
To run our code, go to the /src folder.


```bash
python run_GNN_frac_all.py 
--dataset  Cora, Citeseer, Pubmed, CoauthorCS, CoauthorPhy, Computers, Photo
--function laplacian/ transformer
--block constant_frac/ att_frac
--method predictor/ predictor_corrector
--alpha_ode  between (0,1] the value of beta in the paper
--time     integration time
--step_size  
```

## Reference 

Our code is developed based on the following repo:

The FDE solver is from [torchfde](https://github.com/zknus/torchfde).  

The graph neural ODE model is based on the  [GRAND](https://github.com/twitter-research/graph-neural-pde), [GraphCON](https://github.com/tk-rusch/GraphCON), and [GraphCDE](https://github.com/zknus/Graph-Diffusion-CDE)  framework.  


## Citation 

If you find our work useful, please cite us as follows:
```
@INPROCEEDINGS{KanZhaDin:C24,
    author = {Qiyu Kang and Kai Zhao and Qinxu Ding and Feng Ji and Xuhao Li and Wenfei Liang and Yang Song and Wee Peng Tay},
    title={Unleashing the Potential of Fractional Calculus in Graph Neural Networks with {FROND}},
    booktitle={Proc. International Conference on Learning Representations},
    year={2024},
    address = {Vienna, Austria},
    note ={\textbf{spotlight}},
}
```






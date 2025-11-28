# Reinterpreting Hypergraph Kernels: Insights Through Homomorphism Analysis

This repository contains the code for the paper "" published in IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 2024 by Yifan Zhang, Shaoyi Du*, [Yifan Feng](https://fengyifan.site/), Shihui Ying, and Yue Gao. The paper is available at [here](https://ieeexplore.ieee.org/document/11159289).

![algorithm](doc/alg.jpg)

<!-- ## Abstract
Designing expressive hypergraph kernels that can effectively capture high-order structural information is a fundamental challenge in hypergraph learning. In this paper, we propose a novel comparison framework based on hypergraph homomorphisms to evaluate and compare the expressive ability of existing hypergraph kernels. We revisit classical kernels such as Hypergraph Weisfeiler-Lehman (HG WL) and Hypergraph Rooted kernels, providing theoretical conditions under which they fail to distinguish non-isomorphic hypergraphs. Motivated by these insights, we introduce the Hypergraph Subtree-Cycle Kernel, which augments subtree-based features with cycle-based structural patterns to enhance expressiveness. We propose two variants: HG SCKernelv1 and HG SCKernelv2. Extensive experiments on five graph and ten hypergraph classification benchmarks demonstrate the superior performance of our methods, confirming the effectiveness of integrating homomorphism-guided design into hypergraph kernels.
. -->

## Introduction
In this repository, we provide our implementation of HG SKernel and some compared methods including Graph Subtree Kernel, Graphlet Kernel, Hypergraph Directed Line Kernel, and Hypergraph Rooted Kernel. The implementation is based on the following libraries:
* [python 3.9](https://www.python.org/): basic programming language.
* [dhg 0.9.3](https://github.com/iMoonLab/DeepHypergraph): for hypergraph representation and learning. 
* [torch 1.12.1](https://pytorch.org/): for computation.
* [hydra-core 1.3.2](https://hydra.cc/docs/intro/): for configuration and multi-run management.
* [scikit-multilearn 0.2.0](http://scikit.ml/): for multi-label learning.

## Installation
1. Clone this repository.
2. Install the required libraries.
``` bash
pip install -r requirements.txt
```

## Usage
Modify the `root` path in `ml_config.yaml` to the absolute path of the `data` folder in this repository. Then, run the following command to reproduce the results in the paper:
``` bash
python ml_main.py
```

You can change the name of `model` and `dataset` in `ml_config.yaml` to reproduce the results of other models and datasets. All available models and datasets are listed in the following:

**Models**
- `graphlet_sampling`: [Efficient
graphlet kernels for large graph comparison. PMLR 2009](https://proceedings.mlr.press/v5/shervashidze09a/shervashidze09a.pdf).
- `graph_subtree`: [Fast subtree kernels on graphs. NIPS 2009](https://is.mpg.de/fileadmin/user_upload/files/publications/NIPS2009-Shervashidze_6080[0].pdf).
- `hypergraph_rooted`: [Learning from interpretations: a rooted
kernel for ordered hypergraphs. ICML 2007](https://icml.cc/imls/conferences/2007/proceedings/papers/467.pdf).
- `hypergraph_directed_line`: [A Hypergraph Kernel from Isomorphism Tests. ICPR 2014](https://ieeexplore.ieee.org/document/6977378).


**Datasets**
- Graph Classification Datasets: `RG_macro`, `RG_sub`, `IMDBBINARY`, `IMDBMULTI`, `MUTAG`, `NCI1`, `PROTEINS`
- Hypergraph Classification Datasets: `RHG_3`, `RHG_10`, `RHG_table`, `RHG_pyramid`, `IMDB_dir_form`, `IMDB_dir_genre`, `IMDB_wri_form`, `IMDB_wri_genre`, `IMDB_dir_genre_m`, `IMDB_wri_genre_m`, `stream_player`, `twitter_friend`

## Citation
If you find this repository useful in your research, please cite our following papers:
```
@article{feng2024hypergraph,
  title={Hypergraph Isomorphism Computation},
  author={Feng, Yifan and Han, Jiashu and Ying, Shihui and Gao, Yue},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}

@article{gao2022hgnn+,
  title={HGNN+: General hypergraph neural networks},
  author={Gao, Yue and Feng, Yifan and Ji, Shuyi and Ji, Rongrong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={45},
  number={3},
  pages={3181--3199},
  year={2022},
  publisher={IEEE}
}
```


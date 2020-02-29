## GENERATIVE GRAPH CONVOLUTIONAL NETWORK FOR GROWING GRAPHS (ICASSP 2019)

#### Authors: Da Xu*, Chuanwei Ruan*, Kamiya Motwani, Sushant Kumar, Evren Korpeoglu, Kannan Achan

#### Please contact Da.Xu@walmartlabs.com or Chuanwei.Ruan@walmartlabs.com for questions.

### Introduction
Modeling generative process of growing graphs has wide applications in social networks and recommendation systems. Despite the emerging literature in learning graph representation and graph generation, most of them can not handle isolated new nodes without nontrivial modifications. The challenge arises due to the fact that learning to generate representations for nodes in observed graph relies heavily on topological features, whereas for new nodes only node attributes are available. 

Here we propose a unified generative graph convolutional network that learns node representations for all nodes adaptively in a generative model framework, by sampling graph generation sequences constructed from observed graph data. We optimize over a variational lower bound that consists of a graph reconstruction term and an adaptive Kullback-Leibler divergence regularization term. 

![illustration](Illustration.png?raw=true "workflow visualization")


### Inductive representation learning on temporal graphs
In our [Inductive Representation Learning on Temporal Graphs (ICLR 2020)](https://openreview.net/pdf?id=rJeW1yHYwH) paper, we discuss how to learning node embeddings inductively on temporal graphs, which is an extension to the growing graphs discussed in this paper. The implementation is also avaiable at the [github page](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs).

### Datasets
The public dataset of <strong>Cora</strong>, <strong>Citeseer</strong> and <strong>Pubmed</strong> are provided in the data repository. The raw-formated data are in the data/test folder, and the .npz data files in the data repository have been preprocessed into sparisity format.

### Running experiments

* For link prediction tasks on isolated new nodes:
```bash
python ./scr/run_iso_nodes.py --data_set [cora, citeseer, pubmed]
```

* For link prediction on existing nodes:
```bash
python ./src/run_exist_nodes.py --data_set [cora, citeseer, pubmed]
```

### Cite

```
@INPROCEEDINGS{8682360, 
author={D. {Xu} and C. {Ruan} and K. {Motwani} and E. {Korpeoglu} and S. {Kumar} and K. {Achan}}, 
booktitle={ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
title={Generative Graph Convolutional Network for Growing Graphs}, 
year={2019}, 
volume={}, 
number={}, 
pages={3167-3171}, 
keywords={Task analysis;Encoding;Decoding;Standards;Adaptation models;Training;Social networking (online);Graph representation learning;sequential generative model;variational autoencoder;growing graph}, 
doi={10.1109/ICASSP.2019.8682360}, 
ISSN={2379-190X}, 
month={May},}
```

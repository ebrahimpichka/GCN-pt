# Simple PyTorch implementation of Graph Convolutional Networks

PyTorch implementation of Graph Convolutional Networks (GCNs) based on [(Kipf & Welling, ICLR 2017)](https://arxiv.org/abs/1609.02907)

## Main Components

### Graph Convolution Layer:
The Graph Convolution layer module is implemented in `gc_layer.py` file as `GraphConvolution` class. It follows the official PyTorch layer module interface.

This module computes the following expression as part message passing mechanism:

![graph convolution layer](.\images\gclayer.png)

where the `D⁻¹/²ÃD⁻¹/²` part is recieved as an input argument to the `forward` method in **sparse matrix tensor** format and the node features `H` is also recieved as an in*put argument to the `forward` method.

### Graph Convolutional Network (GCN):
The main Graph Convolutional Network module is implemented in `gcn.py` file as `GCN` class. The `forward` method performs a full two-layer GCN operation on the given input features based on [the paper](https://arxiv.org/abs/1609.02907) as shown below:

![graph convolution layer](.\images\gcn.png)

The module requies number of input features (`nfeat`), number of hidden features (`nhid`), number of output classes (`nclass`), dropout rate (`dropout`) and PyTorch device object (`device`) as input args for instantiation.


### Tests:
The implemented GCN is tested on the KarateClub toy dataset from PyG  in the `test.ipynb` notebook.
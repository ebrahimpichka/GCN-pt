import torch
import torch.nn as nn
import torch.nn.functional as F

from gc_layer import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, device):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, device=device)
        self.gc2 = GraphConvolution(nhid, nclass, device=device)
        self.dropout = dropout

    def forward(self, h, adj_mat):
        """computes Z = f(X, A) = LogSoftmax(Â ReLu(ÂXW⁽⁰⁾) W⁽¹⁾) 
            where Â = D⁻¹/²ÃD⁻¹/² + I is the pre-computed adjacency matrix with self-connections added

        Args:
            h (torch.Tensor): input node feature tensor of shape (batch_size, in_features)
            adj_mat (torch.Tensor): sparse adjacency matrix of the undirected graph

        Returns:
            output embeddings h: torch.Tensor:  
        """
        h = F.relu(self.gc1(h, adj_mat))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gc2(h, adj_mat)
        return h
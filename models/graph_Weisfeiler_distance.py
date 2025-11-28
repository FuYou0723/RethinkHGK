import numpy as np
import torch
import networkx as nx
import torch.nn as nn
def weighted_transition_matrix(G,q):
    A=np.asarray(nx.adjacency_matrix(G,weight=None)).todense()
    n=A.shape[0]
    D=np.sum(A,axis=1)
    mask=D==0
    D[mask]=1
    D=D.reshape((A.shape[0],1))
    A=(1-q)*A/D
    A=A+q*np.identity(n)
    single_node_inds=np.nonzero(mask)[0]
    A[single_node_inds,single_node_inds]=1
    return A
class MlpBlock(nn.Module):
    def __init__(self,in_features,
                 out_features,
                 depth_of_mlp,
                 act=nn.functional.relu_) -> None:
        super().__init__()
        self.act=act
        self.convs=nn.ModuleList()
        for _ in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_features,out_features,
                                    kernel_size=1,padding=0,bias=True))
            in_features=out_features
        for conv in self.convs:
            conv.reset_parameters()
            
    def forward(self,inputs):
        out=inputs
        for conv_layer in self.convs:
            out=self.act(conv_layer(out))
        return out
    
class SkipConnection(nn.Module):
    def __init__(self,in_features,
                 out_features) -> None:
        super().__init__()
        self.conv=nn.Conv2d(in_features,out_features,
                            kernel_size=1,padding=0,bias=True)
    def reset_para(self):
        self.conv.reset_parameters()
    def forward(self,in1,in2):
        # in1: N x d1 x m x m
        # in2: N x d2 x m x m
        out=torch.cat((in1,in2),dim=1)
        out=self.conv(out) # N x out_depth x m x m
        return out
    
import torch
import dhg
from tqdm import tqdm
class ShortestPath:
    def __init__(self,normalize=True,D=400) -> None:
        self._normalize=normalize
        self._D=D
        
    def fit_transform(self,g_list):
        num_g=len(g_list)
        self.featureSP=torch.zeros(num_g,self._D) # num_g x _D
        for g_idx,g in enumerate(g_list):
            if isinstance(g['dhg'],dhg.Graph):
                self.featureSP[g_idx,:]=self.floyd_warshall(g['dhg'])
            else:
                self.featureSP[g_idx,:]=self._floyd_warshanll_sparse(g['dhgg'])
            #print(f"graph {g_idx} is ok")
        self.train_ft=torch.cdist(self.featureSP,self.featureSP,p=2) 
        sigma=torch.median(self.train_ft)
        self.train_ft=torch.exp(-self.train_ft/sigma)
        print('training is ok!')
        return self.train_ft

    def transform(self,g_list):
        num_g=len(g_list)
        featureSP=torch.zeros(num_g,self._D) # num_g x _D
        for g_idx,g in enumerate(g_list):
            featureSP[g_idx,:]=self.floyd_warshall(g['dhg'])
        test_ft=torch.cdist(featureSP,self.featureSP,p=2)
        sigma=torch.median(test_ft)
        test_ft=torch.exp(-test_ft/sigma)
        print('testing is ok!')
        return test_ft

    def floyd_warshall(self,g:dhg.Graph):
        '''
        adj: (nxn)
        dist: (nxn)
        '''
        num_v=g.num_v
        dist=g.A.to_dense()
        # 0 用来测图规模，#inf 用来测联通性
        dist[dist==0]=self._D-1 # self._D as inf
        dist.diagonal(0)
        for k in range(num_v):
            for i in range(num_v):
                for j in range(num_v):
                    if dist[i,j]>dist[i,k]+dist[k,j]:
                        dist[i,j]=dist[i,k]+dist[k,j]
        feature_g=torch.histc(dist,bins=self._D,min=0) 

        return feature_g
    def _floyd_warshanll_sparse(self,g:dhg.Graph):
        dist=g.A.clone()
        dist_values=dist._values()
        dist_indices=dist._indices()
        #-----#
        inf_mask=(dist_values==0)&(dist_indices[0]!=dist_indices[1])
        dist_values[inf_mask]=self._D-1
        #zero_mask=(dist_indice[0]==dist_indice[1])
        for i in range(g.num_v):
            dist_values[(dist_indices[0]==i)&(dist_indices[1]==i)]=0
        
        for k in range(g.num_v):
            for i in range(g.num_v):
                for j in range(g.num_v):
                    ik = dist_values[(dist_indices[0] == i) & (dist_indices[1] == k)]
                    kj = dist_values[(dist_indices[0] == k) & (dist_indices[1] == j)]
                    ij = dist_values[(dist_indices[0] == i) & (dist_indices[1] == j)]
                    if len(ik) > 0 and len(kj) > 0 and len(ij) > 0:
                        if ij[0] > ik[0] + kj[0]:
                            # 更新稀疏矩阵中的元素
                            dist_values[(dist_indices[0] == i) & (dist_indices[1] == j)] = ik[0] + kj[0]
                
        # 计算直方图
        hist = torch.histc(dist_values, bins=self._D, min=0, max=self._D-1)
        return hist
    
    def _floyd_warshall(self,A:torch.Tensor):
        '''
        adj: (nxn)
        dist: (nxn)
        '''
        num_v=A.shape[1]
        dist=A.clone()
        for k in range(num_v):
            for i in range(num_v):
                for j in range(num_v):
                    if dist[i,j]>dist[i,k]+dist[k,j]:
                        dist[i,j]=dist[i,k]+dist[k,j]
        return dist    
    
# 示例用法
def test():
    # 创建一个邻接矩阵，使用inf表示没有边
    inf = float('inf')
    adj_matrix = torch.tensor([[0, 3, inf, inf, inf, inf, inf, 2],
                            [3, 0, 1, inf, inf, inf, inf, inf],
                            [inf, 1, 0, 7, inf, 2, inf, inf],
                            [inf, inf, 7, 0, 9, 3, 4, inf],
                            [inf, inf, inf, 9, 0, 2, 4, 2],
                            [inf, inf, 2, 3, 2, 0, 5, inf],
                            [inf, inf, inf, 4, 4, 5, 0, inf],
                            [2, inf, inf, inf, 2, inf, inf, 0]], dtype=torch.float)
    kernel=ShortestPath()

    shortest_paths = kernel._floyd_warshall(adj_matrix)
    shortest_paths=torch.triu(shortest_paths)

    print("The shortest path distance matrix is:\n", shortest_paths)  
    a=torch.histc(shortest_paths,bins=10,min=0,max=9,out=None)
    print(a[1:])
    #print('sum path',shortest_paths.sum(1))  
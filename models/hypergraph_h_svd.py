import torch
import dhg
import math
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
class HypergraphHSVD:
    def __init__(self,k=10,normalize=True,D=4000,gamma=10,p=2) -> None:
        self._step=k
        self._normalize=normalize
        self._D=D
        self._gamma=gamma
        self._w=self._gamma*torch.randn(self._step,self._D)
        self._b=torch.rand(self._D)*2*math.pi
        self._p=p
    def _get_graph_ft(self,hg):
        # Convert hypergraph to incidence matrix
        incidence_matrix = hg.H.to_dense().cpu().detach().numpy()
        v,e =incidence_matrix.shape
        assert isinstance(v,int)
        # Perform SVD        
        try:
            max_k=min(v,e)-1
            safe_k=min(self._step,max_k)
            # 计算最大的k_max个奇异值和对应的奇异向量
            #assert safe_k>0,"saft==0"
            u, s, vt = svds(incidence_matrix, k=safe_k)
            # 选择最小的self.k个奇异值及其对应的左奇异向量
            if u.shape[1] >= self._step:
                u = u[:, :self._step]  # 对应的左奇异向量
            else:
                # 如果奇异值数量不足，则补零
                #s = np.pad(s, (0, self.k - len(s)), 'constant')
                u = np.pad(u, ((0, 0), (0, self._step - u.shape[1])), 'constant')
            
        except Exception as e:
            #print(f"SVD计算失败: {e}")
            #s = np.zeros(self._step)
            u = np.random.randn(incidence_matrix.shape[0], self._step)
        u=u.T
        u=torch.tensor(u.copy(),dtype=torch.float32)
        return u
    def fit_transform(self,hg_list):
        num_g=len(hg_list)
        self.featureMMD=torch.zeros(self._D,num_g) # _dx g.num_v
        for hg_idx,hg in enumerate(hg_list):
            feature_g=self._get_graph_ft(hg['dhg'])
            Z=math.sqrt(2/self._D)*torch.cos(torch.matmul(self._w.t(),feature_g)) # _stepx_D _Dx g.num_v =_stepxg.num_v
            self.featureMMD[:,hg_idx]=torch.matmul(Z,torch.ones(feature_g.shape[1]))/feature_g.shape[1] # _step x g.num_v g.num_vx1
        # 
        self.train_ft=torch.cdist(self.featureMMD.T,self.featureMMD.T,p=2) # (num_g,num_g)
        sigma=torch.median(self.train_ft)+1e-5
        self.train_ft=torch.exp(-self.train_ft/sigma)
        return self.train_ft

    def transform(self,hg_list):
        num_g=len(hg_list)
        featureMMD=torch.zeros(self._D,num_g) # _d x g.num_v
        for hg_idx,hg in enumerate(hg_list):
            feature_g=self._get_graph_ft(hg['dhg']) # 
            Z=math.sqrt(2/self._D)*torch.cos(torch.matmul(self._w.t(),feature_g)) # _stepx_D _stepx g.num_v =_stepxg.num_v
            featureMMD[:,hg_idx]=torch.matmul(Z,torch.ones(feature_g.shape[1]))/feature_g.shape[1] # _step x g.num_v g.num_vx1            
        #
        test_ft=torch.cdist(featureMMD.T,self.featureMMD.T,p=2) # num_g x train_num_g
        sigma=torch.median(test_ft)+1e-5
        test_ft=torch.exp(-test_ft/sigma)
        return test_ft 
import torch
import numpy as np
from collections import defaultdict
import dhg
from dhg import Graph
from models.graph_subtree_kernel import GraphSubtreeKernel

    
class WeightAssigner:
    def __init__(self,p=2) -> None:
        self.p=p
        self.tau=1
        self._mode='ve'
    
    def _e_weight_func(self,src_lbls,dst_lbls,r=0.5):
        return r*(np.array(src_lbls[0])-np.mean(np.array(dst_lbls[1])))**2\
            +(1-r)*(np.array(dst_lbls[0])-np.mean(np.array(src_lbls[0])))**2        
    
    def fit_transform(self,g):
        e_weights=[]
        for edge in g['dhg'].e[0]:
            src_idx,dst_idx=edge
            # all lbl are {cur_lbl},{nbr_lbl}
            src_lbls=g['pre_v_lbl'][src_idx]
            dst_lbls=g['pre_v_lbl'][dst_idx]
            # nbr_lbl
            if self._mode == 'v':
                e_weight=(src_lbls[0]!=dst_lbls[0])+self._minikowsi(src_lbls[1],dst_lbls[1])+self.tau
            elif self._mode == 've':
                e_weight=self._e_weight_func(src_lbls,dst_lbls)
            else:
                e_weight=self._uniform(src_lbls[0],dst_lbls[0])
            e_weights.append(e_weight)
        g['e_weights']=e_weights
        return g

    def _vector(self,A,B):
        '''
        A B are lists,
        '''
        label_to_dict=defaultdict(int)
        for i in A+B:
            if i not in label_to_dict:
                label_to_dict[i]=len(label_to_dict)
        A=np.zeros(len(label_to_dict))
        B=np.zeros(len(label_to_dict))
        for i in A:
            A[label_to_dict[i]]+=1
        for i in B:
            B[label_to_dict[i]]+=1
        return A,B
    
    def _uniform(self,a,b):
        return 1.0
    
    def _minikowsi(self,a,b):
        a,b=self._vector(a,b)
        return np.linalg.norm(a-b,ord=2)
    
class UnionFind:
    def __init__(self,nv) -> None:
        self._parents=list(range(nv))
    def _find(self,x):
        if self._parents[x]!=x:
            self._parents[x]=self._find(self._parents[x])
        return self._parents[x]
        
    def _merge(self,x,y):
        rootx=self._find(x)
        rooty=self._find(y)
        if rootx!=rooty:
            self._parents[rooty]=rootx
    
    def _roots(self):
        for vertex,parent in enumerate(self._parents):
            if vertex==parent:
                yield vertex
        
    def _connect(self,x,y):
        self._find(x)==self._find(y)

class PersistenceCalc:
    def __init__(self) -> None:
        self._pairs=[]
        self._betti=None
        self._vatt=None
    def reset_pfg(self):
        self._pairs=[]
        self._betti=None
        self._vatt=None
    def fit_transform(self,g):
        assert isinstance(g['dhg'],dhg.Graph)
        cyc_edge=[]
        uf=UnionFind(nv=g['dhg'].num_v)
        e_weights=np.array(g['e_weights'])
        e_indices=np.argsort(e_weights,kind='stable') # 升序排列
        for e_idx,e_weight in zip(e_indices,e_weights[e_indices]):
            src,dst=g['dhg'].e[0][e_idx]
            src_pa=uf._find(src)
            dst_pa=uf._find(dst)
            if src_pa == dst_pa:
                cyc_edge.append(e_idx)
                continue
            if src_pa>dst_pa:
                src,dst=dst,src
                src_pa,dst_pa=dst_pa,src_pa
            uf._merge(src,dst)
            src_att=g['v_lbl'][src] if self._vatt is not None else 0           
            self._pairs.append((src_att,e_weight,src_pa)) # pairs
        unpaired_value=e_weights[e_indices[-1]] # 
        for root in uf._roots():
            root_att=g['v_lbl'][root] if self._vatt is not None else 0
            self._pairs.append((root_att,unpaired_value,root))
        self._betti=len(list(uf._roots())) # betti 表示连通性
        return self._pairs,self._betti,cyc_edge #
    
class PersistenceWL:
    def __init__(self,p=2,
                 use_wl=True,
                 use_cyc=True,
                 use_lbl=True,
                 normalize=False) -> None:
        self._p=p
        self._use_cyc=use_cyc
        self._use_wl=use_wl
        self._use_lbl=use_lbl
        self.normalize=normalize
        self.n_iter=2
        self._subtree_map={}
        assert self._use_cyc or self._use_lbl or self._use_wl
        
    def remap(self, g_list, count, drop=False):
        for g_idx, g in enumerate(g_list):
            for v_idx in range(g["num_v"]):
                cur_lbl = g["v_lbl"][v_idx]
                if cur_lbl not in self._subtree_map:
                    if drop:
                        g["v_lbl"][v_idx] = -1
                        continue
                    else:
                        self._subtree_map[cur_lbl] = len(self._subtree_map)
                g["v_lbl"][v_idx] = self._subtree_map[cur_lbl]
                count[g_idx][self._subtree_map[cur_lbl]] += 1
        return g_list, count

    def count2mat(self, count):
        row_idx, col_idx, data = [], [], []
        for idx, g in enumerate(count):
            for lbl, cnt in g.items():
                row_idx.append(idx)
                col_idx.append(lbl)
                data.append(cnt)
        return (
            torch.sparse_coo_tensor(
                torch.tensor([row_idx, col_idx]),
                torch.tensor(data),
                size=(len(count), len(self._subtree_map)),
            )
            .coalesce()
            .float()
        )
    def get_wl_feature(self,g_list):
        self._count = [defaultdict(int) for _ in range(len(g_list))]
        self.remap(g_list, self._count)
        for _ in range(self.n_iter):
            for g in g_list:
                tmp,_tmp = [],[]
                for v_idx in range(g["num_v"]):
                    cur_lbl = g["v_lbl"][v_idx]
                    nbr_lbl = sorted(
                        [g["v_lbl"][u_idx] for u_idx in g["dhg"].nbr_v(v_idx)]
                    )
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                    _tmp.append((cur_lbl,nbr_lbl)) #
                g["v_lbl"] = tmp
                g['pre_v_lbl']=_tmp
                
            self.remap(g_list, self._count)
        self.train_cnt = self.count2mat(self._count)
        self.train_ft = self.train_cnt.mm(self.train_cnt.t()).to_dense()
        if self.normalize:
            self.train_ft_diag = torch.diag(self.train_ft)
            self.train_ft = (
                self.train_ft
                / torch.outer(self.train_ft_diag, self.train_ft_diag).sqrt()
            )
            self.train_ft[torch.isnan(self.train_ft)] = 0
        return self.train_ft

    def _get_wl_feature(self, g_list):
        count = [defaultdict(int) for _ in range(len(g_list))]
        self.remap(g_list, count, drop=True)
        for _ in range(self.n_iter):
            for g in g_list:
                tmp,_tmp = [],[]
                for v_idx in range(g["num_v"]):
                    cur_lbl = g["v_lbl"][v_idx]
                    nbr_lbl = sorted(
                        [g["v_lbl"][u_idx] for u_idx in g["dhg"].nbr_v(v_idx)]
                    )
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                    _tmp.append((cur_lbl,nbr_lbl))
                g["v_lbl"] = tmp
                g["pre_v_lbl"] = _tmp
            self.remap(g_list, count, drop=True)
        test_cnt = self.count2mat(count)
        test_ft = test_cnt.mm(self.train_cnt.t()).to_dense()
        if self.normalize:
            test_ft_diag = torch.sparse.sum(test_cnt * test_cnt, dim=1).to_dense()
            test_ft = test_ft / torch.outer(test_ft_diag, self.train_ft_diag).sqrt()
            test_ft[torch.isnan(test_ft)] = 0
        return test_ft    
    
    def fit_transform(self,g_list):
        num_g=len(g_list)
        wa=WeightAssigner(p=2)
        pfg=PersistenceCalc()
        x_wl=self.get_wl_feature(g_list)
        self.num_labels=self.train_cnt.shape[1]
        num_ft=self._use_cyc*self.num_labels+\
            self._use_lbl*self.num_labels
        if num_ft>0:
            self.feature_ft=torch.zeros((num_g,num_ft))      
        
        for g_idx,g in enumerate(g_list):
            g=wa.fit_transform(g=g)
            pfg.reset_pfg()
            pairs,_betti,cyc_edge=pfg.fit_transform(g) # g is a dict
            
            if self._use_lbl:
                # ver_att, y-e_weight, parent_idx
                x_lbl=torch.zeros(self.num_labels)
                for x,y,c in pairs:
                    label=g['v_lbl'][c] #
                    persistence=abs(x-y)**self._p
                    x_lbl[label]+=persistence
                self.feature_ft[g_idx,:self._use_lbl*self.num_labels]=x_lbl

            if self._use_cyc:
                num_cycle=g['dhg'].num_e-g['num_v']+_betti
                assert num_cycle == len(cyc_edge) # 
                x_cyc=torch.zeros(self.num_labels)
                for e_idx in cyc_edge:
                    src,dst=g['dhg'].e[0][e_idx]
                    e_weight=g['e_weights'][e_idx]
                    x_cyc[g['v_lbl'][src]]+=e_weight**self._p
                    x_cyc[g['v_lbl'][dst]]+=e_weight**self._p
                self.feature_ft[g_idx,self._use_lbl*self.num_labels:]=x_cyc
        if num_ft>0:
            self.feature_g=torch.cdist(self.feature_ft,self.feature_ft) # num_g x num_g
            sigma=torch.median(self.feature_g)
            self.feature_g=torch.exp(-self.feature_g/sigma)  # augment feature   
            #self.feature_g=self.feature_ft@self.feature_ft.t()
            x_wl=x_wl+self.feature_g  # original feature and augment feature
        return x_wl
    
    def transform(self,g_list):
        num_g=len(g_list)
        #wl=GraphSubtreeKernel(normalize=True,n_iter=1)
        wa=WeightAssigner(p=2)
        pfg=PersistenceCalc()
        x_wl=self._get_wl_feature(g_list)

        num_ft=self._use_cyc*self.num_labels+\
            self._use_lbl*self.num_labels
        if num_ft>0:
            feature_g=torch.zeros((num_g,num_ft))
        for g_idx,g in enumerate(g_list):
            g=wa.fit_transform(g=g)
            pfg.reset_pfg()
            pairs,_betti,cyc_edge=pfg.fit_transform(g) # g is a dict
            if self._use_lbl:
                # ver_att, y-e_weight, parent_idx
                x_lbl=torch.zeros(self.num_labels)
                for x,y,c in pairs:
                    label=g['v_lbl'][c] #
                    persistence=abs(x-y)**self._p
                    x_lbl[label]+=persistence
                feature_g[g_idx,:self._use_lbl*self.num_labels]=x_lbl

            if self._use_cyc:
                num_cycle=g['dhg'].num_e-g['num_v']+_betti
                assert num_cycle == len(cyc_edge) # 
                x_cyc=torch.zeros(self.num_labels)
                for e_idx in cyc_edge:
                    src,dst=g['dhg'].e[0][e_idx]
                    e_weight=g['e_weights'][e_idx]
                    x_cyc[g['v_lbl'][src]]+=e_weight**self._p
                    x_cyc[g['v_lbl'][dst]]+=e_weight**self._p
                feature_g[g_idx,self._use_lbl*self.num_labels:]=x_cyc
        if num_ft>0:
            feature_g=torch.cdist(feature_g,self.feature_ft) # num_g x self.num_g
            sigma=torch.median(feature_g)
            feature_g=torch.exp(-feature_g/sigma)  # augment feature
            #feature_g=feature_g@self.feature_ft.t()
            x_wl=x_wl+feature_g   
            #x_wl=torch.cat((x_wl,feature_g),dim=0) # original feature and augment feature
        return x_wl
    
    
      
    
    
        
            
        
    
    
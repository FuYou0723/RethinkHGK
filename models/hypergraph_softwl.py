import torch
import dhg
from collections import defaultdict
import numpy as np
import math 
from typing import List

class UnionFind:
    def __init__(self,num_v) -> None:
        self._parents=list(range(num_v))
    def _find(self,x):
        if self._parents[x]!=x:
            self._parents[x]=self._find(self._parents[x])
        return self._parents[x]
    def _merge(self,x,y):
        bx,by=self._find(x),self._find(y)
        if bx!=by:
            bx,by=by,bx if bx>by else bx,by # 保证小标号作为parent
            self._parents[by]=bx
    def _roots(self):
        for vertex,parent in enumerate(self._parents):
            if vertex==parent:
                yield vertex
'''
在此处声明，我们当前说的环，是一种跨超边的环，a-e1-b-e2-a才构成环，
同一个超边中即使有>=3个点，也不会成环
'''                
class HGPersistenceCalc:
    def __init__(self) -> None:
        self.reset_pfg()

    def reset_pfg(self):
        self._pairs=[]
        self._betti=None
        self._vatt=None

    def fit_transform(self,hg):
        assert isinstance(hg['dhg'],dhg.Hypergraph)
        cyc_edge=[]
        uf=UnionFind(num_v=hg['dhg'].num_v+hg['dhg'].num_e)
        _offset=hg['dhg'].num_v
        e_weights=np.array(hg['e_weights'])
        e_indices=np.argsort(e_weights,kind='stable') # 升序，稳定
        for e_idx,e_weight in zip(e_indices,e_weights[e_indices]):
            src_idx,dst_idx=hg['dhg'].v2e_src()[e_idx],hg['dhg'].v2e_dst()[e_idx] 
                # src-v,dst-e
            src_pa,dst_pa=uf._find(src_idx),uf._find(dst_idx+_offset)
            if src_pa==dst_pa:
                cyc_edge.append(e_idx)
                continue # v-e make up an cycle
            if src_pa>dst_pa:
               src,dst=dst,src
               src_pa,dst_pa=dst_pa,src_pa
            uf._merge(src,dst) 
            src_att=hg['v_lbl'][src] if self._vatt is not None else 0
            self._pairs.append((src_att,e_weight,src_pa)) # pairs

        unpaired_value=e_weights[e_indices[-1]] # 对root赋值
        for root in uf._roots():
            root_att=hg['v_lbl'][root] if self._vatt is not None else 0
            self._pairs.append((root_att,unpaired_value,root))
        self._betti=len(list(uf._roots())) # _betti 表示联通性
        return self._pairs,self._betti,cyc_edge #        

    
class HGWeightAssigner:
    def __init__(self,p=2) -> None:
        self._p=p
        self.tau=1
        self.e_weight_norm=False
        self.v_weight_norm=True
        
    def fit_transform(self,hg:defaultdict):
        e_weights=[]
        # 
        if 'pre_e_lbl' not in hg.keys():
            self._generate_pre_labelv2(hg)
        _indices=hg['dhg'].H_T.mm(hg['dhg'].H)._indices().cpu().numpy()
        #_indices_idx=_indices[0]<_indices[1]
        for i in range(_indices.shape[1]):
            src_idx,dst_idx=_indices[0][i],_indices[1][i]
            src_lbls=hg['pre_e_lbl'][src_idx]
            dst_lbls=hg['pre_e_lbl'][dst_idx]
            # nbr_lbl
            e_bet=(src_lbls[0]!=dst_lbls[0])+self._minikowsi(src_lbls[1],dst_lbls[1])+self.tau
            e_weights.append(e_bet)
        _indices=torch.tensor(_indices)
        e_weights=torch.tensor(e_weights)
        #A=torch.nonzero(hg['dhg'].H_T.mm(hg['dhg'].H))
        w_e=torch.sparse_coo_tensor(_indices,values=e_weights,size=[hg['dhg'].num_e,hg['dhg'].num_e]
                                ,device=hg['dhg'].device)
        # print(w_e.shape)
        # print(hg['dhg'].num_e)
        # normalize 
        if self.e_weight_norm:
            e_weight=torch.sparse.sum(w_e,dim=1)
            #e_weight=e_weight._values()/e_weight.sum(dim=0)
            e_weight=e_weight.tolist()
        else:
            e_weight=w_e # 
            #print(type(e_weight))
        hg['e_weights']=e_weight
        return hg
    def _fit_transform_v(self,hg:defaultdict):
        v_weights=[]
        if 'pre_v_lbl' not in hg.keys():
            self._generate_pre_labelv3(hg)
        _indices=hg['dhg'].H.mm(hg['dhg'].H_T)._indices().cpus().numpy()
        for i in range(_indices.shape[1]):
            src_idx,dst_idx=_indices[0][i]
            src_lbls=hg['pre_v_lbl'][src_idx]
            dst_lbls=hg['pre_v_lbl'][dst_idx]
            # nbr_lbl
            v_bet=(src_lbls[0]!=dst_lbls[0])+self._minikowsi(src_lbls[1],dst_lbls[1])+self.tau
            v_weights.append(v_bet)
        _indices=torch.tensor(_indices)
        v_weights=torch.tensor(v_weights)
        w_v=torch.sparse_coo_tensor(_indices,values=v_weights,size=[hg['dhg'].num_v,hg['dhg'].num_v]
                                    ,device=hg['dhg'].device)
        if self.v_weight_norm:
            v_weights=torch.sparse.sum(w_v,dim=1)
            v_weights=v_weights.tolist()
        else:
            v_weights=w_v
        hg['v_weights']=v_weights
        return hg
            
        
    def _generate_pre_labelv3(self,hg):
        tmp=[]
        for e_idx in range(hg['dhg'].num_e):
            nbr_lbl=sorted(
                hg['v_lbl'][v_idx] for v_idx in hg['dhg'].nbr_v(e_idx)
            )
            tmp.append(nbr_lbl)
        hg['v_lbl']=tmp
        tmp=[]
        for v_idx in range(hg['dhg'].num_v):
            cur_lbl=hg['v_lbl'][v_idx]
            nbr_lbl=[]
             
        pass
    
    def _generate_pre_label(self,hg):
        # precomputed_nbr_v=[sorted(hg['v_lbl'][v_idx] for v_idx in hg['dhg'].nbr_v(e_idx)) \
        #     for e_idx in range(hg['dhg'].num_e)]
        tmp=[]
        for v_idx in range(hg['dhg'].num_v):
            nbr_lbl=sorted(
                hg['e_lbl'][e_idx] for e_idx in hg['dhg'].nbr_e(v_idx)
            )
            #nbr_lbl.extend(hg['e_lbl'][e_idx] for e_idx in hg['dhg'].nbr_e(v_idx))
            tmp.append(nbr_lbl)
        hg['v_lbl']=tmp
        tmp=[]
        for e_idx in range(hg['dhg'].num_e):
            cur_lbl=hg['e_lbl'][e_idx]
            nbr_lbl=[]
            for v_idx in hg['dhg'].nbr_v(e_idx):
                nbr_lbl.extend(hg['v_lbl'][v_idx])            
            tmp.append([cur_lbl,sorted(nbr_lbl)])
        hg['pre_e_lbl']=tmp
        return hg

    def _generate_pre_labelv2(self,hg):
        tmp=[]
        for e_idx in range(hg['dhg'].num_e):
            cur_lbl=hg['e_lbl'][e_idx]
            nbr_lbl=sorted(
                hg['v_lbl'][v_idx] for v_idx in hg['dhg'].nbr_v(e_idx)
            )
            tmp.append([cur_lbl,nbr_lbl])
        hg['pre_e_lbl']=tmp
        return hg
    
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
    
    def _minikowsi(self,a,b):
        a,b=self._vector(a,b)
        return np.linalg.norm(a-b,ord=2)

    def fit_transform_ve(self,hg):
        assert isinstance(hg['dhg'])==dhg.Hypergraph
        e_weights=[]
        for edge in hg['dhg'].e[0]:
            src_idx,dst_idx=edge
            # all lbl are {cur_lbl,nbr_lbl}
            src_lbls=hg['pre_v_lbl'][src_idx]
            dst_lbls=hg['pre_v_lbl'][dst_idx]
            # nbr_lbl
            e_weight=self._e_weight_fun(src_lbls,dst_lbls)    
            e_weights.append(e_weight)
        hg['e_weights']=e_weights # H v e e_weights
        return hg
    
    def _e_weight_fun(self,src_lbls,dst_lbls,r=0.5):
        return r*(np.array(src_lbls[0])-np.mean(np.array(dst_lbls[1])))**2\
            +(1-r)*(np.array(dst_lbls[0])-np.mean(np.array(src_lbls[0])))**2
    
    
    def _minikowsi(self,a,b):
        a,b=np.array(a),np.array(b)
        return np.linalg.norm(a-b,ord=self._p)

class RetHGWeightKernel:
    def __init__(self,n_step=3,normalize=True,D=4000,gamma=10,p=2) -> None:
        self._step=n_step
        self._normalize=normalize
        self._D=D
        self._gamma=gamma
        self._w=self._gamma*torch.randn(self._step,self._D)
        self._b=torch.rand(self._D)*2*math.pi
        self._p=p
        self.we_a=HGWeightAssigner(p=2)
        self._r=0.2
        
    def fit_transform(self,hg_list):
        num_g=len(hg_list)
        self.featureMMD=torch.zeros(self._D,num_g) # _dx g.num_v
        for hg_idx,hg in enumerate(hg_list):
            self.we_a.fit_transform(hg)
            if "e_weights" not in hg.keys():
                hg['e_weights']=hg['dhg'].D_e_neg_1
            elif isinstance(hg["e_weights"],List):
                hg['e_weights']=torch.sparse_coo_tensor(indices=hg['dhg'].D_e_neg_1._indices(),
                                    values=self._r*torch.tensor(hg['e_weights'])+\
                                        (1-self._r)*hg['dhg'].D_e_neg_1._values(),
                                    size=(hg['dhg'].num_e,hg['dhg'].num_e),
                                    device=hg['dhg'].device)
            elif isinstance(hg['e_weights'],torch.Tensor):
                # _tmp=hg["e_weights"]._values()
                # _tmp=self._r*_tmp+(1-self._r)*hg['dhg'].D_e_neg_1._values()
                # _tmp=_tmp.float()
                # hg['e_weights']=_tmp
                hg["e_weights"]=self._r*hg["e_weights"]+(1-self._r)*hg['dhg'].D_e_neg_1
                hg["e_weights"]=hg['e_weights'].float() # 
                #print(hg['e_weights'].shape)
            else:
                raise ImportError
            #hg['e_weights']=torch.tensor(hg['e_weights'])
            feature_g=self.n_step_return(hg['dhg'],hg['e_weights'])
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
            self.we_a.fit_transform(hg)
            if "e_weights" not in hg.keys():
                hg['e_weights']=hg['dhg'].D_e_neg_1
            elif isinstance(hg["e_weights"],List):
                hg['e_weights']=torch.sparse_coo_tensor(indices=hg['dhg'].D_e_neg_1._indices(),
                                    values=self._r*torch.tensor(hg['e_weights'])+\
                                        (1-self._r)*hg['dhg'].D_e_neg_1._values(),
                                    size=(hg['dhg'].num_e,hg['dhg'].num_e),
                                    device=hg['dhg'].device)
            elif isinstance(hg['e_weights'],torch.Tensor):
                hg["e_weights"]=self._r*hg["e_weights"]+(1-self._r)*hg['dhg'].D_e_neg_1
                hg["e_weights"]=hg['e_weights'].float() # 
            feature_g=self.n_step_return(hg['dhg'],hg['e_weights']) # 
            Z=math.sqrt(2/self._D)*torch.cos(torch.matmul(self._w.t(),feature_g)) # _stepx_D _stepx g.num_v =_stepxg.num_v
            featureMMD[:,hg_idx]=torch.matmul(Z,torch.ones(feature_g.shape[1]))/feature_g.shape[1] # _step x g.num_v g.num_vx1            
        #
        test_ft=torch.cdist(featureMMD.T,self.featureMMD.T,p=2) # num_g x train_num_g
        sigma=torch.median(test_ft)+1e-5
        test_ft=torch.exp(-test_ft/sigma)
        return test_ft
    def n_step_return(self,hg:dhg.Hypergraph,e_weights:torch.tensor):
        num_v=hg.num_v 
        feature_v=torch.zeros((self._step,num_v))

        if self._step<=5:
            # if hg.D_e_neg_1.to_dense().unique().shape[0]==1:
            #     Tranpro=hg.D_v_neg_1.mm(hg.H).mm(hg.D_e_neg_1).mm(hg.H_T).to_dense()-torch.eye(hg.num_v)+hg.D_e_neg_1[0][0]
            # else:
            Tranpro=hg.D_v_neg_1.mm(hg.H).mm(e_weights).mm(hg.H_T).to_dense() #
                
            iTranpro=Tranpro
            for i in range(self._step):
                feature_v[i,:]=iTranpro.diagonal()
                iTranpro=iTranpro@Tranpro
        else:
            Tranpro=hg.D_v_neg_1_2.mm(hg.H).mm(e_weights).mm(hg.H_T).mm(hg.D_v_neg_1_2).to_dense() #
            Tranpro=(Tranpro+Tranpro.T)/2
            colD,U=torch.linalg.eigh(Tranpro)
            tempD=colD
            U=U**2
            for s in range(self._step):
                feature_v[s,:]=torch.matmul(U,tempD)
                tempD*=colD
        return feature_v

class HypergraphPersistenceWL:
    def __init__(self,p=2,
                 use_wl=True,
                 use_cyc=False,
                 use_lbl=False,
                 normalize=True) -> None:
        self._p=p
        self._use_wl=use_wl
        self._use_cyc=use_cyc
        self._use_lbl=use_lbl
        self._normalize=normalize
        self._trans={'lap','ori'}
        self.n_iter=1
        self._subtree_map={}
        assert self._use_cyc or self._use_lbl or self._use_wl

    def remap_v(self, hg_list, cnt, drop=False):
        for hg_idx, hg in enumerate(hg_list):
            for v_idx in range(hg["num_v"]):
                cur_lbl = hg["v_lbl"][v_idx]
                cur_lbl = "v" + str(cur_lbl)
                if cur_lbl not in self._subtree_map:
                    if drop:
                        hg["v_lbl"][v_idx] = -1
                        continue
                    else:
                        self._subtree_map[cur_lbl] = len(self._subtree_map)
                hg["v_lbl"][v_idx] = self._subtree_map[cur_lbl]
                cnt[hg_idx][self._subtree_map[cur_lbl]] += 1
        return hg_list, cnt

    def remap_e(self, hg_list, cnt, drop=False):
        for hg_idx, hg in enumerate(hg_list):
            for e_idx in range(hg["dhg"].num_e):
                cur_lbl = hg["e_lbl"][e_idx]
                cur_lbl = "e" + str(cur_lbl)
                if cur_lbl not in self._subtree_map:
                    if drop:
                        hg["e_lbl"][e_idx] = -1
                        continue
                    else:
                        self._subtree_map[cur_lbl] = len(self._subtree_map)
                hg["e_lbl"][e_idx] = self._subtree_map[cur_lbl]
                cnt[hg_idx][self._subtree_map[cur_lbl]] += 1
        return hg_list, cnt

    def cnt2mat(self, raw_cnt):
        # filter count
        cnt = []
        valid_id_set = set(
            [v for k, v in self._subtree_map.items() if k.startswith("v")]
        )
        id_map = {k: v for v, k in enumerate(sorted(valid_id_set))}
        for c in raw_cnt:
            cnt.append({id_map[k]: v for k, v in c.items() if k in valid_id_set})
        # count
        row_idx, col_idx, data = [], [], []
        for idx, g in enumerate(cnt):
            for lbl, c in g.items():
                row_idx.append(idx)
                col_idx.append(lbl)
                data.append(c)
        return (
            torch.sparse_coo_tensor(
                torch.tensor([row_idx, col_idx]),
                torch.tensor(data),
                size=(len(cnt), len(self._subtree_map)),
            )
            .coalesce()
            .float()
        )
                
    def get_hgwl_feature(self, hg_list):
        # if self.degree_as_label:
        #     for hg in hg_list:
        #         hg["v_lbl"] = [int(v) for v in hg["dhg"].deg_v]
        #         hg["e_lbl"] = [int(e) for e in hg["dhg"].deg_e]
        self._cnt = [defaultdict(int) for _ in range(len(hg_list))]
        self.remap_v(hg_list, self._cnt)
        self.remap_e(hg_list, self._cnt)
        for _ in range(self.n_iter):
            for hg in hg_list:
                tmp = []
                for e_idx in range(hg["dhg"].num_e):
                    cur_lbl = hg["e_lbl"][e_idx]
                    nbr_lbl = sorted(
                        hg["v_lbl"][v_idx] for v_idx in hg["dhg"].nbr_v(e_idx)
                    )
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                hg["e_lbl"] = tmp
            self.remap_e(hg_list, self._cnt)
            for hg in hg_list:
                tmp = []
                for v_idx in range(hg["dhg"].num_v):
                    cur_lbl = hg["v_lbl"][v_idx]
                    nbr_lbl = sorted(
                        hg["e_lbl"][e_idx] for e_idx in hg["dhg"].nbr_e(v_idx)
                    )
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                hg["v_lbl"] = tmp
            self.remap_v(hg_list, self._cnt)
        self.train_cnt = self.cnt2mat(self._cnt)

        self.train_ft = self.train_cnt.mm(self.train_cnt.t()).to_dense()
        if self._normalize:
            self.train_ft_diag = torch.diag(self.train_ft)
            self.train_ft = (
                self.train_ft
                / torch.outer(self.train_ft_diag, self.train_ft_diag).sqrt()
            )
            self.train_ft[torch.isnan(self.train_ft)] = 0
        return self.train_ft

    def _get_hgwl_feature(self, hg_list):
        # if self.degree_as_label:
        #     for hg in hg_list:
        #         hg["v_lbl"] = [int(v) for v in hg["dhg"].deg_v]
        #         hg["e_lbl"] = [int(e) for e in hg["dhg"].deg_e]
        cnt = [defaultdict(int) for _ in range(len(hg_list))]
        self.remap_v(hg_list, cnt, drop=True)
        self.remap_e(hg_list, cnt, drop=True)
        for _ in range(self.n_iter):
            for hg in hg_list:
                tmp = []
                for e_idx in range(hg["dhg"].num_e):
                    cur_lbl = hg["e_lbl"][e_idx]
                    nbr_lbl = sorted(
                        hg["v_lbl"][v_idx] for v_idx in hg["dhg"].nbr_v(e_idx)
                    )
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                hg["e_lbl"] = tmp
            self.remap_e(hg_list, cnt, drop=True)
            for hg in hg_list:
                tmp = []
                for v_idx in range(hg["dhg"].num_v):
                    cur_lbl = hg["v_lbl"][v_idx]
                    nbr_lbl = sorted(
                        hg["e_lbl"][e_idx] for e_idx in hg["dhg"].nbr_e(v_idx)
                    )
                    tmp.append(f"{cur_lbl},{nbr_lbl}")
                hg["v_lbl"] = tmp
            self.remap_v(hg_list, cnt, drop=True)
        test_cnt = self.cnt2mat(cnt)

        test_ft = test_cnt.mm(self.train_cnt.t()).to_dense()
        if self._normalize:
            test_ft_diag = torch.sparse.sum(test_cnt * test_cnt, dim=1).to_dense()
            test_ft = test_ft / torch.outer(test_ft_diag, self.train_ft_diag).sqrt()
            test_ft[torch.isnan(test_ft)] = 0
        return test_ft
    
    def fit_transform(self,hg_list):
        num_hg=len(hg_list)
        wa=HGWeightAssigner(p=2)
        pfg=HGPersistenceCalc()
        x_hgwl=self.get_hgwl_feature(hg_list)
        # num vertex
        self.num_labels=self.train_cnt.shape[1]
        num_ft=self._use_cycle*self.num_labels+\
            self._use_lbl*self.num_labels
        if num_ft>0:
            self.feature_ft=torch.zeros((num_hg,num_ft))
        for hg_idx,hg in enumerate(hg_list):
            hg=wa.fit_transform_ve(hg=hg)
            pfg.reset_pfg()
            pairs,_betti,cyc_edge=pfg.fit_transform(hg)
            if self._use_cycle:
                num_cycle=len(cyc_edge) # 
                '''
                值得注意的是，对于超图，并不满足_betti数和cycle数的相关性
                但对于图，根据欧拉公式：
                num_e-num_v+_betti
                '''
                x_cyc=torch.zeros(self.num_labels)
                for e_idx in cyc_edge:
                    src=hg['dhg'].v2e_src()[e_idx]
                    e_weight=hg['e_weights'][e_idx]
                    x_cyc[hg['v_lbl'][src]]+=e_weight**self._p

                    # dst=hg['dhg'].v2e_dst()[e_idx]
                    # x_cyc[hg['e_lbl'][dst]]
                self.feature_ft[hg_idx,self._use_lbl*self.num_labels:]=x_cyc
            if num_ft>0:
                if self._trans == 'lap':
                    self.feature_g=torch.cdist(self.feature_ft,self.feature_ft)
                    sigma=torch.median(self.feature_g)
                    self.feature_g=torch.exp(-self.feature_g/sigma)
                elif self._trans == 'ori':
                    self.feature_g=self.feature_ft@self.feature_ft.t()
                else:
                    self.feature_g=torch.tensor(0)
                x_hgwl=x_hgwl+self.feature_g            
            return x_hgwl
    
            
    def transform(self,hg_list):
        num_g=len(hg_list)
        #wl=GraphSubtreeKernel(normalize=True,n_iter=1)
        wa=HGWeightAssigner(p=2)
        pfg=HGPersistenceCalc()
        x_hgwl=self._get_hgwl_feature(hg_list)

        num_ft=self._use_cyc*self.num_labels+\
            self._use_lbl*self.num_labels
        if num_ft>0:
            feature_g=torch.zeros((num_g,num_ft))
        for hg_idx,hg in enumerate(hg_list):
            g=wa.fit_transform_ve(hg=hg)
            pfg.reset_pfg()
            pairs,_betti,cyc_edge=pfg.fit_transform(g) # g is a dict
            
            if self._use_cyc:
                #num_cycle=g['dhg'].num_e-g['num_v']+_betti
                num_cycle = len(cyc_edge) # 
                x_cyc=torch.zeros(self.num_labels)
                for e_idx in cyc_edge:
                    #src,dst=g['dhg'].e[0][e_idx]
                    src = hg['dhg'].v2e_src()[e_idx]
                    e_weight=g['e_weights'][e_idx]
                    x_cyc[hg['v_lbl'][src]]+=e_weight**self._p
                    #x_cyc[g['v_lbl'][dst]]+=e_weight**self._p
                feature_g[hg_idx,self._use_lbl*self.num_labels:]=x_cyc
        if num_ft>0:
            if self._trans=='lap':
                feature_g=torch.cdist(feature_g,self.feature_ft) # num_g x self.num_g
                sigma=torch.median(feature_g)
                feature_g=torch.exp(-feature_g/sigma)  # augment feature
            elif self._trans == 'ori':
                feature_g=feature_g@self.feature_ft.t()
            else:
                feature_g=torch.tensor(0)
            x_hgwl=x_hgwl+feature_g   
            #x_wl=torch.cat((x_wl,feature_g),dim=0) # original feature and augment feature
        return x_hgwl

if __name__=='__main__':
    hg={
        'dhg':dhg.Hypergraph(num_v=3,e_list=[(0,1),(1,2),(2,0)]),
        'e_lbl':[0,1,0],
        'v_lbl':[1,0,1],
    }
    model=HGWeightAssigner()
    model._generate_pre_label(hg)
    print(hg)
    model.fit_transform(hg)
    print(hg['e_weights'])
    hg=dhg.Hypergraph(num_v=hg['dhg'].num_v,e_list=hg['dhg'].e[0],e_weight=hg['e_weights'])
    
    print(hg)
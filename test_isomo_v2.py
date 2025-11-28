import dhg 
from dhg import Hypergraph
import numpy as np
from collections import defaultdict
import torch

class HGSCKernel:
    def __init__(self,step=5,n_iter=2,use_tree=False,use_cycle=False) -> None:
        self._subtree_map={}
        self.n_iter=n_iter
        self._step=step
        self._use_tree_feature=use_tree
        self._use_cycle_feature=use_cycle
        self._use_tc=False
        if self._use_cycle_feature and self._use_tree_feature:
            self._use_tc=True
        if (self._use_cycle_feature or self._use_tree_feature) is False:
            raise ImportError 
        #,'use tree feature or cycle feature'
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
    
    def gen_hgdict(self,G:Hypergraph,e_lbl=None,v_lbl=None):
        return {
            'num_v':G.num_v,
            "num_e":G.num_e,
            'v_lbl':G.deg_v if v_lbl is None else v_lbl, # 纯结构
            'e_lbl':G.deg_e if e_lbl is None else e_lbl,
            'dhg': G,
        }
    
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
    def _n_step_cycle(self,hg:Hypergraph):
        num_v=hg.num_v
        v_feature=torch.zeros((self._step,num_v))
        if self._step<10:
            # if hg.D_e_neg_1.to_dense().unique().shape[0]==1:
            #     tran_pro=hg.D_v_neg_1.mm(hg.H).mm(hg.D_e_neg_1).mm(hg.H_T).to_dense()-torch.eye(hg.num_v)+hg.D_e_neg_1[0][0]
            # else:
            #     tran_pro=hg.D_v_neg_1.mm(hg.H).mm(hg.D_e_neg_1).mm(hg.H_T).to_dense() #
       
            tran_pro=hg.H.mm(hg.H_T).to_dense()
            #tran_pro.diagonal().add_(1)
            itran_pro=tran_pro
            for i in range(self._step):
                v_feature[i,:]=torch.diag(itran_pro)
                itran_pro=itran_pro.mm(tran_pro)           
             
        return v_feature.sum(dim=1)
    def test_isomo(self,G1:Hypergraph,G2:Hypergraph):


        hg_list=[]
        hg_list.append(self.gen_hgdict(G1))
        hg_list.append(self.gen_hgdict(G2))
        self._cnt=[defaultdict(int) for _ in range(len(hg_list))]
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
        test_cnt = self.cnt2mat(self._cnt).to_dense()

        G1_c_ft=self._n_step_cycle(G1).reshape(-1)
        G2_c_ft=self._n_step_cycle(G2).reshape(-1)
        print(G1_c_ft)
        print(G2_c_ft)
        G1_t_ft=test_cnt[0]
        G2_t_ft=test_cnt[1]
        if self._use_tc:
            G1_ft=torch.cat((G1_t_ft,G1_c_ft),dim=0)
            G2_ft=torch.cat((G2_t_ft,G2_c_ft),dim=0)
        else:
            if self._use_tree_feature:
                G1_ft=G1_t_ft
                G2_ft=G2_t_ft
            if self._use_cycle_feature:
                G1_ft=G1_c_ft
                G2_ft=G2_c_ft
        
        err_test=torch.norm(G1_ft-G2_ft,p=2)
        print(err_test)
        if err_test<1e-5:
            return True
        else:
            return False

# e_list1=[(0,1,3),(0,2,3),(0,1,2,4),(0,4,5,6),(4,5,7),(4,7,6)]
# e_list2=[(2,3,4),(0,2,4),(0,1,2,3),(0,1,5,6),(1,6,7),(0,5,7)]

# G1=Hypergraph(8,e_list1)
# G2=Hypergraph(8,e_list2)
'''
0  1  2  3
4  5  6  7
8  9  10 11
12 13 14 15
'''
rook4_edge=[
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 8], [0, 12],
    [1, 2], [1, 3], [1, 5], [1, 9], [1, 13],
    [2, 3], [2, 6], [2, 10], [2, 14],
    [3, 7], [3, 11], [3, 15],
    [4, 5], [4, 6], [4, 7], [4, 8], [4, 12],
    [5, 6], [5, 7], [5, 9], [5, 13],
    [6, 7], [6, 10], [6, 14],
    [7, 11], [7, 15],
    [8, 9], [8, 10], [8, 11], [8, 12],
    [9, 10], [9, 11], [9, 13],
    [10, 11], [10, 14],
    [11, 15],
    [12, 13], [12, 14], [12, 15],
    [13, 14], [13, 15],
    [14, 15]
]
shrik4=[
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 6], [0, 9],
    [1, 2], [1, 3], [1, 5], [1, 7], [1, 8], [1, 10],
    [2, 5], [2, 6], [2, 8], [2, 11], [2, 13],
    [3, 6], [3, 7], [3, 12], [3, 14],
    [4, 5], [4, 7], [4, 10], [4, 11], [4, 15],
    [5, 12], [5, 13], [5, 15],
    [6, 11], [6, 12], [6, 14],
    [7, 13], [7, 14], [7, 15],
    [8, 9], [8, 10], [8, 13], [8, 14],
    [9, 10], [9, 11], [9, 12], [9, 15],
    [10, 13], [10, 14],
    [11, 15],
    [12, 13], [12, 15],
    [13, 14], [13, 15],
    [14, 15]
]
dis=[[0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1], [1, 2, 3, 0]]
allow=[[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]

def hack(num_v=16):
    #num_v=16
    e_list=[]
    for i in range(num_v-1):
        for j in range(i+1,num_v):
            if i>>2==j>>2 or (i and 3)==(j and 3):
                e_list.append((i,j))
    return e_list 

def shrik(num_v=16):
    e_list=[]
    for i in range(num_v-1):
        for j in range(num_v):
            if allow[dis[i>>2][j>>2]][dis[i and 3][j and 3]]:
                e_list.append((i,j))
    return e_list

# e_list1=[(0,1),(0,2),(0,3),(2,3),(1,4),(1,5),(4,5)]
# e_list2=[(0,1),(0,2),(2,3),(1,3),(0,4),(1,5),(4,5)]
# print(len(shrik()))
# G1=Hypergraph(16,shrik4)
# G2=Hypergraph(16,rook4_edge)
G1=Hypergraph(6,[(0,2,5),(1,2,5),(0,1,5),(2,3,4),(2,5,4),(2,5,3)])
G2=Hypergraph(6,[(0,2,5),(1,2,5),(0,1,5),(2,3,4),(2,5,4),(2,5,3)])
model=HGSCKernel(use_cycle=False,use_tree=True)
print(model.test_isomo(G1,G2))
#print(shrik())        
from itertools import combinations
from collections import defaultdict

import torch
import numpy as np
from dhg import Hypergraph, Graph, DiGraph


# Hypergraph WL Subtree Kernel from "Feng et al. 2024 IEEE TPAMI Hypergraph Isomorphism Computation"
class HypergraphSubtreeKernel:
    def __init__(self, n_iter=4, degree_as_label=True, normalize=True,_oa=False):
        self.n_iter = n_iter
        self.normalize = normalize
        self.degree_as_label = degree_as_label
        self._subtree_map = {}
        self._oa=_oa

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

    def fit_transform(self, hg_list):
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
        if self._oa:
            #self.train_ft=self._optim_assign(self.train_cnt.to_dense(),self.train_cnt.to_dense())
            self.train_ft=self._optim_assign_sparse(self.train_cnt,self.train_cnt).to_dense()
        else: 
            self.train_ft = self.train_cnt.mm(self.train_cnt.t()).to_dense()
        if self.normalize:
            self.train_ft_diag = torch.diag(self.train_ft)
            self.train_ft = (
                self.train_ft
                / torch.outer(self.train_ft_diag, self.train_ft_diag).sqrt()
            )
            self.train_ft[torch.isnan(self.train_ft)] = 0
        return self.train_ft

    def transform(self, hg_list):
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
        if self._oa:
            #test_ft=self._optim_assign(test_cnt.to_dense(),self.train_cnt.to_dense())
            test_ft=self._optim_assign_sparse(test_cnt,self.train_cnt).to_dense()
        else:
            test_ft = test_cnt.mm(self.train_cnt.t()).to_dense()
        if self.normalize:
            test_ft_diag = torch.sparse.sum(test_cnt * test_cnt, dim=1).to_dense()
            test_ft = test_ft / torch.outer(test_ft_diag, self.train_ft_diag).sqrt()
            test_ft[torch.isnan(test_ft)] = 0
        return test_ft
    def _optim_assign(self,x_srt:torch.tensor,y_dst:torch.tensor):
        # x_srt:(nxd), y_dst:(mxd) 
        #return X_ft: (nxm)
        x_srt_expanded=x_srt.unsqueeze(1) # (n,1,d)
        y_dst_expanded=y_dst.unsqueeze(0) # (1,m,d)
        result=torch.min(x_srt_expanded,y_dst_expanded).sum(dim=2) # (n,m)
        
        return result
    def _optim_assign_sparse(self,x_srt:torch.sparse.Tensor,
                             y_dst:torch.sparse.Tensor):
        # get index and values
        x_indices=x_srt._indices()  # 2,nnz_x
        x_values=x_srt._values()    # nnz_x
        y_indices=y_dst._indices()  # 2,nnz_y
        y_values=y_dst._values()    # nnz_y
        # get x_srt and y_dst size
        n,m=x_srt.size(0),y_dst.size(0)
        x_col_indices=x_indices[1]  # col
        y_col_indices=y_indices[1]
        # match 
        match=x_col_indices.unsqueeze(1)==y_col_indices.unsqueeze(0) # nnz_x,nnz_y
        match_min_values=torch.min(x_values.unsqueeze(1),y_values.unsqueeze(0)) # (nnz_x,nnz_y)
        # filter
        filter_min_values=match_min_values[match]
        filter_x_row_indices=x_indices[0].unsqueeze(1).expand(-1,y_indices.size(1))[match] # 
        filter_y_col_indices=y_indices[0].unsqueeze(0).expand(x_indices.size(1),-1)[match]
        # sparse tensor
        result_indices=torch.stack([filter_x_row_indices,filter_y_col_indices]) # (2,filter_nnz)
        result_sparse=torch.sparse_coo_tensor(result_indices,filter_min_values,(n,m))
        
        return result_sparse
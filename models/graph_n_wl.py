import dhg
import torch
from collections import defaultdict
from tqdm import tqdm
class G2WLSubtree:
    def __init__(self,n_iter=4,normalize=True) -> None:
        self.n_iter=n_iter
        self.normalize=normalize
        self._subtree_map={}
    def remap(self,g_list,count,drop=False):
        for g_idx,g in enumerate(g_list):
            # num of (u,v)
            for pair,cur_lbl in g['pair_v'].items():
                if cur_lbl not in self._subtree_map:
                    if drop:
                        g['pair_v'][pair]=-1
                        continue
                    else:
                        self._subtree_map[cur_lbl]=len(self._subtree_map)
                g['pair_v'][pair]=self._subtree_map[cur_lbl]
                count[g_idx][self._subtree_map[cur_lbl]]+=1
        return g_list,count
    def _init_remap(self,g_list,count):
        '''
        init vertex pair, (u,v) and (v,u)-->
        '''
        for g_idx,g in tqdm(enumerate(g_list), total=len(g_list)):
            g['pair_v']={}
            for u in range(g['num_v']):
                for v in range(g['num_v']):
                    cur_lbl=f"{g['v_lbl'][u]},{g['v_lbl'][v]}"
                    if cur_lbl not in self._subtree_map:
                        self._subtree_map[cur_lbl]=len(self._subtree_map)
                    g['pair_v'][(u,v)]=self._subtree_map[cur_lbl]
                    count[g_idx][self._subtree_map[cur_lbl]]+=1
        return g_list,count
    def wl_step_2d(self,g_list,count,drop=False):
        '''2-WL update labels'''
        for g_idx,g in tqdm(enumerate(g_list), total=len(g_list)):
            tmp={}
            for u in range(g['num_v']):
                for v in range(u,g['num_v']):
                    cur_label=g['pair_v'][(u,v)]
                    # obtain neighbor labels
                    neighbor_pair_labels=[]
                    
                    for neighbor_u in sorted(g['dhg'].nbr_v(u)):
                        for neighbor_v in sorted(g['dhg'].nbr_v(v)):
                            if neighbor_u>neighbor_v:
                                neighbor_u,neighbor_v=neighbor_v,neighbor_u
                            neighbor_pair_labels.append(g['pair_v'][(neighbor_u,neighbor_v)])
                    #neighbor_pair_labels.sort(key=lambda i: neighbor_pair_labels[i][0])
                    neighbor_pair_labels.sort()
                    combined_label=f"{cur_label},{neighbor_pair_labels}"
                    
                    tmp[(u,v)]=combined_label
            self.remap(g_list,count,drop=drop)
    def count2mat(self,count):
        row_idx,col_idx,data=[],[],[]
        for idx,g in enumerate(count):
            for lbl,cnt in g.items():
                row_idx.append(idx)
                col_idx.append(lbl)
                data.append(cnt)
        return (
            torch.sparse_coo_tensor(
                torch.tensor([row_idx,col_idx]),
                torch.tensor(data),
                size=(len(count),len(self._subtree_map))
            ).coalesce().float()
        )
    def fit_transform(self,g_list):
        self._count=[defaultdict(int) for _ in range(len(g_list))]
        self._init_remap(g_list,self._count)
        for _ in range(self.n_iter):
            self.wl_step_2d(g_list,self._count,drop=True)
        self.train_cnt=self.count2mat(self._count)
        self.train_ft=self.train_cnt.mm(self.train_cnt.t()).to_dense()
        if self.normalize:
            self.train_ft_diag=torch.diag(self.train_ft)
            self.train_ft=(
                self.train_ft
                /torch.outer(self.train_ft_diag,self.train_ft_diag).sqrt()
            )
            self.train_ft[torch.isnan(self.train_ft)]=0
        return self.train_ft
    def transform(self,g_list):
        count=[defaultdict(int) for _ in range(len(g_list))]
        self._init_remap(g_list,count)
        for _ in range(self.n_iter):
            self.wl_step_2d(g_list,count,drop=False)
        test_cnt=self.count2mat(count)
        test_ft=test_cnt.mm(self.train_cnt.t()).to_dense()
        if self.normalize:
            test_ft_diag=torch.sparse.sum(test_cnt*test_cnt,dim=1).to_dense()
            test_ft=test_ft/torch.outer(test_ft_diag,self.train_ft_diag).sqrt()
            test_ft[torch.isnan(test_ft)]=0
        return test_ft
    

        
        
                
                
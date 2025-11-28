import torch
import torch.nn as nn
import numpy as np

class HighOrderMotif(nn.Module):
    def __init__(self,normalize=True):
        super().__init__()
        self.normalize=normalize
    def _load_ft(self,file_path):
        import pickle
        with open(file_path, 'rb') as f:
            pickle_data = pickle.load(f)
        self.ft=torch.tensor(pickle_data, dtype=torch.float32)
        return self.ft
    def fit_transform(self,hg_list):
        train_idx=torch.tensor(hg_list,dtype=torch.long)
        self.train_cnt = self.ft[train_idx]
        self.train_ft = self.train_cnt.mm(self.train_cnt.t())
        if self.normalize:
            self.train_ft_diag =torch.diag(self.train_ft)
            self.train_ft = (
                self.train_ft/torch.outer(self.train_ft_diag,self.train_ft_diag).sqrt()
            )
            self.train_ft[torch.isnan(self.train_ft)] = 0
        return self.train_ft
    def transform(self,hg_list):
        test_idx=torch.tensor(hg_list,dtype=torch.long)
        test_cnt = self.ft[test_idx]
        test_ft = test_cnt.mm(self.train_cnt.t())
        if self.normalize:
            test_ft_diag = torch.sum(test_cnt*test_cnt,dim=1)
            test_ft = (
                test_ft/torch.outer(test_ft_diag,self.train_ft_diag).sqrt()
            )
            test_ft[torch.isnan(test_ft)] = 0
        return test_ft
    
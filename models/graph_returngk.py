import dhg 
import torch
import math
import numpy as np

class RetGK:
    def __init__(self,n_step=3,normalize=True,D=4000,gamma=10,p=2) -> None:
        self._step=n_step
        self._normalize=normalize
        self._D=D
        self._gamma=gamma
        self._w=self._gamma*torch.randn(self._step,self._D)
        self._b=torch.rand(self._D)*2*math.pi
        self._p=p
        
    def fit_transform(self,g_list):
        num_g=len(g_list)
        self.featureMMD=torch.zeros(self._D,num_g) # _dx g.num_v
        for g_idx,g in enumerate(g_list):
            feature_g=self.n_step_return(g['dhg'])
            Z=math.sqrt(2/self._D)*torch.cos(torch.matmul(self._w.t(),feature_g)) # _stepx_D _Dx g.num_v =_stepxg.num_v
            self.featureMMD[:,g_idx]=torch.matmul(Z,torch.ones(feature_g.shape[1]))/feature_g.shape[1] # _step x g.num_v g.num_vx1
        # 
        self.train_ft=torch.cdist(self.featureMMD.T,self.featureMMD.T,p=2) # (num_g,num_g)
        sigma=torch.median(self.train_ft)+1e-5
        self.train_ft=torch.exp(-self.train_ft/sigma)
        return self.train_ft

    def transform(self,g_list):
        num_g=len(g_list)
        featureMMD=torch.zeros(self._D,num_g) # _d x g.num_v
        for g_idx,g in enumerate(g_list):
            feature_g=self.n_step_return(g['dhg']) if isinstance(g['dhg'],dhg.Graph) else self.n_step_return(g['dhgg'])
            Z=math.sqrt(2/self._D)*torch.cos(torch.matmul(self._w.t(),feature_g)) # _stepx_D _stepx g.num_v =_stepxg.num_v
            featureMMD[:,g_idx]=torch.matmul(Z,torch.ones(feature_g.shape[1]))/feature_g.shape[1] # _step x g.num_v g.num_vx1            
        #
        test_ft=torch.cdist(featureMMD.T,self.featureMMD.T,p=2) # num_g x train_num_g
        sigma=torch.median(test_ft)+1e-5
        test_ft=torch.exp(-test_ft/sigma)
        return test_ft
    def n_step_return(self,g:dhg.Graph):
        num_v=g.num_v 
        feature_v=torch.zeros((self._step,num_v))
        g.add_extra_selfloop()
        #Adjmat=g.A.to_dense()
        if self._step<=5:
            Tranpro=g.D_v_neg_1.mm(g.A).to_dense() #
            iTranpro=Tranpro
            for i in range(self._step):
                feature_v[i,:]=iTranpro.diagonal()
                iTranpro=iTranpro@Tranpro
        else:
            Tranpro=g.D_v_neg_1_2.mm(g.A).mm(g.D_v_neg_1_2).to_dense() #
            Tranpro=(Tranpro+Tranpro.T)/2
            colD,U=torch.linalg.eigh(Tranpro)
            tempD=colD
            U=U**2
            for s in range(self._step):
                feature_v[s,:]=torch.matmul(U,tempD)
                tempD*=colD
        return feature_v

class RetHGK:
    def __init__(self,n_step=3,normalize=True,D=4000,gamma=10,p=2) -> None:
        self._step=n_step
        self._normalize=normalize
        self._D=D
        self._gamma=gamma
        self._w=self._gamma*torch.randn(self._step,self._D)
        self._b=torch.rand(self._D)*2*math.pi
        self._p=p
        
    def fit_transform(self,hg_list):
        num_g=len(hg_list)
        self.featureMMD=torch.zeros(self._D,num_g) # _dx g.num_v
        for hg_idx,hg in enumerate(hg_list):
            feature_g=self.n_step_return(hg['dhg'])
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
            feature_g=self.n_step_return(hg['dhg']) # 
            Z=math.sqrt(2/self._D)*torch.cos(torch.matmul(self._w.t(),feature_g)) # _stepx_D _stepx g.num_v =_stepxg.num_v
            featureMMD[:,hg_idx]=torch.matmul(Z,torch.ones(feature_g.shape[1]))/feature_g.shape[1] # _step x g.num_v g.num_vx1            
        #
        test_ft=torch.cdist(featureMMD.T,self.featureMMD.T,p=2) # num_g x train_num_g
        sigma=torch.median(test_ft)+1e-5
        test_ft=torch.exp(-test_ft/sigma)
        return test_ft
    def random_walk(self,hg:dhg.Hypergraph):
        num_v=hg.num_v
        feature_v=torch.zeros((self._step,num_v))
        if self._step<=5:
            Tranpro=hg.D_v_neg_1_2.mm(hg.H).mm(hg.D_e_neg_1).mm(hg.H_T).mm(hg.D_v_neg_1_2).to_dense() #
            iTranpro=Tranpro
            for i in range(self._step):
                feature_v[i,:]=iTranpro.diagonal()
                iTranpro=iTranpro@Tranpro
        else:
            Tranpro=hg.D_v_neg_1.mm(hg.H).mm(hg.D_e_neg_1).mm(hg.H_T).to_dense() #
            Tranpro=(Tranpro+Tranpro.T)/2
            colD,U=torch.linalg.eigh(Tranpro)
            tempD=colD
            U=U**2
            for s in range(self._step):
                feature_v[s,:]=torch.matmul(U,tempD)
                tempD*=colD
        return feature_v
    def n_step_return(self,hg:dhg.Hypergraph):
        num_v=hg.num_v 
        feature_v=torch.zeros((self._step,num_v))
        
        if self._step<=5:
            if hg.D_e_neg_1.to_dense().unique().shape[0]==1:
                Tranpro=hg.D_v_neg_1.mm(hg.H).mm(hg.D_e_neg_1).mm(hg.H_T).to_dense()-torch.eye(hg.num_v)+hg.D_e_neg_1[0][0]
            else:
                Tranpro=hg.D_v_neg_1.mm(hg.H).mm(hg.D_e_neg_1).mm(hg.H_T).to_dense() #
                
            iTranpro=Tranpro
            for i in range(self._step):
                feature_v[i,:]=iTranpro.diagonal()
                iTranpro=iTranpro@Tranpro
        else:
            Tranpro=hg.D_v_neg_1_2.mm(hg.H).mm(hg.D_e_neg_1).mm(hg.H_T).mm(hg.D_v_neg_1_2).to_dense() #
            Tranpro=(Tranpro+Tranpro.T)/2
            colD,U=torch.linalg.eigh(Tranpro)
            tempD=colD
            U=U**2
            for s in range(self._step):
                feature_v[s,:]=torch.matmul(U,tempD)
                tempD*=colD
        return feature_v

class EVMN:
    def __init__(self,n_step=3,normalize=True,D=4000,gamma=10,p=2) -> None:
        self._step=n_step
        self._normalize=normalize
        self._D=D
        self._gamma=gamma
        self._w=self._gamma*torch.randn(self._step,self._D)
        self._b=torch.rand(self._D)*2*math.pi
        self._p=p
        
    def fit_transform(self,hg_list):
        num_g=len(hg_list)
        self.featureMMD=torch.zeros(self._D,num_g) # _dx g.num_v
        for hg_idx,hg in enumerate(hg_list):
            feature_g=self.n_step_evwn(hg['dhg'])
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
            feature_g=self.n_step_evwn(hg['dhg']) # 
            Z=math.sqrt(2/self._D)*torch.cos(torch.matmul(self._w.t(),feature_g)) # _stepx_D _stepx g.num_v =_stepxg.num_v
            featureMMD[:,hg_idx]=torch.matmul(Z,torch.ones(feature_g.shape[1]))/feature_g.shape[1] # _step x g.num_v g.num_vx1            
        #
        test_ft=torch.cdist(featureMMD.T,self.featureMMD.T,p=2) # num_g x train_num_g
        sigma=torch.median(test_ft)+1e-5
        test_ft=torch.exp(-test_ft/sigma)
        return test_ft       
    def n_step_evwn(self,hg:dhg.Hypergraph):
        degv=hg.D_v._values()# [n]
        dege=hg.D_e._values()# [e]
        if not isinstance(degv,torch.Tensor):
            degv=torch.tensor(degv)
        if not isinstance(dege,torch.Tensor):
            dege=torch.tensor(dege)
        mat=torch.ger(degv,dege) #[n,e]
        hv_mat = torch.diag(mat.sum(1))  # [n, n]
        he_mat = torch.diag(mat.sum(0))  # [e, e]
        # 求逆
        hv_inv = torch.linalg.inv(hv_mat)
        he_inv = torch.linalg.inv(he_mat)
        # 计算转移矩阵
        Tranpro = hv_inv.mm(mat).mm(he_inv).mm(mat.T)
        num_v=hg.num_v 
        feature_v=torch.zeros((self._step,num_v))

        # if self._step<=5:
        iTranpro=Tranpro
        for i in range(self._step):
            feature_v[i,:]=iTranpro.diagonal()
            iTranpro=iTranpro@Tranpro
        # else:
        #     Tranpro=hg.D_v_neg_1_2.mm(hg.H).mm(hg.D_e_neg_1).mm(hg.H_T).mm(hg.D_v_neg_1_2).to_dense() #
        #     Tranpro=(Tranpro+Tranpro.T)/2
        #     colD,U=torch.linalg.eigh(Tranpro)
        #     tempD=colD
        #     U=U**2
        #     for s in range(self._step):
        #         feature_v[s,:]=torch.matmul(U,tempD)
        #         tempD*=colD
        return feature_v    
class SC:
    def __init__(self,n_step=3,normalize=True,D=4000,gamma=10,p=2) -> None:
        self._step=n_step
        self._normalize=normalize
        self._D=D
        self._gamma=gamma
        self._w=self._gamma*torch.randn(self._step,self._D)
        self._b=torch.rand(self._D)*2*math.pi
        self._p=p
        
    def fit_transform(self,hg_list):
        num_g=len(hg_list)
        self.featureMMD=torch.zeros(self._D,num_g) # _dx g.num_v
        for hg_idx,hg in enumerate(hg_list):
            feature_g=self.n_step_sc(hg['dhg'])
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
            feature_g=self.n_step_sc(hg['dhg']) # 
            Z=math.sqrt(2/self._D)*torch.cos(torch.matmul(self._w.t(),feature_g)) # _stepx_D _stepx g.num_v =_stepxg.num_v
            featureMMD[:,hg_idx]=torch.matmul(Z,torch.ones(feature_g.shape[1]))/feature_g.shape[1] # _step x g.num_v g.num_vx1            
        #
        test_ft=torch.cdist(featureMMD.T,self.featureMMD.T,p=2) # num_g x train_num_g
        sigma=torch.median(test_ft)+1e-5
        test_ft=torch.exp(-test_ft/sigma)
        return test_ft       
        
    def n_step_sc(self,hg:dhg.Hypergraph):
        L=hg.L_HGNN
        L=(L+L.T)/2
        pass       
      
            
        
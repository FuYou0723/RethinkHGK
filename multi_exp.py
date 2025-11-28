import dhg
import time
import hydra
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from typing import List
from omegaconf import DictConfig, OmegaConf

from models import (
    #----------------#
    HypergraphRootedKernel,
    GraphSubtreeKernel,
    GraphletSampling,
    HypergraphDirectedLineKernel,
    HypergraphSubtreeKernel,
    HypergraphHyedgeKernel,
    #----------------#
    GHC,
    RetGK,
    RetHGK,
    ShortestPath,
    PersistenceWL,
    HypergraphESubtree,
    RetHGWeightKernel,
    #----------------#
)
from utils import load_data, separate_data, train_infer_SVM,_ml_train_infer

#print = logging.info
multi_label, criterion = None, None
import yaml
import datetime
import os,csv
import pandas as pd

def multi_exp():
    ##
    # 配置日志系统
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("app.log"),  # 日志记录到文件
                            logging.StreamHandler()          # 日志输出到控制台
                        ])
    # 创建Logger对象
    logger = logging.getLogger('ExampleLogger')
    
    #---#
    yaml_file='/home/zhangyifan/alive/HIC/ml_config.yaml'
    with open(yaml_file,'r') as f:
        f_cont=yaml.safe_load(f)
        cfg=DictConfig(f_cont)
    methods=[
            # 'graph_subtree',
            'graph_homo',
            # 'graph_retgk',
            #'hypergraph_ve',
            #'hypergraph_oa',
            #'hypergraph_directed_line',
            #'hypergraph_subtree',
            # 'hypergraph_hyedge',
            #'hypergraph_sck',
            #"hypergraph_spk",
            # 'graphlet_sampling',
            # 'hypergraph_rooted',
            # 'hypergraph_rethgk',
            'graph_shortestpath',
             ]
    exp_dict=defaultdict()
    exp_dict={
        'name':[],
        'method':[],
        'mean_acc':[],
        'var_acc':[],
        'f1_ma':[],
        'var_f1_ma':[],
        'f1_weight':[],
        'var_f1_weight':[]
    }
    datasets={
        # 1: "stream_player",
        # 2: "IMDB_wri_form",
        # 3: "IMDB_wri_genre",
        # 4: "IMDB_dir_form",
        # 5: "IMDB_dir_genre",
        # 6: "MUTAG",
        # 7: "NCI1",
        # 8: "PROTEINS",
        # 9: 'RHG_3',
        # 10:"RHG_10",
        # 11:"IMDBBINARY",
        # 12:"IMDBMULTI"
        13: "RG_sub",
        14: "RG_macro",
    } 
    #--init model---# 
    
    # 
    for idx,data_items in datasets.items():
        cfg.data.name=data_items
        for method in methods:
            cfg.model.name=method
            #------#
            if cfg.model.name in [
                "hypergraph_rooted",
                "hypergraph_directed_line",
                "hypergraph_subtree",
                "hypergraph_hyedge",
                "hypergraph_rethgk",
                "hypergraph_sck",
                "hypergraph_spk",
                'hypergraph_oa',
                'hypergraph_ve'
            ]:
                model_type = "hypergraph"
            else:
                model_type = "graph"    
            print(OmegaConf.to_yaml(cfg))
            dhg.random.set_seed(cfg.seed)
            x_list, y_list, meta = load_data(
                cfg.data.name, cfg.data.root, cfg.data.degree_as_tag, model_type
            )
            multi_label = meta["multi_label"]
            n_classes = meta["n_classes"]

            n_fold_idx = separate_data(x_list, y_list, cfg.data.n_fold, cfg.seed)
            #-------#
            if cfg.model.name == "graph_subtree":
                model = GraphSubtreeKernel(normalize=cfg.model.normalize)
            elif cfg.model.name == "graphlet_sampling":
                model = GraphletSampling(normalize=cfg.model.normalize, sampling={})
            elif cfg.model.name == 'graph_retgk':
                model = RetGK(n_step=3)
            elif cfg.model.name == 'hypergraph_rethgk':
                model = RetHGK(n_step=3,gamma=2)
            elif cfg.model.name == 'graph_shortestpath':
                model = ShortestPath(normalize=True,D=40)
            elif cfg.model.name == "hypergraph_rooted":
                model = HypergraphRootedKernel(normalize=cfg.model.normalize)
            elif cfg.model.name == "hypergraph_directed_line":
                model = HypergraphDirectedLineKernel(normalize=cfg.model.normalize)
            elif cfg.model.name == "hypergraph_subtree":
                model = HypergraphSubtreeKernel(normalize=cfg.model.normalize,n_iter=4)
            elif cfg.model.name == "hypergraph_hyedge":
                model = HypergraphHyedgeKernel(normalize=cfg.model.normalize)
            elif cfg.model.name == "hypergraph_ewl":
                model = HypergraphESubtree(normalize=cfg.model.normalize,
                                        n_iter=4,threshold=2)
            elif cfg.model.name == 'graph_persistencewl':
                model = PersistenceWL(p=1,normalize=cfg.model.normalize)
            elif cfg.model.name=='graph_homo':
                model = GHC(types='t')
            elif cfg.model.name=='hypergraph_sck':
                model1 = HypergraphSubtreeKernel(normalize=cfg.model.normalize,n_iter=4)
                model2 = RetHGK(n_step=6)
                #model2 = GHC(normalize=True,size=6,types='c')
            elif cfg.model.name=='hypergraph_spk':
                model1 = HypergraphSubtreeKernel(normalize=cfg.model.normalize,n_iter=4)
                model2 = GHC(normalize=True,size=6,types='c')
                #model2 = ShortestPath(normalize=cfg.model.normalize)
            elif cfg.model.name=='hypergraph_soft':
                model = RetHGWeightKernel(n_step=4,p=2)
            elif cfg.model.name=='hypergraph_ve':
                model = HypergraphSubtreeKernel(normalize=cfg.model.normalize,n_iter=0)
            elif cfg.model.name=='hypergraph_oa':
                model = HypergraphSubtreeKernel(normalize=cfg.model.normalize,_oa=True,n_iter=4)
            else:
                raise NotImplementedError

            test_res, test_all_res = [], defaultdict(list)
            time_and_memo=defaultdict(list)
            for fold_idx, (train_idx, test_idx) in enumerate(n_fold_idx):
                _x_list, _y_list = deepcopy(x_list), deepcopy(y_list)
                train_x_list, train_y_list, test_x_list, test_y_list = [], [], [], []
                for idx in train_idx:
                    train_x_list.append(_x_list[idx])
                    train_y_list.append(_y_list[idx])
                for idx in test_idx:
                    test_x_list.append(_x_list[idx])
                    test_y_list.append(_y_list[idx])

                train_y, test_y = np.array(train_y_list), np.array(test_y_list)
                st_time=time.perf_counter()

                if cfg.model.name in ['hypergraph_sck','hypergraph_spk']:
                    K_train1 = model1.fit_transform(train_x_list).cpu().numpy()
                    K_test1 = model1.transform(test_x_list).cpu().numpy()         
                    K_train2 = model2.fit_transform(train_x_list).cpu().numpy()
                    K_test2 = model2.transform(test_x_list).cpu().numpy()  
                    #------#  
                    K_train=K_train1+K_train2
                    K_test=K_test1+K_test2 
                elif cfg.model.name in ['hypergraph_ewl']:
                    K_train = model._fit_transform_(train_x_list).cpu().numpy()
                    K_test = model._transform(test_x_list).cpu().numpy()
                else:
                    K_train = model.fit_transform(train_x_list).cpu().numpy()
                    K_test = model.transform(test_x_list).cpu().numpy()            
                els_time = time.perf_counter()-st_time
                print(f"K_train shape :{K_train.shape}")
                print(f"K_test shape :{K_test.shape}")
                print(f"time: {els_time:.4f}s")
                if 'subtree' in cfg.model.name or 'wl' in cfg.model.name:
                    print(f"tree shape: {model.train_cnt.size(1)}")
                    time_and_memo['tree cnt'].append(len(model._subtree_map))
                # --------------------------------------------------------------
                test_val, best_res = train_infer_SVM(
                    K_train, train_y, K_test, test_y, multi_label
                )
                # --------------------------------------------------------------
                print(f"[{fold_idx+1}/{len(n_fold_idx)}] test results: {test_val:.4f}")
                time_and_memo['time'].append(els_time)

                for k, v in best_res.items():
                    test_all_res[k].append(v)

            res = {k: sum(v) / len(v) for k, v in test_all_res.items()}
            tms = {k: sum(v) / len(v) for k,v in time_and_memo.items()}
            print(f"mean test results: {' | '.join([f'{k}:{v:.5f}' for k, v in res.items()])}")
            print(f"time and memory: {' | '.join([f'{k}:{v:.4f}' for k, v in tms.items()])}")
            print("--------------------------------------------------")     
            res_var={k:sum((x-res[k])**2 for x in v)/len(v) for k,v in test_all_res.items()}    
            exp_dict['name'].append(data_items)
            exp_dict['method'].append(method)
            #exp_dict['max_acc']=max(best_res['acc'])
            exp_dict['mean_acc'].append(round(res['acc'],6))
            exp_dict['f1_ma'].append(round(res['f1_macro'],6))
            exp_dict['f1_weight'].append(round(res['f1_weighted'],6))   
            #--#    
            exp_dict['var_acc'].append(round(res_var['acc'],6))
            exp_dict['var_f1_ma'].append(round(res_var['f1_macro'],6))
            exp_dict['var_f1_weight'].append(round(res_var['f1_weighted'],6))     
    file_path='Result'
    curr_time=datetime.datetime.now().strftime("%d-%H-%M")
    name=file_path+'_'+curr_time+'.csv'  
    
    df=pd.DataFrame(exp_dict)
    df.to_csv(name,index=True)  

def _ablation_exp():
    ##
    # 配置日志系统
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("app.log"),  # 日志记录到文件
                            logging.StreamHandler()          # 日志输出到控制台
                        ])
    # 创建Logger对象
    logger = logging.getLogger('ExampleLogger')
    
    #---#
    yaml_file='/home/zhangyifan/alive/HIC/ml_config.yaml'
    with open(yaml_file,'r') as f:
        f_cont=yaml.safe_load(f)
        cfg=DictConfig(f_cont)
    methods=[
        'hypergraph_rethgk',
        #'hypergraph_sck',
        #'hypergraph_spk',
             ]
    exp_dict=defaultdict()
    exp_dict={
        'name':[],
        'method':[],
        'mean_acc':[],
        'n_step':[],
        #'var_acc':[],
        'f1_ma':[],
        #'var_f1_ma':[],
        'f1_weight':[],
        #'var_f1_weight':[]
    }
    datasets={
        1: "stream_player",
        2: "IMDB_wri_form",
        3: "IMDB_wri_genre",
        4: "IMDB_dir_form",
        5: "IMDB_dir_genre",
        # 6: "MUTAG",
        # 7: "NCI1",
        # 8: "PROTEINS",
        # 9: 'RHG_3',
        # 10:"RHG_10",
        #11:"IMDBBINARY",
        #12:"IMDBMULTI"
    } 
    #--init model---# 
    
    # 
    for _n_step in range(2,9):
        for idx,data_items in datasets.items():
            cfg.data.name=data_items
            for method in methods:
                cfg.model.name=method
                #------#
                if cfg.model.name in [
                    "hypergraph_rooted",
                    "hypergraph_directed_line",
                    "hypergraph_subtree",
                    "hypergraph_hyedge",
                    'hypergraph_ewl',
                    "hypergraph_rethgk",
                    "hypergraph_sck",
                    "hypergraph_spk",
                    'hypergraph_soft',
                ]:
                    model_type = "hypergraph"
                else:
                    model_type = "graph"    
                print(OmegaConf.to_yaml(cfg))
                dhg.random.set_seed(cfg.seed)
                x_list, y_list, meta = load_data(
                    cfg.data.name, cfg.data.root, cfg.data.degree_as_tag, model_type
                )
                multi_label = meta["multi_label"]
                n_classes = meta["n_classes"]

                n_fold_idx = separate_data(x_list, y_list, cfg.data.n_fold, cfg.seed)
                #-------#
                if cfg.model.name == "graph_subtree":
                    model = GraphSubtreeKernel(normalize=cfg.model.normalize)
                elif cfg.model.name=='graph_homo':
                    model = GHC(types='t')
                elif cfg.model.name=='hypergraph_sck':
                    # model1 = HypergraphESubtree(normalize=cfg.model.normalize,
                    #                             n_iter=4,threshold=1)
                    model1 = HypergraphSubtreeKernel(normalize=cfg.model.normalize,n_iter=2)
                    model2 = RetHGK(n_step=_n_step,gamma=2.5)
                    #model2 = GHC(normalize=True,size=6,types='c')
                elif cfg.model.name=='hypergraph_spk':
                    model1 = HypergraphSubtreeKernel(normalize=cfg.model.normalize,n_iter=2)
                    model2 = GHC(normalize=True,size=2*_n_step,types='c')
                    #model2 = ShortestPath(normalize=cfg.model.normalize)
                elif cfg.model.name == 'hypergraph_rethgk':
                    model = RetHGK(n_step=_n_step,gamma=2.5)
                elif cfg.model.name=='hypergraph_soft':
                    model = RetHGWeightKernel(n_step=4,p=2)
                else:
                    raise NotImplementedError

                test_res, test_all_res = [], defaultdict(list)
                time_and_memo=defaultdict(list)
                for fold_idx, (train_idx, test_idx) in enumerate(n_fold_idx):
                    _x_list, _y_list = deepcopy(x_list), deepcopy(y_list)
                    train_x_list, train_y_list, test_x_list, test_y_list = [], [], [], []
                    for idx in train_idx:
                        train_x_list.append(_x_list[idx])
                        train_y_list.append(_y_list[idx])
                    for idx in test_idx:
                        test_x_list.append(_x_list[idx])
                        test_y_list.append(_y_list[idx])

                    train_y, test_y = np.array(train_y_list), np.array(test_y_list)
                    st_time=time.perf_counter()

                    if cfg.model.name in ['hypergraph_sck','hypergraph_spk']:
                        K_train1 = model1.fit_transform(train_x_list).cpu().numpy()
                        K_test1 = model1.transform(test_x_list).cpu().numpy()         
                        K_train2 = model2.fit_transform(train_x_list).cpu().numpy()
                        K_test2 = model2.transform(test_x_list).cpu().numpy()  
                        #------#  
                        K_train=K_train1+K_train2
                        K_test=K_test1+K_test2 
                    elif cfg.model.name in ['hypergraph_ewl']:
                        K_train = model._fit_transform_(train_x_list).cpu().numpy()
                        K_test = model._transform(test_x_list).cpu().numpy()
                    else:
                        K_train = model.fit_transform(train_x_list).cpu().numpy()
                        K_test = model.transform(test_x_list).cpu().numpy()            
                    els_time = time.perf_counter()-st_time
                    print(f"K_train shape :{K_train.shape}")
                    print(f"K_test shape :{K_test.shape}")
                    print(f"time: {els_time:.4f}s")
                    if 'subtree' in cfg.model.name or 'wl' in cfg.model.name:
                        print(f"tree shape: {model.train_cnt.size(1)}")
                        time_and_memo['tree cnt'].append(len(model._subtree_map))
                    # --------------------------------------------------------------
                    test_val, best_res = train_infer_SVM(
                        K_train, train_y, K_test, test_y, multi_label
                    )
                    # --------------------------------------------------------------
                    print(f"[{fold_idx+1}/{len(n_fold_idx)}] test results: {test_val:.4f}")
                    time_and_memo['time'].append(els_time)

                    for k, v in best_res.items():
                        test_all_res[k].append(v)

                res = {k: sum(v) / len(v) for k, v in test_all_res.items()}
                tms = {k: sum(v) / len(v) for k,v in time_and_memo.items()}
                print(f"mean test results: {' | '.join([f'{k}:{v:.5f}' for k, v in res.items()])}")
                print(f"time and memory: {' | '.join([f'{k}:{v:.4f}' for k, v in tms.items()])}")
                print("--------------------------------------------------")     
                res_var={k:sum((x-res[k])**2 for x in v)/len(v) for k,v in test_all_res.items()}    
                exp_dict['name'].append(data_items)
                exp_dict['method'].append(method)
                #exp_dict['max_acc']=max(best_res['acc'])
                exp_dict['mean_acc'].append(round(res['acc'],6))
                exp_dict['f1_ma'].append(round(res['f1_macro'],6))
                exp_dict['f1_weight'].append(round(res['f1_weighted'],6))   
                exp_dict['n_step'].append(_n_step)
                #--#    
                # exp_dict['var_acc'].append(round(res_var['acc'],6))
                # exp_dict['var_f1_ma'].append(round(res_var['f1_macro'],6))
                # exp_dict['var_f1_weight'].append(round(res_var['f1_weighted'],6))     
    file_path='Result'
    curr_time=datetime.datetime.now().strftime("%d-%H-%M")
    name=file_path+'_'+curr_time+'.csv'  
    
    df=pd.DataFrame(exp_dict)
    df.to_csv(name,index=True)      

def _ablation_exp_d():

    #---#
    yaml_file='/home/zhangyifan/alive/HIC/ml_config.yaml'
    with open(yaml_file,'r') as f:
        f_cont=yaml.safe_load(f)
        cfg=DictConfig(f_cont)
    methods=[
        'hypergraph_sck',
             ]
    exp_dict=defaultdict()
    exp_dict={
        'name':[],
        'method':[],
        'mean_acc':[],
        'D0':[],
        # 'n_step':[],
        #'var_acc':[],
        'f1_ma':[],
        #'var_f1_ma':[],
        'f1_weight':[],
        #'var_f1_weight':[]
    }
    datasets={
        # 1: "stream_player",
        # 2: "IMDB_wri_form",
        # 3: "IMDB_wri_genre",
        # 4: "IMDB_dir_form",
        # 5: "IMDB_dir_genre",
        6: "MUTAG",
        7: "NCI1",
        8: "PROTEINS",
        # 9: 'RHG_3',
        # 10:"RHG_10",
        11:"IMDBBINARY",
        12:"IMDBMULTI"
    } 
    #--init model---# 
    
    # 
    D0=[10,50,100,200,1000,2000]
    for d0 in D0:
        for idx,data_items in datasets.items():
            cfg.data.name=data_items
            for method in methods:
                cfg.model.name=method
                #------#
                if cfg.model.name in [
                    "hypergraph_rooted",
                    "hypergraph_directed_line",
                    "hypergraph_subtree",
                    "hypergraph_hyedge",
                    'hypergraph_ewl',
                    "hypergraph_rethgk",
                    "hypergraph_sck",
                    "hypergraph_spk",
                    'hypergraph_soft',
                ]:
                    model_type = "hypergraph"
                else:
                    model_type = "graph"    
                print(OmegaConf.to_yaml(cfg))
                dhg.random.set_seed(cfg.seed)
                x_list, y_list, meta = load_data(
                    cfg.data.name, cfg.data.root, cfg.data.degree_as_tag, model_type
                )
                multi_label = meta["multi_label"]
                n_classes = meta["n_classes"]

                n_fold_idx = separate_data(x_list, y_list, cfg.data.n_fold, cfg.seed)
                #-------#
                if cfg.model.name == "graph_subtree":
                    model = GraphSubtreeKernel(normalize=cfg.model.normalize)
                elif cfg.model.name=='hypergraph_sck':
                    # model1 = HypergraphESubtree(normalize=cfg.model.normalize,
                    #                             n_iter=4,threshold=1)
                    model1 = HypergraphSubtreeKernel(normalize=cfg.model.normalize,n_iter=2)
                    model2 = RetHGK(n_step=5,D=d0)
                    #model2 = GHC(normalize=True,size=6,types='c')
                else:
                    raise NotImplementedError

                test_res, test_all_res = [], defaultdict(list)
                time_and_memo=defaultdict(list)
                for fold_idx, (train_idx, test_idx) in enumerate(n_fold_idx):
                    _x_list, _y_list = deepcopy(x_list), deepcopy(y_list)
                    train_x_list, train_y_list, test_x_list, test_y_list = [], [], [], []
                    for idx in train_idx:
                        train_x_list.append(_x_list[idx])
                        train_y_list.append(_y_list[idx])
                    for idx in test_idx:
                        test_x_list.append(_x_list[idx])
                        test_y_list.append(_y_list[idx])

                    train_y, test_y = np.array(train_y_list), np.array(test_y_list)
                    st_time=time.perf_counter()

                    if cfg.model.name in ['hypergraph_sck','hypergraph_spk']:
                        K_train1 = model1.fit_transform(train_x_list).cpu().numpy()
                        K_test1 = model1.transform(test_x_list).cpu().numpy()         
                        K_train2 = model2.fit_transform(train_x_list).cpu().numpy()
                        K_test2 = model2.transform(test_x_list).cpu().numpy()  
                        #------#  
                        K_train=K_train1+K_train2
                        K_test=K_test1+K_test2 
                    else:
                        K_train = model.fit_transform(train_x_list).cpu().numpy()
                        K_test = model.transform(test_x_list).cpu().numpy()            
                    els_time = time.perf_counter()-st_time
                    print(f"K_train shape :{K_train.shape}")
                    print(f"K_test shape :{K_test.shape}")
                    print(f"time: {els_time:.4f}s")
                    if 'subtree' in cfg.model.name or 'wl' in cfg.model.name:
                        print(f"tree shape: {model.train_cnt.size(1)}")
                        time_and_memo['tree cnt'].append(len(model._subtree_map))
                    # --------------------------------------------------------------
                    test_val, best_res = train_infer_SVM(
                        K_train, train_y, K_test, test_y, multi_label
                    )
                    # --------------------------------------------------------------
                    print(f"[{fold_idx+1}/{len(n_fold_idx)}] test results: {test_val:.4f}")
                    time_and_memo['time'].append(els_time)

                    for k, v in best_res.items():
                        test_all_res[k].append(v)

                res = {k: sum(v) / len(v) for k, v in test_all_res.items()}
                tms = {k: sum(v) / len(v) for k,v in time_and_memo.items()}
                print(f"mean test results: {' | '.join([f'{k}:{v:.5f}' for k, v in res.items()])}")
                print(f"time and memory: {' | '.join([f'{k}:{v:.4f}' for k, v in tms.items()])}")
                print("--------------------------------------------------")     
                res_var={k:sum((x-res[k])**2 for x in v)/len(v) for k,v in test_all_res.items()}    
                exp_dict['name'].append(data_items)
                exp_dict['method'].append(method)
                #exp_dict['max_acc']=max(best_res['acc'])
                exp_dict['mean_acc'].append(round(res['acc'],6))
                exp_dict['f1_ma'].append(round(res['f1_macro'],6))
                exp_dict['f1_weight'].append(round(res['f1_weighted'],6))   
                exp_dict['D0'].append(d0)
                #--#    
                # exp_dict['var_acc'].append(round(res_var['acc'],6))
                # exp_dict['var_f1_ma'].append(round(res_var['f1_macro'],6))
                # exp_dict['var_f1_weight'].append(round(res_var['f1_weighted'],6))     
    file_path='Result'
    curr_time=datetime.datetime.now().strftime("%d-%H-%M")
    name=file_path+'_'+curr_time+'.csv'  
    
    df=pd.DataFrame(exp_dict)
    df.to_csv(name,index=True)   

def _ablation_clf():
        ##
    # 配置日志系统
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("app.log"),  # 日志记录到文件
                            logging.StreamHandler()          # 日志输出到控制台
                        ])
    # 创建Logger对象
    logger = logging.getLogger('ExampleLogger')
    
    #---#
    yaml_file='/home/zhangyifan/alive/HIC/ml_config.yaml'
    with open(yaml_file,'r') as f:
        f_cont=yaml.safe_load(f)
        cfg=DictConfig(f_cont)
    methods=[
        #'hypergraph_sck',
            #"hypergraph_spk",
            #'graph_subtree',
            # 'graph_homo',
            #'graph_retgk',
            'hypergraph_subtree',
            'hypergraph_hyedge',
            'hypergraph_sck',
            # #'graphlet_sampling',
            # #'hypergraph_rooted',
            # 'hypergraph_rethgk',
            #'graph_shortestpath',
             ]
    exp_dict=defaultdict()
    exp_dict={
        'name':[],
        'method':[],
        'mean_acc':[],
        'var_acc':[],
        'f1_ma':[],
        'var_f1_ma':[],
        'f1_weight':[],
        'var_f1_weight':[],
        'clf':[]
    }
    datasets={
        1: "stream_player",
        2: "IMDB_wri_form",
        3: "IMDB_wri_genre",
        4: "IMDB_dir_form",
        5: "IMDB_dir_genre",
        # 6: "MUTAG",
        # 7: "NCI1",
        # 8: "PROTEINS",
        # 9: 'RHG_3',
        # 10:"RHG_10",
        #11:"IMDBBINARY",
        #12:"IMDBMULTI"
    } 
    #--init model---# 
    #clfs=['knn','rf','by']
    clfs=['lr','ada','boot']
    # 
    for clf in clfs:
        for idx,data_items in datasets.items():
            cfg.data.name=data_items
            for method in methods:
                cfg.model.name=method
                #------#
                if cfg.model.name in [
                    "hypergraph_rooted",
                    "hypergraph_directed_line",
                    "hypergraph_subtree",
                    "hypergraph_hyedge",
                    'hypergraph_ewl',
                    "hypergraph_rethgk",
                    "hypergraph_sck",
                    "hypergraph_spk",
                    'hypergraph_soft',
                ]:
                    model_type = "hypergraph"
                else:
                    model_type = "graph"    
                print(OmegaConf.to_yaml(cfg))
                dhg.random.set_seed(cfg.seed)
                x_list, y_list, meta = load_data(
                    cfg.data.name, cfg.data.root, cfg.data.degree_as_tag, model_type
                )
                multi_label = meta["multi_label"]
                n_classes = meta["n_classes"]

                n_fold_idx = separate_data(x_list, y_list, cfg.data.n_fold, cfg.seed)
                #-------#
                if cfg.model.name == "graph_subtree":
                    model = GraphSubtreeKernel(normalize=cfg.model.normalize)
                elif cfg.model.name == "graphlet_sampling":
                    model = GraphletSampling(normalize=cfg.model.normalize, sampling={})
                elif cfg.model.name == 'graph_retgk':
                    model = RetGK(n_step=3)
                elif cfg.model.name == 'hypergraph_rethgk':
                    model = RetHGK(n_step=3,gamma=2)
                elif cfg.model.name == 'graph_shortestpath':
                    model = ShortestPath(normalize=True,D=40)
                elif cfg.model.name == "hypergraph_rooted":
                    model = HypergraphRootedKernel(normalize=cfg.model.normalize)
                elif cfg.model.name == "hypergraph_directed_line":
                    model = HypergraphDirectedLineKernel(normalize=cfg.model.normalize)
                elif cfg.model.name == "hypergraph_subtree":
                    model = HypergraphSubtreeKernel(normalize=cfg.model.normalize,n_iter=4)
                elif cfg.model.name == "hypergraph_hyedge":
                    model = HypergraphHyedgeKernel(normalize=cfg.model.normalize)
                elif cfg.model.name == "hypergraph_ewl":
                    model = HypergraphESubtree(normalize=cfg.model.normalize,
                                            n_iter=4,threshold=2)
                elif cfg.model.name == 'graph_persistencewl':
                    model = PersistenceWL(p=1,normalize=cfg.model.normalize)
                elif cfg.model.name=='graph_homo':
                    model = GHC(types='t')
                elif cfg.model.name=='hypergraph_sck':
                    # model1 = HypergraphESubtree(normalize=cfg.model.normalize,
                    #                             n_iter=4,threshold=1)
                    model1 = HypergraphSubtreeKernel(normalize=cfg.model.normalize,n_iter=4)
                    model2 = RetHGK(n_step=6,D=50)
                    #model2 = GHC(normalize=True,size=6,types='c')
                elif cfg.model.name=='hypergraph_spk':
                    model1 = HypergraphSubtreeKernel(normalize=cfg.model.normalize,n_iter=4)
                    model2 = GHC(normalize=True,size=6,types='c')
                    #model2 = ShortestPath(normalize=cfg.model.normalize)
                elif cfg.model.name=='hypergraph_soft':
                    model = RetHGWeightKernel(n_step=4,p=2)
                else:
                    raise NotImplementedError

                test_res, test_all_res = [], defaultdict(list)
                time_and_memo=defaultdict(list)
                for fold_idx, (train_idx, test_idx) in enumerate(n_fold_idx):
                    _x_list, _y_list = deepcopy(x_list), deepcopy(y_list)
                    train_x_list, train_y_list, test_x_list, test_y_list = [], [], [], []
                    for idx in train_idx:
                        train_x_list.append(_x_list[idx])
                        train_y_list.append(_y_list[idx])
                    for idx in test_idx:
                        test_x_list.append(_x_list[idx])
                        test_y_list.append(_y_list[idx])

                    train_y, test_y = np.array(train_y_list), np.array(test_y_list)
                    st_time=time.perf_counter()

                    if cfg.model.name in ['hypergraph_sck','hypergraph_spk']:
                        K_train1 = model1.fit_transform(train_x_list).cpu().numpy()
                        K_test1 = model1.transform(test_x_list).cpu().numpy()         
                        K_train2 = model2.fit_transform(train_x_list).cpu().numpy()
                        K_test2 = model2.transform(test_x_list).cpu().numpy()  
                        #------#  
                        K_train=K_train1+K_train2
                        K_test=K_test1+K_test2 
                    elif cfg.model.name in ['hypergraph_ewl']:
                        K_train = model._fit_transform_(train_x_list).cpu().numpy()
                        K_test = model._transform(test_x_list).cpu().numpy()
                    else:
                        K_train = model.fit_transform(train_x_list).cpu().numpy()
                        K_test = model.transform(test_x_list).cpu().numpy()            
                    els_time = time.perf_counter()-st_time
                    print(f"K_train shape :{K_train.shape}")
                    print(f"K_test shape :{K_test.shape}")
                    print(f"time: {els_time:.4f}s")
                    if 'subtree' in cfg.model.name or 'wl' in cfg.model.name:
                        print(f"tree shape: {model.train_cnt.size(1)}")
                        time_and_memo['tree cnt'].append(len(model._subtree_map))
                    # --------------------------------------------------------------
                    # test_val, best_res = train_infer_SVM(
                    #     K_train, train_y, K_test, test_y, multi_label
                    # )
                    test_val,best_res = _ml_train_infer(K_train,train_y,K_test,test_y,method=clf)
                    # --------------------------------------------------------------
                    print(f"[{fold_idx+1}/{len(n_fold_idx)}] test results: {test_val:.4f}")
                    time_and_memo['time'].append(els_time)

                    for k, v in best_res.items():
                        test_all_res[k].append(v)

                res = {k: sum(v) / len(v) for k, v in test_all_res.items()}
                tms = {k: sum(v) / len(v) for k,v in time_and_memo.items()}
                print(f"mean test results: {' | '.join([f'{k}:{v:.5f}' for k, v in res.items()])}")
                print(f"time and memory: {' | '.join([f'{k}:{v:.4f}' for k, v in tms.items()])}")
                print("--------------------------------------------------")     
                res_var={k:sum((x-res[k])**2 for x in v)/len(v) for k,v in test_all_res.items()}    
                exp_dict['name'].append(data_items)
                exp_dict['method'].append(method)
                #exp_dict['max_acc']=max(best_res['acc'])
                exp_dict['mean_acc'].append(round(res['acc'],6))
                exp_dict['f1_ma'].append(round(res['f1_macro'],6))
                exp_dict['f1_weight'].append(round(res['f1_weighted'],6))   
                #--#    
                exp_dict['var_acc'].append(round(res_var['acc'],6))
                exp_dict['var_f1_ma'].append(round(res_var['f1_macro'],6))
                exp_dict['var_f1_weight'].append(round(res_var['f1_weighted'],6))     
                exp_dict['clf'].append(clf)
    file_path='Result'
    curr_time=datetime.datetime.now().strftime("%d-%H-%M")
    name=file_path+'_'+curr_time+'.csv'  
    
    df=pd.DataFrame(exp_dict)
    df.to_csv(name,index=True)  

    pass
if __name__ == "__main__":
    multi_exp()
    #_ablation_exp()
    #_ablation_exp_d()
    #_ablation_clf()
import dhg
import time
import hydra
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf

from models import (
    #----------------#
    HypergraphRootedKernel,
    GraphSubtreeKernel,
    GraphletSampling,
    HypergraphDirectedLineKernel,
    HypergraphSubtreeKernel,
    HypergraphHyedgeKernel,
    HighOrderMotif,
    #----------------#
    GHC,
    RetGK,
    RetHGK,
    ShortestPath,
    PersistenceWL,
    HypergraphESubtree,
    RetHGWeightKernel,
    G2WLSubtree,
    HypergraphHSVD,
    #----------------#
)
from utils import load_data, separate_data, train_infer_SVM,_ml_train_cluster

print = logging.info
multi_label, criterion = None, None


@hydra.main(config_path=".", config_name="cl_config",version_base='1.2')
def main(cfg: DictConfig):
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
        'hypergraph_v_e',
        'hypergraph_sum_wl',
        'hypergraph_hsvd',
        'hypergraph_motifs',
        'hypergraph_mochy',
    ]:
        model_type = "hypergraph"
    else:
        model_type = "graph"
    print(OmegaConf.to_yaml(cfg))
    global multi_label, criterion
    dhg.random.set_seed(cfg.seed)
    x_list, y_list, meta = load_data(
        cfg.data.name, cfg.data.root, cfg.data.degree_as_tag, model_type
    )
    multi_label = meta["multi_label"]
    n_classes = meta["n_classes"]

    n_fold_idx = separate_data(x_list, y_list, cfg.data.n_fold, cfg.seed)

    if cfg.model.name == "graph_subtree":
        model = GraphSubtreeKernel(normalize=cfg.model.normalize)
    elif cfg.model.name == "graphlet_sampling":
        model = GraphletSampling(normalize=cfg.model.normalize, sampling={})
    elif cfg.model.name == 'graph_retgk':
        model = RetGK(n_step=3)
    elif cfg.model.name == 'hypergraph_rethgk':
        model = RetHGK(n_step=3)
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
    elif cfg.model.name in [ "hypergraph_ewl",'hypergraph_sum_wl']:
        model = HypergraphESubtree(normalize=cfg.model.normalize,
                                   n_iter=2,threshold=2)
    elif cfg.model.name in 'graph_2wl':
        model = G2WLSubtree(normalize=cfg.model.normalize,n_iter=2)
    elif cfg.model.name == 'graph_persistencewl':
        model = PersistenceWL(p=1,normalize=cfg.model.normalize)
    elif cfg.model.name=='graph_homo':
        model = GHC(types='t')
    elif cfg.model.name=='hypergraph_sck':
        # model1 = HypergraphESubtree(normalize=cfg.model.normalize,
        #                             n_iter=4,threshold=1)
        model1 = HypergraphSubtreeKernel(normalize=cfg.model.normalize,n_iter=2)
        #model2 = RetHGK(n_step=3)
        model2 = GHC(normalize=True,size=6,types='c')
    elif cfg.model.name=='hypergraph_spk':
        model1 = HypergraphSubtreeKernel(normalize=cfg.model.normalize,n_iter=2)
        model2 = RetHGK(n_step=5)
        #model2 = ShortestPath(normalize=cfg.model.normalize)
    elif cfg.model.name=='hypergraph_motifs':
        model = HighOrderMotif(normalize=cfg.model.normalize)
    elif cfg.model.name=='hypergraph_mochy':
        model = HighOrderMotif(normalize=cfg.model.normalize)
    elif cfg.model.name=='hypergraph_soft':
        model = RetHGWeightKernel(n_step=4,p=2)
    elif cfg.model.name=='hypergraph_hsvd':
        model = HypergraphHSVD(k=20,normalize=cfg.model.normalize)
    elif cfg.model.name=='hypergraph_v_e':
        model = HypergraphSubtreeKernel(normalize=cfg.model.normalize,n_iter=0)
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
        elif cfg.model.name in ['hypergraph_motifs']:
            ft_path=f'/home/zhangyifan/alive/higher-order-motifs/{cfg.data.name}_results.pickle'
            model._load_ft(ft_path)
            K_train = model.fit_transform(train_idx).cpu().numpy()
            K_test = model.transform(test_idx).cpu().numpy()
        elif cfg.model.name in ['hypergraph_mochy']:
            ft_path=f'/home/zhangyifan/alive/MoCHy/{cfg.data.name}_results.pickle'
            model._load_ft(ft_path)
            K_train = model.fit_transform(train_idx).cpu().numpy()
            K_test = model.transform(test_idx).cpu().numpy()
        elif cfg.model.name in ['hypergraph_ewl']:
            # K_train = model._fit_transform_(train_x_list).cpu().numpy()
            # K_test = model._transform(test_x_list).cpu().numpy()
            K_train = model.fit_transform(train_x_list).cpu().numpy()
            K_test = model.transform(test_x_list).cpu().numpy()
        elif cfg.model.name in ['hypergraph_sum_wl']:
            K_train = model._fit_transform_sum(train_x_list).cpu().numpy()
            K_test = model._transform_sum(test_x_list).cpu().numpy()
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
        # # --------------------------------------------------------------
        test_val,best_res = _ml_train_cluster(
            K_train,train_y,K_test,test_y,method='kmeans',n_clusters=n_classes,
        )
        # test_val, best_res = train_infer_SVM(
        #     K_train, train_y, K_test, test_y, multi_label
        # )
        
        # # --------------------------------------------------------------
        print(f"[{fold_idx+1}/{len(n_fold_idx)}] test results: {test_val:.4f}")
        time_and_memo['time'].append(els_time)

        for k, v in best_res.items():
            test_all_res[k].append(v)

    res = {k: sum(v) / len(v) for k, v in test_all_res.items()}
    tms = {k: sum(v) / len(v) for k,v in time_and_memo.items()}
    print(f"mean test results: {' | '.join([f'{k}:{v:.5f}' for k, v in res.items()])}")
    print(f"time and memory: {' | '.join([f'{k}:{v:.4f}' for k, v in tms.items()])}")
    print("--------------------------------------------------")
    return test_res


if __name__ == "__main__":
    main()

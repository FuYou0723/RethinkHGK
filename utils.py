import dhg
import math
import scipy
import torch
import numpy as np
from dhg import Graph, Hypergraph
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from skmultilearn.problem_transform import BinaryRelevance

from sklearn.model_selection import GridSearchCV

g2hg_func = dhg.Hypergraph.from_graph
hg2g_func = dhg.Graph.from_hypergraph_clique
hg2gstar_func=dhg.Graph.from_hypergraph_star
def read_labels(filename):
    labels=[]
    with open(filename) as f:
        labels=f.readlines()
        labels=[label.strip() for label in labels]
    return labels

def load_data_(name,root,degree_as_tag,model_type):
    if name in ['DD','PTC-MM']:
        data_type='graph'
        folder=name # 
    import igraph as ig
    graphs=[ig.read(filename) for filename in name]    
    labels=read_labels(name)
    x_list=[]
    for graph in graphs:
        if 'label' not in graph.vs.attributes():
            graph.vs['label']=[0]*len(graph.vs)
        if 'weight' in graph.es.attributes():
            graph.es['weight']=[0]*len(graph.es)
            
        d=dhg.Graph(num_v=graph.vs,e_list=graph.get_edgelist())
        if data_type == "graph" and model_type == "hypergraph":
            trans_func = g2hg_func
        elif data_type == "hypergraph" and model_type == "graph":
            trans_func = hg2g_func
        else:
            trans_func = lambda x: x     
        d=trans_func(d)   
        x_list.append(
            {
                "num_v": d.num_v,
                "num_e": d.num_e,
                "v_lbl": graph.vs['label'],
                "g_lbl": graph.es['weight'],
                "e_list": d.e[0],
                "dhg": d,
#                "dhgg":ds,
            }
        )
    
        

def load_data(name, root, degree_as_tag, model_type):
    # graph dataset
    if name in ["RG_macro", "RG_sub"]:
        data_type = "graph"
        folder = "RG"
        multi_label = False
    elif name in ["MUTAG", "NCI1", "PROTEINS", "IMDBMULTI", "IMDBBINARY"]:
        data_type = "graph"
        folder = name
        multi_label = False
    elif name in ["RHG_3", "RHG_10", "RHG_table", "RHG_pyramid"]:
        data_type = "hypergraph"
        folder = "RHG"
        multi_label = False
    elif name in ["stream_player"]:
        data_type = "hypergraph"
        folder = "STEAM"
        multi_label = False
    elif name in ["IMDB_dir_genre_m"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = True
    elif name in ["IMDB_dir_form", "IMDB_dir_genre"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = False
    elif name in ["IMDB_wri_genre_m"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = True
    elif name in ["IMDB_wri_form", "IMDB_wri_genre"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = False
    elif name in ["twitter_friend"]:
        data_type = "hypergraph"
        folder = "TWITTER"
        multi_label = False
    elif name.startswith("p"):
        data_type = 'hypergraph'
        folder = "Performance"
        multi_label = False
    else:
        raise NotImplementedError
    if data_type == "graph" and model_type == "hypergraph":
        trans_func = g2hg_func
    elif data_type == "hypergraph" and model_type == "graph":
        trans_func = hg2g_func
    else:
        trans_func = lambda x: x
        
    # read data
    x_list = []
    with open(f"{root}/{data_type}/{folder}/{name}.txt", "r") as f:
        n_g = int(f.readline().strip())
        for _ in range(n_g):
            row = f.readline().strip().split()
            num_v, num_e = int(row[0]), int(row[1])
            g_lbl = [int(x) for x in row[2:]]
            v_lbl = f.readline().strip().split()
            v_lbl = [[int(x) for x in s.split("/")] for s in v_lbl]
            e_list = []
            for _ in range(num_e):
                row = f.readline().strip().split()
                e_list.append([int(x) for x in row])
            if data_type == "graph":
                d = Graph(num_v, e_list)
            else:
                d = Hypergraph(num_v, e_list)
            #d,_ = trans_func(d)
            d=trans_func(d)
            ds=None
            if isinstance(d,dhg.Hypergraph):
                ds,_=Graph.from_hypergraph_star(d)
                #ds=Graph.from_hypergraph_clique(d)
            x_list.append(
                {
                    "num_v": d.num_v,
                    "num_e": d.num_e,
                    "v_lbl": v_lbl,
                    "g_lbl": g_lbl,
                    "e_list": d.e[0],
                    "dhg": d,
                    "dhgg":ds,
                }
            )
    for x in x_list:
        if degree_as_tag:
            x["v_lbl"] = [int(v) for v in x["dhg"].deg_v]
        if isinstance(x["dhg"], Graph):
            x["e_lbl"] = [2] * x["num_e"]
        else:
            x["e_lbl"] = [int(e) for e in x["dhg"].deg_e]

    v_lbl_set, e_lbl_set, g_lbl_set = set(), set(), set()
    for x in x_list:
        if isinstance(x["v_lbl"][0], list):
            for v_lbl in x["v_lbl"]:
                v_lbl_set.update(v_lbl)
        else:
            v_lbl_set.update(x["v_lbl"])
        e_lbl_set.update(x["e_lbl"])
        g_lbl_set.update(x["g_lbl"])
    # re-map labels
    v_lbl_map = {x: i for i, x in enumerate(sorted(v_lbl_set))}
    e_lbl_map = {x: i for i, x in enumerate(sorted(e_lbl_set))}
    g_lbl_map = {x: i for i, x in enumerate(sorted(g_lbl_set))}
    ft_dim, n_classes = len(v_lbl_set), len(g_lbl_set)
    for x in x_list:
        x["g_lbl"] = [g_lbl_map[c] for c in x["g_lbl"]]
        if isinstance(x["v_lbl"][0], list):
            x["v_lbl"] = [tuple(sorted([v_lbl_map[c] for c in s])) for s in x["v_lbl"]]
        else:
            x["v_lbl"] = [v_lbl_map[c] for c in x["v_lbl"]]
        x["e_lbl"] = [e_lbl_map[c] for c in x["e_lbl"]]
        x["v_ft"] = np.zeros((x["num_v"], ft_dim))
        row_idx, col_idx = [], []
        for v_idx, v_lbls in enumerate(x["v_lbl"]):
            if isinstance(v_lbls, list) or isinstance(v_lbls, tuple):
                for v_lbl in v_lbls:
                    row_idx.append(v_idx)
                    col_idx.append(v_lbl)
            else:
                row_idx.append(v_idx)
                col_idx.append(v_lbls)
        x["v_ft"][row_idx, col_idx] = 1
    y_list = []
    if multi_label:
        for x in x_list:
            tmp = np.zeros(n_classes).astype(int)
            tmp[x["g_lbl"]] = 1
            y_list.append(tmp.tolist())
    else:
        y_list = [g["g_lbl"][0] for g in x_list]
    meta = {
        "multi_label": multi_label,
        "data_type": data_type,
        "ft_dim": ft_dim,
        "n_classes": len(g_lbl_set),
    }
    return x_list, y_list, meta

def separate_data(x_list, y_list, n_fold, seed):
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=seed)
    n_fold_idx = []
    for train_idx, test_idx in kf.split(x_list, y_list):
        n_fold_idx.append((train_idx, test_idx))
    return n_fold_idx

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_squared_log_error
def train_SVR(train_X,train_Y,test_X,test_Y):
    svr=SVR(kernel='rbf',C=100,epsilon=0.1)
    svr.fit(train_X,train_Y)
    y_pred=svr.predict(test_X)
    mse=mean_squared_error(test_Y,y_pred)
    mae=mean_absolute_error(test_Y,y_pred)
    log_mse=0
    #log_mse=mean_squared_log_error(test_Y,y_pred)
    return mse,mae,log_mse
    

def train_infer_SVM(train_X, train_Y, test_X, test_Y, multi_label):
    if not multi_label:
        clf = SVC(kernel="precomputed")
    else:
        clf = BinaryRelevance(
            classifier=SVC(kernel="precomputed"),
            require_dense=[True, True],
        )
    clf.fit(train_X, train_Y)
    outputs = clf.predict(test_X)
    test_val, best_res = performance(outputs, test_Y, multi_label)
    return test_val, best_res
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import  LogisticRegression

def _ml_train_infer(train_X,train_Y,test_X,test_Y, method='knn'):
    if method=='knn':
        clf=KNeighborsClassifier(n_neighbors=3,
                                 weights='uniform',
                                 leaf_size=20,
                                 algorithm='kd_tree',
                                 p=2,
                                 #verbose=0,
                                 #random_seed=2024,
                                 )
    if method=='rf':
        clf=RandomForestClassifier(n_estimators=100,
                                   criterion='gini', # entropy
                                   max_depth=None,
                                   min_samples_leaf=1,
                                   max_features='auto', # auto,sqrt,log2
                                   )
    if method == 'lr':
        clf= LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
    if method=='by':
        clf=GaussianNB(var_smoothing=1e-9,
                       priors=None)
    if method=='ada':
        base_estimator = DecisionTreeClassifier(max_depth=1)
        clf=AdaBoostClassifier(base_estimator=base_estimator, 
                                    n_estimators=50, 
                                    learning_rate=1.0, 
                                    algorithm='SAMME.R', 
                                    random_state=42)
    if method=='boot':
        clf=GradientBoostingClassifier(n_estimators=100, 
                                       learning_rate=0.1, 
                                       max_depth=3, 
                                       random_state=42)
        
    clf.fit(train_X,train_Y)
    outputs = clf.predict(test_X)
    test_val,best_res = performance(outputs,test_Y,multi_label=False)
    return test_val,best_res
import numpy as np
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score,v_measure_score
#(NMI,ARI,v_measure))
def _ml_train_cluster(train_X,train_labels,test_X,test_labels,method='kmeans',n_clusters=3,random_state=42):
    '''
    训练聚类模型并评估性能  
    '''
    if method == 'kmeans':
        model=KMeans(n_clusters=n_clusters, random_state=random_state)
    else:
        raise ValueError(f"Unsupported method: {method}. Supported methods are 'kmeans', 'dbscan', 'hierarchical'.")
    # 训练模型
    model.fit(train_X)
    _train_labels=model.labels_
    # 获取聚类标签
    if hasattr(model, 'predict'):
        _test_labels = model.predict(test_X)
    else:
        _test_labels = model.fit_predict(test_X)
    
    metric=performance_cluster(_test_labels,test_labels,test_X)
    return metric['silhouette'],metric



# -------------------- Metrics ----------------------------
def calc_class_acc(preds,targets):
    preds=np.array(preds)
    targets=np.array(targets)
    classes=np.unique(targets)
    for cls_idx,cls in enumerate(classes):
        # 找到索引
        cls_indices=(targets==cls)
        # 计算对应索引的精度
        cls_acc=accuracy_score(targets[cls_indices],preds[cls_indices]) # 
        cls_sum=np.sum(targets[cls_indices]==preds[cls_indices])
        cls_total=np.sum(cls_indices)
        print(f"class index {cls_idx} acc: {cls_acc:.4f},\
            ({cls_sum}/{cls_total})")
def performance_cluster(preds,targets,test_X,with_gt=True):
    # 计算内部评估
    metrics= {}
    metrics['silhouette'] = silhouette_score(test_X, preds)
    metrics['davies_bouldin'] = davies_bouldin_score(test_X, preds)
    metrics['calinski_harabasz'] = calinski_harabasz_score(test_X, preds)
    #-----------------------#
    metrics['nmi']=metrics['ari']=metrics['acc']=metrics['f1']=0
    if with_gt:
        # 计算外部评估
        metrics['nmi'] = normalized_mutual_info_score(preds, targets)
        metrics['ari'] = adjusted_rand_score(preds, targets)
        metrics['acc'] = accuracy_score(preds, targets)
        metrics['f1'] = f1_score(preds, targets, average='macro')
    return metrics

def performance(preds: np.ndarray, targets: np.ndarray, multi_label: bool):
    if multi_label:
        if isinstance(preds, scipy.sparse.csc_matrix):
            preds = preds.todense()
        else:
            preds = (preds > 0.5).astype(int)
        # multi-label classification metric:
        # https://medium.datadriveninvestor.com/a-survey-of-evaluation-metrics-for-multilabel-classification-bb16e8cd41cd
        # acc = (preds==lbls).mean()
        # Exact Match Ratio (EMR)
        EMR = (preds == targets).all(1).mean()
        # Example-based Accuracy
        EB_acc = (
            np.logical_and(preds, targets).sum(1) / np.logical_or(preds, targets).sum(1)
        ).mean()
        # Example-based Precision
        EB_pre = np.logical_and(preds, targets).sum(1) / preds.sum(1)
        EB_pre[np.isnan(EB_pre)] = 0
        EB_pre = EB_pre.mean()
        res = {"EMR": EMR, "EB_acc": EB_acc, "EB_pre": EB_pre}
        return EMR, res
    else:
        if len(preds.shape) == 2:
            preds = np.argmax(preds, axis=1)
        calc_class_acc(preds,targets)
        acc = accuracy_score(targets, preds)
        f1_micro = f1_score(targets, preds, average="micro")
        f1_macro = f1_score(targets, preds, average="macro")
        f1_weighted = f1_score(targets, preds, average="weighted")
        res = {
            "acc": acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }
        return acc, res



if __name__ == "__main__":
    g_list, y_list, meta = load_data("MUTAG", "data", True, "graph")
    all_deg=0
    all_v=0
    for g in g_list:
        v=sum(g['dhg'].deg_v)
        num_v=g['dhg'].num_v
        all_deg+=v
        all_v+=num_v
    print(all_deg/all_v)
    # print(g_list[0])
    # g_list, y_list, meta = load_data("RHG_3", "data", True, "graph")
    # print(g_list[0])

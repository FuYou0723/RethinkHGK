import numpy as np
import torch
import dhg
from collections import defaultdict

from dhg import Graph
from collections import deque,defaultdict
import heapq
##------------hom_lib--------------------##
class UnionFind:
    def __init__(self,num_v) -> None:
        self.parents=list(range(num_v))
        
    def _find(self,x):
        if self.parents[x]!=x:
            self.parents[x]=self._find(self.parents[x]) # 路径压缩
        return self.parents[x]
    def _merge(self,x,y):
        if x!=y:
            self.parents[self._find(y)]=self._find(x)
    def _connected(self,u,v):
        return self._find(u)==self._find(v)
    def _roots(self):
        for vertex in range(len(self.parents)):
            if self.parents[vertex]==vertex:
                yield vertex
    
class GraphHomo:
    def __init__(self) -> None:
        pass
    
    def is_tree(self,graph:Graph):
        uf = UnionFind(graph.num_v)
        edges_count = 0
        for u in range(graph.num_v):
            for v in graph.nbr_v(u):
                if u<v:
                    edges_count += 1
                    if uf._connected(u, v):
                        return False
                    uf._merge(u, v)

        # 检查是否只有一个连通分量
        root = uf._find(0)
        for i in range(1, graph.num_v):
            if uf._find(i) != root:
                return False

        # 树的边数应该等于节点数减一
        # print(edges_count)
        return edges_count == graph.num_v - 1

    def connected_components(self,graph:Graph):
        uf = UnionFind(graph.num_v)
        for u in range(graph.num_v):
            for v in graph.nbr_v(u):
                uf._merge(u, v)

        components_dict = {}
        for node in range(graph.num_v):
            root = uf._find(node)
            if root not in components_dict:
                components_dict[root] = []
            components_dict[root].append(node)

        components = []
        for nodes in components_dict.values():
            #component = Graph(len(nodes))
            self.hash_v=defaultdict(int)
            e_list=[]
            num_v=len(nodes)
            for u in nodes:
                self._remap_v_(u)
                for v in graph.nbr_v(u):
                #v=graph.e_dst.tolist()[node_map[u]]
                    if u < v:
                        self._remap_v_(v)
                        e_list.append((self.hash_v[u],self.hash_v[v]))
            #  remap vertex 
            g=Graph(num_v=num_v,e_list=e_list)
            components.append(g)

        return components
    def _remap_v_(self,v):
        if v not in self.hash_v:
            self.hash_v[v]=len(self.hash_v)

class NiceTreeDecomposition:
    INTRODUCE, FORGET, JOIN = 0, 1, 2 # 引入顶点，遗忘，合并

    def __init__(self,nodes=[],root=None):
        self.nodes = nodes
        self.root = root

    def type(self, x):
        return self.nodes[x][0] # type

    def vertex(self, x):
        return self.nodes[x][1] # vertex idx

    def child(self, x):
        return self.nodes[x][2] # child

    def left(self, x):
        return self.nodes[x][1] # left

    def right(self, x):
        return self.nodes[x][2] # right

    def is_leaf(self, x):
        return self.type(x) == self.INTRODUCE and self.child(x) == -1
    # 没有任何子节点，其袋是空的，或者只有一个子节点
    def is_introduce(self, x):
        return self.type(x) == self.INTRODUCE 
    # 引入节点，节点有一个子节点，其袋比子节点的袋多一个顶点。
    def is_join(self, x):
        return self.type(x) == self.JOIN
    # 结合节点，节点有两个子节点，其袋于子节点的袋相同，表示将两个子树结合起来，共享相同的袋
    def is_forget(self, x):
        return self.type(x) == self.FORGET
    # 忘记节点，节点有一个子节点，其袋比子节点的袋少一个顶点。
    def display(self):
        self._display(self.root, 0)

    def _display(self, x, tab):
        if x == -1 or x is None:
            return
        if self.type(x) == self.INTRODUCE:
            print(f"{' ' * tab}{x}: Introduce {self.vertex(x)}")
            self._display(self.child(x), tab + 2)
        elif self.type(x) == self.FORGET:
            print(f"{' ' * tab}{x}: Forget {self.vertex(x)}")
            self._display(self.child(x), tab + 2)
        elif self.type(x) == self.JOIN:
            print(f"{' ' * tab}{x}: Join")
            self._display(self.left(x), tab + 2)
            self._display(self.right(x), tab + 2)

def set_difference(A,B):
    return sorted(set(A)-set(B))

def set_union(A,B):
    return sorted(set(A).union(set(B)))

'''
图论中，消除顺序elimination ordering指的是确定一个顶点的顺序，逐个消除这些顶点，并简化图结构。
贪心步骤：
1.选择顶点v
2.消除该顶点，并消除改点相邻的所有顶点的连边，称为“完全化邻居”或“添加填充边”
3.更新图，更新后图不在包含v，且该点所有邻居形成一个clique
4.迭代1，直至图中没有顶点
'''

def nice_tree_decomposition(G:Graph):
    n=G.num_v
    nbh=[sorted(set(sorted(G.nbr_v(u))+[u])) for u in range(n)]
    # 贪心规则-消除顺序 elimination ordering
    order,X=[],[]
    pq=[(len(nbh[u]),u) for u in range(n)]
    heapq.heapify(pq)
    
    # BFS
    while pq:
        deg,u=heapq.heappop(pq) # deg, u
        if deg!=len(nbh[u]):
            continue
        order.append(u)
        is_maximal=all(set_difference(nbh[u],x) for x in X) # nbh[u] u 的邻居
        if is_maximal:
            X.append(nbh[u]) #极大团，或者极大子集
        for v in nbh[u]:
            if u==v:
                continue
            U=set_union(nbh[u],nbh[v])
            U.remove(u)
            nbh[v]=U
            if len(nbh[v])<len(U):
                heapq.heappush(pq,(len(U),v))
    
    # From elimination ordering to nice tree decomposition
    NTD=NiceTreeDecomposition()
    children=[[] for _ in range(len(X))]
    head=[-1]*len(X)
    
    for i in range(len(X)):
        for j in children[i]:
            for u in set_difference(X[j],X[i]):
                NTD.nodes.append((NiceTreeDecomposition.FORGET,u,head[j]))
                head[j]=len(NTD.nodes)-1
            for u in set_difference(X[i],X[j]):
                NTD.nodes.append((NiceTreeDecomposition.INTRODUCE,u,head[j]))
                head[j]=len(NTD.nodes)-1
            if head[i]==-1:
                head[i]=head[j]
            else:
                NTD.nodes.append((NiceTreeDecomposition.JOIN,head[i],head[j]))
                head[i]=len(NTD.nodes)-1
        if head[i]==-1:
            for u in X[i]:
                NTD.nodes.append((NiceTreeDecomposition.INTRODUCE,u,head[i]))
                head[i]=len(NTD.nodes)-1
        for j in range(i+1,len(X)):
            if len(set_difference(X[i],X[j]))==1:
                children[j].append(i)
                break
    for u in X[-1]:
        NTD.nodes.append((NiceTreeDecomposition.FORGET,u,head[-1]))
        head[-1]=len(NTD.nodes)-1
    NTD.root=head[-1]
    return NTD

class HomomorphismCounting:
    def __init__(self,F:dhg.Graph,G:dhg.Graph) -> None:
        self._F=F
        self._G=G
        self.NTD=nice_tree_decomposition(F)
    class VectorHash:
        def __call__(self,x):
            p=int(1e9+7)
            hash_val=0
            for i in x:
                hash_val=p*hash_val+i
            return hash_val
    def run(self):
        I, X = self._run(self.NTD.root)
        return I[tuple()]#sum(I.values())  # Empty list in Python is an empty tuple
    def _run(self,x):
        I,J,K,X={},{},{},[]
        assert isinstance(self.NTD,NiceTreeDecomposition)
        if self.NTD.is_leaf(x):
            X=[self.NTD.vertex(x)]
            for a in range(self._G.num_v):
                I[(a,)]=1
        elif self.NTD.is_introduce(x):
            J,X=self._run(self.NTD.child(x))
            p=np.searchsorted(X,self.NTD.vertex(x)) # 查找插入位置
            candidate_id,i=[],0
            for v in sorted(self._F.nbr_v(self.NTD.vertex(x))):
                while i<len(X) and X[i]<v:
                    i+=1
                if i==len(X):
                    break
                if X[i]==v:
                    candidate_id.append(i)
            if not candidate_id:
                candidate_id=list(range(self._G.num_v))
            for phi,val in J.items():
                candidate_a=[]
                candidate_id.sort(key=lambda i: len(self._G.nbr_v(i)))
                for i in candidate_id:
                    a=phi[i]
                    if not candidate_a:
                        candidate_a=sorted(self._G.nbr_v(a))
                    else:
                        candidate_a=list(set(candidate_a).intersection(set(sorted(self._G.nbr_v(a)))))
                psi=list(phi)
                psi.insert(p,0)
                for a in candidate_a:
                    psi[p]=a
                    I[tuple(psi)]=val
            X.insert(p,self.NTD.vertex(x))
            
        elif self.NTD.is_forget(x):
            J,X = self._run(self.NTD.child(x))
            p=np.searchsorted(X,self.NTD.vertex(x))
            X.pop(p)
            for phi,val in J.items():
                psi=list(phi)
                psi.pop(p)
                I[tuple(psi)]=I.get(tuple(psi),0)+val
        elif self.NTD.is_join(x):
            J,X=self._run(self.NTD.left(x))
            K,X=self._run(self.NTD.right(x))
            for phi,val in J.items():
                if phi in K:
                    I[phi]=val*K[phi]
        return I,X


    
class HomomorphismCountingTree:
    # hom_tree count 
    def __init__(self, F:Graph, G:Graph):
        self.F = F
        self.G = G

    def run(self):
        hom_r = self.run_recursive(0, -1)
        return sum(hom_r)

    def run_recursive(self, x, p):
        hom_x = [1] * self.G.num_v
        for y in self.F.nbr_v(x):
            if y == p:
                continue
            hom_y = self.run_recursive(y, x)
            for a in range(self.G.num_v):
                hom_x[a] *= sum(hom_y[b] for b in sorted(self.G.nbr_v(a)))
        return hom_x
    
def hom(F:dhg.Graph,G:dhg.Graph,density=False):
    model=GraphHomo()
    def hom_(F:dhg.Graph, G:dhg.Graph):
        if model.is_tree(F):
            return HomomorphismCountingTree(F, G).run()
        else:
            return HomomorphismCounting(F, G).run()
    if density is True:
        scaler=1.0/(G.num_v**F.num_v)
    else:
        scaler=1.0
    #-------#
    value = scaler
    for Fi in model.connected_components(F):
        value_i = 0
        for Gj in model.connected_components(G):
            value_i += hom_(Fi, Gj)
        value *= value_i
    #-------#
    return value

def test_count_vertex():
    F = Graph(1,[(0,0)])
    G = Graph(4,[(0,1),(0,2),(1,3),(1,2)])
    assert(hom(F,G)==G.num_v)
    print("Number of homomorphisms:", hom(F, G))  
def test_count_edge():
    F = Graph(2,[(0,1)])
    G = Graph(4,[(0,1),(0,2),(1,3),(1,2)])
    assert(hom(F,G)==2*G.num_e)
    print("Number of homomorphisms:", hom(F, G))  
def test_count_triangle():
    F = Graph(4,[(0,1),(1,2),(3,2),(3,0)])
    G = Graph(4,[(0,1),(1,2),(2,3),(3,0)])
    print("Number of triangle:",count_triangles(G))
    #assert(hom(F,G)%count_triangles(G)==0)
    print("Number of homomorphisms:", hom(F, G))

def count_triangles(G:dhg.Graph):
    triangles = 0
    for u in range(G.num_v):
        for v in G.nbr_v(u):
            if v < u:
                continue
            common = set(G.nbr_v(u)).intersection(G.nbr_v(v))
            triangles += sum(1 for a in common if a > v)
    
    return triangles

def test_simple_hom():
    # test_count_vertex()
    # test_count_edge()
    test_count_triangle()

    
##---------------------------------------------## 
import networkx as nx
def tree_list(size=6,num_loops=0):
    '''generate nonisomorphic trees up to k size'''
    t_list=[Graph(num_v=i,e_list=list(tree.edges())) for i in range(2,size+1) for tree in \
        nx.generators.nonisomorphic_trees(i)]
    return t_list
def cycle_list(size=6):
    '''generate undirected cycle up to k size'''
    c_list=[]
    for i in range(2,size+1):
        e_list=[]
        for j in range(1,i):
            # if j==i:
            #     e_list.append((j,0))    
            e_list.append((j-1,j))
        e_list.append((j,0))
        print(e_list)
        c_list.append(Graph(num_v=i,e_list=e_list))
    return c_list
def trick_cycle_list(size=6):
    c_list=[]
    for i in range(2,size+1):
        e_list=[]
        for j in range(1,i):
            # if j==i-1:
            #     e_list.append((j,0))
            e_list.append((j-1,j))
        c_list.append(Graph(num_v=i,e_list=e_list))
    return c_list
def path_list(size=6):
    p_list=[Graph(num_v=i,e_list=list(tree.edges())) for i in range(2,size+1) for tree in \
        nx.generators.path_graph(i)]
    return p_list
##---------------------------------------------##
def tree_profile(G,size=6,density=False):
    t_list=tree_list(size=size,num_loops=0)
    return [hom(t,G,density=density) for t in t_list]

def cycle_profile(G,size=6,density=False):
    c_list=cycle_list(size=size)
    return [hom(c,G,density=density) for c in c_list]


##---------------------------------------------##

class GHC:
    def __init__(self,normalize=True,types='t',p=1,size=6) -> None:
        assert(types in ['t','c','tc']) # tree and cycle 
        self.types=types
        self.normalize=normalize
        self.size=size
        self._p=p
        if self.types == 't':
            self.pre_g=tree_list(size=self.size)
        elif self.types == 'c':
            self.pre_g=trick_cycle_list(size=self.size)
        elif self.types =='tc':
            self.pre_g=tree_list(size=self.size)+trick_cycle_list(size=self.size)
    def fit_transform(self,g_list):
        self.feature_g=torch.zeros(len(g_list),len(self.pre_g)) # feature_g
        for g_idx,g in enumerate(g_list):
            if isinstance(g['dhg'],dhg.Graph):
                self.feature_g[g_idx,:]=torch.log(torch.tensor([hom(preg,g['dhg'],density=False) for preg in self.pre_g]))
            else:
                self.feature_g[g_idx,:]=torch.log(torch.tensor([hom(preg,g['dhgg'],density=False) for preg in self.pre_g]))                
        print('hom for train is ok!')
        self.train_ft=torch.cdist(self.feature_g,self.feature_g,p=self._p)
        sigma=torch.median(self.train_ft)
        self.train_ft=torch.exp(-self.train_ft/sigma)
        return self.train_ft
    
    def transform(self,g_list):
        feature_g=torch.zeros(len(g_list),len(self.pre_g)) # feature_g
        for g_idx,g in enumerate(g_list):
            if isinstance(g['dhg'],dhg.Graph):
                feature_g[g_idx,:]=torch.log(torch.tensor([hom(preg,g['dhg'],density=False) for preg in self.pre_g]))
            else:
                feature_g[g_idx,:]=torch.log(torch.tensor([hom(preg,g['dhgg'],density=False) for preg in self.pre_g]))        
        print('hom for test is ok!') 
        test_ft=torch.cdist(feature_g,self.feature_g,p=self._p)
        sigma=torch.median(test_ft)
        test_ft=torch.exp(-test_ft/sigma)
        return test_ft
   


if __name__ == "__main__":
    G = Graph(4,[(0,1),(0,2),(1,3),(1,2)])
    #print(tree_profile(G=G,size=6,density=False))
    #print(tree_list(size=4,num_loops=0))
    #print(cycle_list(size=6))
    print(test_count_triangle())

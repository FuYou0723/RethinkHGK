import dhg
from dhg import Graph
import heapq
import numpy as np
from functools import lru_cache
from itertools import permutations
from functools import cmp_to_key
from collections import defaultdict
class HomomorphismCounting:
    def __init__(self, F, G):
        self.F = F
        self.G = G
        
    def run(self):
        I, _ = self.run_recursive(self.F, self.G)
        return sum(I.values())#I[()]

    @lru_cache(None)
    def run_recursive(self, F:Graph, G:Graph):
        I = defaultdict(int)
        if F.num_v == 1:
            for a in range(G.num_v):
                I[(a,)] = 1
            return I, [0]

        for perm in permutations(range(G.num_v), F.num_v):
            valid = True
            for u in range(F.num_v):
                for v in F.nbr_v(u):
                    if perm[v] not in sorted(G.nbr_v(perm[u])):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                phi = tuple(perm)
                I[phi] = 1

        return I, list(range(F.num_v))
class HomomoprhismCountingTree:
    def __init__(self,F:dhg.Graph,G:dhg.Graph) -> None:
        self._F=F
        self._G=G
    def run(self):
        hom_r=self._run(0,-1)
        return sum(hom_r)
    def _run(self,x,p):
        hom_x=[1]*self._G.num_v
        for y in sorted(self._F.nbr_v(x)):
            if y==p:
                continue
            hom_y=self._run(y,x)
            for a in range(self._G.num_v):
                sum_y=sum(hom_y[b] for b in sorted(self._G.nbr_v(a)))
                hom_x[a]*=hom_y
        return hom_x
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
    

def hom(F:dhg.Graph,G:dhg.Graph,density=False):
    model=GraphHomo()
    def hom_(F:dhg.Graph, G:dhg.Graph):
        if model.is_tree(F):
            print('is tree')
            return HomomoprhismCountingTree(F, G).run()
        else:
            print('not tree')
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

if __name__ == "__main__":
    # G = Graph(4,[(0,1),(0,2),(1,3),(1,2)])
    # #print(tree_profile(G=G,size=6,density=False))
    # #print(tree_list(size=4,num_loops=0))
    # #print(cycle_list(size=6))
    # print(test_count_vertex())
    F=Graph(4,[(0,1),(1,2),(2,3),(3,0)])
    G=Graph(4,[(0,1),(1,2),(2,3),(3,0),(3,1)])    
    print(f"hom(F,G) is {hom(F,G)}")
    
####--------------------------------------------#### 
'''
count cycle via dfs
'''
def count_cycles_of_length_k(g:dhg.Graph,k):
    visited=[False]*g.num_v
    path=[]
    cycle_count=0
    def dfs(start,current,visited,path,k):
        # g:dhg.Graph
        if len(path)==k:
            # 判断是否成环
            if path[0]==current:
                return 1
            else:
                return 0
        visited[current]=True
        path.append(current)
        cycle_count=0
        # 遍历所有相邻节点
        for nbv in g.nbr_v(current):
            if not visited[nbv]:
                cycle_count+=dfs(start,nbv,visited,path,k)
            elif nbv==start and len(path)==k-1:
                # 如果邻居点是起始点，则成环
                cycle_count+=1
        # 回溯
        visited[current]=False
        path.pop()
        return cycle_count
    for v in range(g.num_v):
        cycle_count+=dfs(v,v,visited,path,k)
        visited[v]=True
    return cycle_count//2
    

# 示例：创建一个图并查找长度为k的环
g = dhg.Graph(5,[(0,1),(1,2),(2,3),(3,4),(4,0),(3,1)])  # 假设有5个顶点


k = 4  # 查找长度为4的环
'''
时间复杂度是O(n(n-1)^{k-1})
'''
print("Number of cycles of length", k, ":", count_cycles_of_length_k(g,k))
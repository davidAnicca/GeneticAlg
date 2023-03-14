import networkx as gf
import random
import matplotlib.pyplot as plt

def run(k):
    G = gf.Graph()

    G.add_nodes_from(range(100))
    com = random.randint(4, 10)

    for offset in range(com):
        for i in range(500//com):
            u = random.randint((100//com)*offset, ((100//com)*(offset+1))-1)  
            v = random.randint((100//com)*offset-1, ((100//com)*(offset+1))-1)  
            if u != v: 
             G.add_edge(u, v)

    gf.write_gml(G, str(k+1) + ".gml")
    return G

def test(G):
   assert(len(G.edges) <= 500)
   assert(len(G.nodes) == 101)

for i in range(10):
    G = run(i)
    test(G)

